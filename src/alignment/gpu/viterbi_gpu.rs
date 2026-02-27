//! GPU-accelerated CTC Viterbi forced alignment via wgpu compute shaders.
//!
//! Enabled with the `gpu-dp` feature flag. Falls back to CPU Viterbi when
//! the feature is disabled or GPU initialization fails.
//!
//! The shader runs the entire T-step DP in a single dispatch using one
//! workgroup with barrier synchronization — no per-frame launch overhead.

use std::sync::OnceLock;

/// Shared GPU context, initialized once on first use.
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

static GPU_CTX: OnceLock<Option<GpuContext>> = OnceLock::new();

fn get_gpu_context() -> Option<&'static GpuContext> {
    GPU_CTX
        .get_or_init(|| {
            pollster::block_on(async {
                let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::VULKAN
                        | wgpu::Backends::DX12
                        | wgpu::Backends::METAL,
                    ..Default::default()
                });

                let adapter = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await?;

                let (device, queue) = adapter
                    .request_device(
                        &wgpu::DeviceDescriptor {
                            label: Some("viterbi-gpu"),
                            required_features: wgpu::Features::empty(),
                            required_limits: wgpu::Limits::default(),
                            ..Default::default()
                        },
                        None,
                    )
                    .await
                    .ok()?;

                let shader_src = include_str!("viterbi.wgsl");
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("viterbi-shader"),
                    source: wgpu::ShaderSource::Wgsl(shader_src.into()),
                });

                let bind_group_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("viterbi-bgl"),
                        entries: &[
                            // 0: log_probs (read-only storage)
                            bgl_entry(0, true),
                            // 1: tokens (read-only storage)
                            bgl_entry(1, true),
                            // 2: params (uniform)
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // 3: bp (read-write storage)
                            bgl_entry(3, false),
                            // 4: scores (read-write storage)
                            bgl_entry(4, false),
                            // 5: out (read-write storage)
                            bgl_entry(5, false),
                        ],
                    });

                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("viterbi-pl"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

                let pipeline =
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("viterbi-pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &shader,
                        entry_point: Some("viterbi_main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

                Some(GpuContext {
                    device,
                    queue,
                    pipeline,
                    bind_group_layout,
                })
            })
        })
        .as_ref()
}

fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Params struct matching the WGSL layout (16 bytes, uniform-aligned).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    t_len: u32,
    s_len: u32,
    v_len: u32,
    final_floor_state: u32,
}

/// Run CTC Viterbi on GPU. Returns the same `Vec<(usize, usize)>` path as the CPU version.
///
/// Returns `None` if GPU is unavailable — caller should fall back to CPU.
pub fn forced_align_viterbi_gpu(
    log_probs: &[Vec<f32>],
    tokens: &[usize],
) -> Option<Vec<(usize, usize)>> {
    let ctx = get_gpu_context()?;

    let t_len = log_probs.len();
    let s_len = tokens.len();
    if t_len == 0 || s_len == 0 {
        return Some(Vec::new());
    }
    let v_len = log_probs[0].len();

    // --- Flatten log_probs into contiguous T×V buffer ---
    let log_probs_flat: Vec<f32> = log_probs.iter().flat_map(|row| row.iter().copied()).collect();
    let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();

    let params = GpuParams {
        t_len: t_len as u32,
        s_len: s_len as u32,
        v_len: v_len as u32,
        final_floor_state: s_len.saturating_sub(2) as u32,
    };

    let device = &ctx.device;
    let queue = &ctx.queue;

    // --- Create GPU buffers ---
    let buf_log_probs = create_buffer_init(
        device,
        "log_probs",
        bytemuck::cast_slice(&log_probs_flat),
        wgpu::BufferUsages::STORAGE,
    );

    let buf_tokens = create_buffer_init(
        device,
        "tokens",
        bytemuck::cast_slice(&tokens_u32),
        wgpu::BufferUsages::STORAGE,
    );

    let buf_params = create_buffer_init(
        device,
        "params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    );

    let bp_size = (t_len * s_len * 4) as u64; // u32 per entry
    let buf_bp = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bp"),
        size: bp_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let scores_size = (2 * s_len * 4) as u64;
    let buf_scores = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scores"),
        size: scores_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: 8, // 2 × f32
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // --- Staging buffers for readback ---
    let staging_bp = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging-bp"),
        size: bp_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging-out"),
        size: 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- Bind group ---
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("viterbi-bg"),
        layout: &ctx.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_log_probs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_tokens.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_bp.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_scores.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buf_out.as_entire_binding(),
            },
        ],
    });

    // --- Dispatch ---
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("viterbi-enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("viterbi-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1); // single workgroup
    }
    encoder.copy_buffer_to_buffer(&buf_bp, 0, &staging_bp, 0, bp_size);
    encoder.copy_buffer_to_buffer(&buf_out, 0, &staging_out, 0, 8);
    queue.submit(std::iter::once(encoder.finish()));

    // --- Readback ---
    let bp_data = read_buffer(device, &staging_bp, bp_size);
    let out_data = read_buffer(device, &staging_out, 8);

    let bp_u32: &[u32] = bytemuck::cast_slice(&bp_data);
    let out_f32: &[f32] = bytemuck::cast_slice(&out_data);

    let score_last = out_f32[0];
    let score_prev = out_f32[1];

    // --- Backtrack on CPU (O(T), trivial) ---
    let mut s = s_len - 1;
    if s_len >= 2 && score_prev > score_last {
        s = s_len - 2;
    }

    let mut path = Vec::with_capacity(t_len);
    path.push((s, t_len - 1));
    for t in (1..t_len).rev() {
        let step = bp_u32[t * s_len + s];
        s = match step {
            0 => s,
            1 => {
                debug_assert!(s >= 1);
                s - 1
            }
            2 => {
                debug_assert!(s >= 2);
                s - 2
            }
            _ => s,
        };
        path.push((s, t - 1));
    }
    path.reverse();

    Some(path)
}

/// Create a buffer initialized with data.
fn create_buffer_init(
    device: &wgpu::Device,
    label: &str,
    data: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage,
    })
}

/// Blocking readback from a mappable buffer.
fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let slice = buffer.slice(..size);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .expect("GPU readback channel closed")
        .expect("GPU readback failed");
    let data = slice.get_mapped_range().to_vec();
    buffer.unmap();
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_matches_cpu() {
        // Small test case: 5 frames, 3 tokens, vocab=4
        let log_probs = vec![
            vec![-1.0f32, -2.0, -3.0, -0.5],
            vec![-0.5, -1.5, -2.5, -1.0],
            vec![-2.0, -0.3, -1.5, -2.0],
            vec![-1.5, -1.0, -0.5, -2.0],
            vec![-0.8, -1.2, -1.0, -0.3],
        ];
        let tokens = vec![0usize, 3, 1]; // blank=0, 'a'=3, 'b'=1

        let cpu_path = crate::alignment::viterbi::forced_align_viterbi(&log_probs, &tokens);

        if let Some(gpu_path) = forced_align_viterbi_gpu(&log_probs, &tokens) {
            assert_eq!(cpu_path, gpu_path, "GPU path must match CPU path");
        } else {
            eprintln!("GPU not available, skipping test");
        }
    }
}
