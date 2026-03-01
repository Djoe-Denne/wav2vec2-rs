//! Per-stage memory profiling for the alignment pipeline (benchmark mode only).
//!
//! Measures peak CPU RSS and optional GPU memory during each pipeline stage.
//! **GPU is asynchronous**: peak must be read after device sync; otherwise GPU work
//! may not be reflected. This module runs the sync before reading GPU.

#![cfg(feature = "alignment-profiling")]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::error::AlignmentError;

/// Snapshot of GPU memory at a point in time (used and total device memory in bytes).
/// Use after device sync for accurate "after stage" readings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct GpuMemorySnapshot {
    pub gpu_used: u64,
    pub gpu_total: u64,
}

/// Returns current GPU memory (used, total) in bytes, or None if CUDA is not available.
/// Prefer calling after synchronize() so async GPU work is reflected.
/// Uses CUDA Driver API (cuMemGetInfo_v2); device-level view may require Runtime API
/// (cudaMemGetInfo) when multiple contexts exist (e.g. ORT + cudarc).
#[cfg(feature = "cuda-dp")]
pub fn gpu_memory_snapshot() -> Option<GpuMemorySnapshot> {
    use cudarc::driver::sys::{cuMemGetInfo_v2, CUresult};
    let mut free: usize = 0;
    let mut total: usize = 0;
    let err = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
    if err == CUresult::CUDA_SUCCESS {
        let used = total.saturating_sub(free);
        Some(GpuMemorySnapshot {
            gpu_used: used as u64,
            gpu_total: total as u64,
        })
    } else {
        None
    }
}

#[cfg(not(feature = "cuda-dp"))]
pub fn gpu_memory_snapshot() -> Option<GpuMemorySnapshot> {
    None
}

/// Peak memory observed during a single pipeline stage (all values in bytes).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageMemory {
    pub peak_cpu_rss_bytes: u64,
    pub peak_gpu_allocated_bytes: u64,
    pub peak_gpu_reserved_bytes: u64,
    /// Total device memory in bytes (0 when no GPU). Same for all stages on a run.
    pub gpu_total_bytes: u64,
}

/// Per-stage memory for the full pipeline (benchmark mode).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StageMemoryMap {
    pub forward: StageMemory,
    pub post: StageMemory,
    pub dp: StageMemory,
    pub group: StageMemory,
    pub conf: StageMemory,
}

/// Optional callback to read GPU memory (used, total) in bytes after sync.
/// Called on the main thread after synchronize(); return (0, 0) if unavailable.
pub type GpuReader = Box<dyn Fn() -> (u64, u64) + Send>;

/// When `cuda-dp` is enabled, returns a GPU reader that reports (used, total) device memory.
/// Uses gpu_memory_snapshot(); caller should use this when the backend device is CUDA.
#[cfg(feature = "cuda-dp")]
pub fn cuda_gpu_reader() -> Option<GpuReader> {
    Some(Box::new(|| {
        gpu_memory_snapshot()
            .map(|s| (s.gpu_used, s.gpu_total))
            .unwrap_or((0, 0))
    }))
}

#[cfg(not(feature = "cuda-dp"))]
pub fn cuda_gpu_reader() -> Option<GpuReader> {
    None
}

/// Tracks peak process RSS and optional GPU memory during a stage.
/// Start sampler, run closure, sync device, then read peaks.
pub struct MemoryTracker {
    gpu_reader: Option<GpuReader>,
    /// Sampler interval (e.g. 5 ms).
    sample_interval_ms: u64,
}

impl MemoryTracker {
    /// Builds a tracker. Use `measure` with a sync closure so the device is synchronized
    /// after each stage before reading GPU peak. When `gpu_reader` is None, GPU bytes are 0.
    pub fn new(gpu_reader: Option<GpuReader>) -> Self {
        Self {
            gpu_reader,
            sample_interval_ms: 5,
        }
    }

    /// Runs `f`, then calls `sync_fn` (device sync), then returns the result and peak memory.
    /// GPU is asynchronous: sync must run before reading GPU so work is reflected.
    /// No allocation in the sampling loop.
    pub fn measure<T, F, S>(
        &mut self,
        _stage_name: &str,
        sync_fn: S,
        f: F,
    ) -> Result<(T, StageMemory), AlignmentError>
    where
        F: FnOnce() -> Result<T, AlignmentError>,
        S: FnOnce() -> Result<(), AlignmentError>,
    {
        let max_rss = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let max_rss_clone = Arc::clone(&max_rss);
        let stop_clone = Arc::clone(&stop);
        let interval_ms = self.sample_interval_ms;

        let sampler = thread::spawn(move || {
            while !stop_clone.load(Ordering::Relaxed) {
                if let Some(rss) = current_process_rss_bytes() {
                    let mut prev = max_rss_clone.load(Ordering::Relaxed);
                    while rss > prev {
                        match max_rss_clone.compare_exchange_weak(
                            prev,
                            rss,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(p) => prev = p,
                        }
                    }
                }
                thread::sleep(Duration::from_millis(interval_ms));
            }
        });

        let result = f()?;

        // GPU is asynchronous: measurements must be taken after device sync so that
        // all work for this stage is complete and allocations are visible.
        sync_fn()?;

        stop.store(true, Ordering::Relaxed);
        let _ = sampler.join();

        let peak_cpu_rss_bytes = max_rss.load(Ordering::Relaxed);

        let (peak_gpu_allocated_bytes, gpu_total_bytes) = self
            .gpu_reader
            .as_ref()
            .map(|r| r())
            .unwrap_or((0, 0));

        Ok((
            result,
            StageMemory {
                peak_cpu_rss_bytes,
                peak_gpu_allocated_bytes,
                peak_gpu_reserved_bytes: 0, // not exposed by driver API
                gpu_total_bytes,
            },
        ))
    }
}

/// Returns current process RSS in bytes, or None on unsupported/error.
fn current_process_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        read_linux_rss()
    }

    #[cfg(target_os = "windows")]
    {
        read_windows_rss()
    }

    #[cfg(target_os = "macos")]
    {
        read_macos_rss()
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        let _ = ();
        None
    }
}

#[cfg(target_os = "linux")]
fn read_linux_rss() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let rest = line.trim_start_matches("VmRSS:").trim();
            let num: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(num.saturating_mul(1024));
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn read_windows_rss() -> Option<u64> {
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    unsafe {
        let process: HANDLE = GetCurrentProcess();
        let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        if GetProcessMemoryInfo(process, &mut counters, counters.cb) != 0 {
            Some(counters.WorkingSetSize as u64)
        } else {
            None
        }
    }
}

#[cfg(target_os = "macos")]
fn read_macos_rss() -> Option<u64> {
    use libc::{proc_pidinfo, PROC_PIDTASKINFO};
    use std::mem::size_of;

    let pid = std::process::id() as i32;
    let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = unsafe {
        proc_pidinfo(
            pid,
            PROC_PIDTASKINFO,
            0,
            &mut info as *mut _ as *mut libc::c_void,
            size_of::<libc::proc_taskinfo>() as u32,
        )
    };
    if size == size_of::<libc::proc_taskinfo>() as i32 {
        Some(info.pti_resident_size)
    } else {
        None
    }
}
