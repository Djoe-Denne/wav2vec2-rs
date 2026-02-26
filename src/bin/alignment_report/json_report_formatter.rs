use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use wav2vec2_rs::Report;

pub fn write_report(path: &Path, report: &Report) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create report output directory '{}': {err}",
                parent.display()
            )
        })?;
    }

    let mut file = File::create(path)
        .map_err(|err| format!("Failed to create report file '{}': {err}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, report).map_err(|err| {
        format!(
            "Failed to serialize report JSON '{}': {err}",
            path.display()
        )
    })?;
    file.write_all(b"\n")
        .map_err(|err| format!("Failed to finalize report file '{}': {err}", path.display()))?;
    Ok(())
}
