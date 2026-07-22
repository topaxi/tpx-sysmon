use anyhow::Result;
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::watch;
use tracing::{debug, warn};

use crate::gpu::{GpuConfig, GpuInfo, GpuProvider, GpuState, model};

pub async fn run(tx: watch::Sender<GpuState>, configs: Vec<GpuConfig>, poll_ms: u64) -> Result<()> {
    if configs.is_empty() {
        debug!("No NVIDIA GPU configs, skipping");
        std::future::pending::<()>().await;
        return Ok(());
    }

    let models = model::resolve_all(configs.iter().map(|c| (c.id.as_str(), c.provider)));

    let interval = Duration::from_millis(poll_ms);
    loop {
        match poll_nvidia(&configs, &models).await {
            Ok(state) => {
                tx.send_replace(state);
            }
            Err(e) => warn!("nvidia-smi error: {e}"),
        }
        tokio::time::sleep(interval).await;
    }
}

async fn poll_nvidia(configs: &[GpuConfig], models: &HashMap<String, String>) -> Result<GpuState> {
    let fields = "pci.bus_id,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw";
    let output = tokio::process::Command::new("nvidia-smi")
        .args([
            &format!("--query-gpu={fields}"),
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let gpus = stdout
        .lines()
        .filter_map(|line| parse_gpu_line(line, configs, models))
        .collect();

    Ok(GpuState { gpus })
}

/// Parse one `--format=csv,noheader,nounits` row into a `GpuInfo`, matching it
/// to a configured GPU by PCI bus id. Returns `None` for malformed rows or GPUs
/// that aren't in `configs`.
fn parse_gpu_line(
    line: &str,
    configs: &[GpuConfig],
    models: &HashMap<String, String>,
) -> Option<GpuInfo> {
    let parts: Vec<&str> = line.split(", ").collect();
    if parts.len() < 6 {
        return None;
    }

    // nvidia-smi BDF is like "00000000:01:00.0"; config id is "0000:01:00.0".
    // Uppercasing the config id normalises the hex digits in either casing.
    let raw_bdf = parts[0].trim();
    let config = configs
        .iter()
        .find(|c| raw_bdf.ends_with(&c.id.to_uppercase()))?;

    debug!(
        "GPU {}: {}% {:.1}°C",
        config.id,
        parts[1].trim(),
        parts[4].trim().parse::<f64>().unwrap_or(0.0)
    );

    Some(GpuInfo {
        id: config.id.clone(),
        label: "dGPU".to_string(),
        model: models.get(&config.id).cloned().unwrap_or_else(|| config.id.clone()),
        provider: GpuProvider::Nvidia,
        gpu_usage: parts[1].trim().parse::<f64>().unwrap_or(0.0),
        mem_used: parts[2].trim().parse::<u64>().unwrap_or(0) * 1024 * 1024,
        mem_total: parts[3].trim().parse::<u64>().unwrap_or(0) * 1024 * 1024,
        gtt_used: 0,
        gtt_total: 0,
        temperature: parts[4].trim().parse::<f64>().unwrap_or(0.0),
        power_watts: parts[5].trim().parse::<f64>().unwrap_or(0.0),
        in_bar: config.bar,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(id: &str) -> GpuConfig {
        GpuConfig {
            id: id.to_string(),
            provider: GpuProvider::Nvidia,
            gpu_temp_sensors: vec![],
            bar: true,
        }
    }

    #[test]
    fn parses_row_and_matches_bdf_case_insensitively() {
        let configs = vec![config("0000:01:00.0")];
        let models = HashMap::from([("0000:01:00.0".to_string(), "GeForce RTX 4090".to_string())]);
        let line = "00000000:01:00.0, 37, 2048, 8192, 55, 120.50";
        let info = parse_gpu_line(line, &configs, &models).unwrap();

        assert_eq!(info.id, "0000:01:00.0");
        assert_eq!(info.model, "GeForce RTX 4090");
        assert_eq!(info.provider, GpuProvider::Nvidia);
        assert_eq!(info.gpu_usage, 37.0);
        assert_eq!(info.mem_used, 2048 * 1024 * 1024);
        assert_eq!(info.mem_total, 8192 * 1024 * 1024);
        assert_eq!(info.temperature, 55.0);
        assert!((info.power_watts - 120.50).abs() < 1e-9);
    }

    #[test]
    fn matches_hex_bus_id_regardless_of_case() {
        let configs = vec![config("0000:0a:00.0")];
        let line = "00000000:0A:00.0, 5, 1, 2, 40, 10";
        assert!(parse_gpu_line(line, &configs, &HashMap::new()).is_some());
    }

    #[test]
    fn skips_unconfigured_gpu() {
        let configs = vec![config("0000:01:00.0")];
        let line = "00000000:02:00.0, 37, 2048, 8192, 55, 120.50";
        assert!(parse_gpu_line(line, &configs, &HashMap::new()).is_none());
    }

    #[test]
    fn na_fields_default_to_zero() {
        let configs = vec![config("0000:01:00.0")];
        // power.draw reports "[N/A]" on GPUs without a power sensor.
        let line = "00000000:01:00.0, [N/A], [N/A], 8192, 55, [N/A]";
        let info = parse_gpu_line(line, &configs, &HashMap::new()).unwrap();
        assert_eq!(info.gpu_usage, 0.0);
        assert_eq!(info.mem_used, 0);
        assert_eq!(info.power_watts, 0.0);
        assert_eq!(info.mem_total, 8192 * 1024 * 1024);
    }

    #[test]
    fn malformed_row_is_skipped() {
        let configs = vec![config("0000:01:00.0")];
        assert!(parse_gpu_line("00000000:01:00.0, 37, 2048", &configs, &HashMap::new()).is_none());
    }
}
