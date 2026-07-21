use anyhow::Result;
use std::time::Duration;
use tokio::sync::watch;
use tracing::{debug, warn};

use crate::gpu::{GpuConfig, GpuInfo, GpuState};

pub async fn run(tx: watch::Sender<GpuState>, configs: Vec<GpuConfig>, poll_ms: u64) -> Result<()> {
    if configs.is_empty() {
        debug!("No NVIDIA GPU configs, skipping");
        std::future::pending::<()>().await;
        return Ok(());
    }

    let interval = Duration::from_millis(poll_ms);
    loop {
        match poll_nvidia(&configs).await {
            Ok(state) => {
                tx.send_replace(state);
            }
            Err(e) => warn!("nvidia-smi error: {e}"),
        }
        tokio::time::sleep(interval).await;
    }
}

async fn poll_nvidia(configs: &[GpuConfig]) -> Result<GpuState> {
    let fields = "pci.bus_id,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw";
    let output = tokio::process::Command::new("nvidia-smi")
        .args([
            &format!("--query-gpu={fields}"),
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(", ").collect();
        if parts.len() < 6 {
            continue;
        }

        // nvidia-smi BDF is like "00000000:01:00.0"; config id is "0000:01:00.0"
        let raw_bdf = parts[0].trim();
        let config = match configs
            .iter()
            .find(|c| raw_bdf.ends_with(&c.id.to_uppercase()))
        {
            Some(c) => c,
            None => continue,
        };

        debug!(
            "GPU {}: {}% {:.1}°C",
            config.id,
            parts[1].trim(),
            parts[4].trim().parse::<f64>().unwrap_or(0.0)
        );

        gpus.push(GpuInfo {
            id: config.id.clone(),
            label: "dGPU".to_string(),
            provider: "nvidia".to_string(),
            gpu_usage: parts[1].trim().parse::<f64>().unwrap_or(0.0),
            mem_used: parts[2].trim().parse::<u64>().unwrap_or(0) * 1024 * 1024,
            mem_total: parts[3].trim().parse::<u64>().unwrap_or(0) * 1024 * 1024,
            gtt_used: 0,
            gtt_total: 0,
            temperature: parts[4].trim().parse::<f64>().unwrap_or(0.0),
            power_watts: parts[5].trim().parse::<f64>().unwrap_or(0.0),
            in_bar: config.bar,
        });
    }

    Ok(GpuState { gpus })
}
