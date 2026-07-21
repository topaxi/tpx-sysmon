use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::watch;
use tracing::debug;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryState {
    pub used_bytes: u64,
    pub total_bytes: u64,
    pub usage: f64, // 0.0 - 1.0
    pub swap_used_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_usage: f64, // 0.0 - 1.0
}

pub async fn run(tx: watch::Sender<MemoryState>) -> Result<()> {
    loop {
        if let Some(state) = read_meminfo() {
            debug!(
                "Memory: {:.1}% used, swap {:.1}%",
                state.usage * 100.0,
                state.swap_usage * 100.0
            );
            tx.send_replace(state);
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

fn read_meminfo() -> Option<MemoryState> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;

    let fields: HashMap<&str, u64> = content
        .lines()
        .filter_map(|line| {
            let mut parts = line.splitn(2, ':');
            let key = parts.next()?.trim();
            let val = parts.next()?.split_whitespace().next()?.parse().ok()?;
            Some((key, val))
        })
        .collect();

    let total_kb = *fields.get("MemTotal")?;
    let available_kb = *fields.get("MemAvailable").unwrap_or(&0);

    let total = total_kb * 1024;
    let available = available_kb * 1024;
    let used = total.saturating_sub(available);
    let usage = if total > 0 {
        used as f64 / total as f64
    } else {
        0.0
    };

    let swap_total = *fields.get("SwapTotal").unwrap_or(&0) * 1024;
    let swap_free = *fields.get("SwapFree").unwrap_or(&0) * 1024;
    let swap_used = swap_total.saturating_sub(swap_free);
    let swap_usage = if swap_total > 0 {
        swap_used as f64 / swap_total as f64
    } else {
        0.0
    };

    Some(MemoryState {
        used_bytes: used,
        total_bytes: total,
        usage,
        swap_used_bytes: swap_used,
        swap_total_bytes: swap_total,
        swap_usage,
    })
}
