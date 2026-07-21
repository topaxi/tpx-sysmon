use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::watch;
use tracing::debug;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CpuState {
    pub usage: f64, // 0.0 - 100.0
}

pub async fn run(tx: watch::Sender<CpuState>) -> Result<()> {
    let mut prev = read_stat().unwrap_or_default();

    loop {
        tokio::time::sleep(Duration::from_secs(2)).await;

        if let Some(curr) = read_stat() {
            let total_delta = curr.total().saturating_sub(prev.total());
            let idle_delta = curr.idle.saturating_sub(prev.idle);

            let usage = if total_delta > 0 {
                (1.0 - idle_delta as f64 / total_delta as f64) * 100.0
            } else {
                0.0
            };

            debug!("CPU usage: {usage:.1}%");
            tx.send_replace(CpuState {
                usage: usage.clamp(0.0, 100.0),
            });
            prev = curr;
        }
    }
}

#[derive(Default, Clone)]
struct CpuTimes {
    user: u64,
    nice: u64,
    system: u64,
    idle: u64,
    iowait: u64,
    irq: u64,
    softirq: u64,
    steal: u64,
}

impl CpuTimes {
    fn total(&self) -> u64 {
        self.user
            + self.nice
            + self.system
            + self.idle
            + self.iowait
            + self.irq
            + self.softirq
            + self.steal
    }
}

fn read_stat() -> Option<CpuTimes> {
    let content = std::fs::read_to_string("/proc/stat").ok()?;
    let line = content.lines().next()?;
    let mut parts = line.split_whitespace();
    parts.next(); // "cpu"

    Some(CpuTimes {
        user: parts.next()?.parse().ok()?,
        nice: parts.next()?.parse().ok()?,
        system: parts.next()?.parse().ok()?,
        idle: parts.next()?.parse().ok()?,
        iowait: parts.next()?.parse().unwrap_or(0),
        irq: parts.next()?.parse().unwrap_or(0),
        softirq: parts.next()?.parse().unwrap_or(0),
        steal: parts.next()?.parse().unwrap_or(0),
    })
}
