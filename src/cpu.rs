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
    let mut prev = read_stat().await.unwrap_or_default();

    loop {
        tokio::time::sleep(Duration::from_secs(2)).await;

        if let Some(curr) = read_stat().await {
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

async fn read_stat() -> Option<CpuTimes> {
    tokio::task::spawn_blocking(|| {
        let content = std::fs::read_to_string("/proc/stat").ok()?;
        parse_stat(&content)
    })
    .await
    .ok()
    .flatten()
}

fn parse_stat(content: &str) -> Option<CpuTimes> {
    let line = content.lines().next()?;
    let mut parts = line.split_whitespace();
    parts.next(); // "cpu"

    Some(CpuTimes {
        user: parts.next()?.parse().ok()?,
        nice: parts.next()?.parse().ok()?,
        system: parts.next()?.parse().ok()?,
        idle: parts.next()?.parse().ok()?,
        iowait: parts.next()?.parse().ok()?,
        irq: parts.next()?.parse().ok()?,
        softirq: parts.next()?.parse().ok()?,
        steal: parts.next()?.parse().ok()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_proc_stat_first_line() {
        let content = "cpu  4705 0 1235 98452 12 0 34 0 0 0\n\
            cpu0 100 0 50 2000 1 0 2 0 0 0\n";
        let t = parse_stat(content).unwrap();
        assert_eq!(t.user, 4705);
        assert_eq!(t.idle, 98452);
        assert_eq!(t.steal, 0);
        assert_eq!(t.total(), 4705 + 1235 + 98452 + 12 + 34);
    }

    #[test]
    fn rejects_truncated_lines() {
        assert!(parse_stat("cpu 1 2 3 4\n").is_none());
        assert!(parse_stat("cpu 1 2\n").is_none());
        assert!(parse_stat("").is_none());
    }

    #[test]
    fn usage_from_deltas() {
        // 50% idle over the window -> 50% usage.
        let prev = parse_stat("cpu 100 0 100 800 0 0 0 0\n").unwrap();
        let curr = parse_stat("cpu 150 0 150 900 0 0 0 0\n").unwrap();
        let total_delta = curr.total() - prev.total();
        let idle_delta = curr.idle - prev.idle;
        let usage = (1.0 - idle_delta as f64 / total_delta as f64) * 100.0;
        assert_eq!(usage, 50.0);
    }
}
