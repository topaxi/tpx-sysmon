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
    /// zswap compressed-pool RAM cost (`Zswap` in /proc/meminfo). `None` when
    /// the kernel does not expose the field (zswap not built in).
    #[serde(default)]
    pub zswap_bytes: Option<u64>,
    /// Uncompressed size of the pages held in zswap (`Zswapped`).
    #[serde(default)]
    pub zswapped_bytes: Option<u64>,
    /// Aggregated zram device stats. `None` when no `zram*` device exists.
    #[serde(default)]
    pub zram: Option<ZramStats>,
}

/// Aggregated stats across all `/sys/block/zram*` devices. All values are in
/// bytes (zram's `mm_stat`/`disksize` are already byte-denominated, unlike
/// /proc/meminfo which is kB).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ZramStats {
    /// RAM actually consumed by the compressed data (`mm_stat` field 2,
    /// `mem_used_total`) - the true cost against physical memory.
    pub mem_used_bytes: u64,
    /// Compressed size of the stored data (`mm_stat` field 1).
    pub compr_data_bytes: u64,
    /// Original, uncompressed size of the stored data (`mm_stat` field 0).
    pub orig_data_bytes: u64,
    /// Configured device size (`disksize`) - the logical swap capacity.
    pub disk_size_bytes: u64,
}

pub async fn run(tx: watch::Sender<MemoryState>) -> Result<()> {
    loop {
        if let Some(state) = read_meminfo().await {
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

async fn read_meminfo() -> Option<MemoryState> {
    tokio::task::spawn_blocking(|| {
        let content = std::fs::read_to_string("/proc/meminfo").ok()?;
        let mut state = parse_meminfo(&content)?;
        state.zram = read_zram();
        Some(state)
    })
    .await
    .ok()
    .flatten()
}

/// Aggregate stats across every `/sys/block/zram*` device. Returns `None` when
/// no such device exists (the common case on hosts without zram).
fn read_zram() -> Option<ZramStats> {
    let mut stats = ZramStats::default();
    let mut found = false;

    for entry in std::fs::read_dir("/sys/block").ok()?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();

        // Match `zram0`, `zram1`, ... but not `zram` alone or other names.
        if !name.starts_with("zram") || name.len() == 4 || !name[4..].bytes().all(|b| b.is_ascii_digit())
        {
            continue;
        }

        let base = entry.path();

        // mm_stat is whitespace separated; the field count varies by kernel, so
        // index from the left and never rely on trailing fields.
        // [0] orig_data_size [1] compr_data_size [2] mem_used_total (RAM cost).
        let Some(mm_stat) = crate::util::read_trimmed(base.join("mm_stat")) else {
            continue;
        };
        let nums: Vec<u64> = mm_stat
            .split_whitespace()
            .map(|f| f.parse().unwrap_or(0))
            .collect();

        stats.orig_data_bytes += nums.first().copied().unwrap_or(0);
        stats.compr_data_bytes += nums.get(1).copied().unwrap_or(0);
        stats.mem_used_bytes += nums.get(2).copied().unwrap_or(0);
        stats.disk_size_bytes += crate::util::read_trimmed(base.join("disksize"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        found = true;
    }

    found.then_some(stats)
}

fn parse_meminfo(content: &str) -> Option<MemoryState> {
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
    let available_kb = *fields.get("MemAvailable")?;

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

    // Zswap/Zswapped are reported in kB like the rest of /proc/meminfo, and are
    // only present when the kernel has zswap built in.
    let zswap_bytes = fields.get("Zswap").map(|kb| kb * 1024);
    let zswapped_bytes = fields.get("Zswapped").map(|kb| kb * 1024);

    Some(MemoryState {
        used_bytes: used,
        total_bytes: total,
        usage,
        swap_used_bytes: swap_used,
        swap_total_bytes: swap_total,
        swap_usage,
        zswap_bytes,
        zswapped_bytes,
        zram: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_used_and_swap() {
        let content = "MemTotal:       16000000 kB\n\
            MemFree:         2000000 kB\n\
            MemAvailable:    8000000 kB\n\
            SwapTotal:       4000000 kB\n\
            SwapFree:        3000000 kB\n";
        let s = parse_meminfo(content).unwrap();
        assert_eq!(s.total_bytes, 16_000_000 * 1024);
        assert_eq!(s.used_bytes, (16_000_000 - 8_000_000) * 1024);
        assert!((s.usage - 0.5).abs() < 1e-9);
        assert_eq!(s.swap_used_bytes, 1_000_000 * 1024);
        assert!((s.swap_usage - 0.25).abs() < 1e-9);
    }

    #[test]
    fn no_swap_reports_zero_usage() {
        let content = "MemTotal: 16000000 kB\nMemAvailable: 8000000 kB\n";
        let s = parse_meminfo(content).unwrap();
        assert_eq!(s.swap_total_bytes, 0);
        assert_eq!(s.swap_usage, 0.0);
    }

    #[test]
    fn missing_required_field_is_none() {
        assert!(parse_meminfo("MemFree: 100 kB\n").is_none());
        assert!(parse_meminfo("MemTotal: 16000000 kB\n").is_none());
    }

    #[test]
    fn parses_zswap_when_present() {
        let content = "MemTotal: 16000000 kB\n\
            MemAvailable: 8000000 kB\n\
            Zswap: 1024 kB\n\
            Zswapped: 4096 kB\n";
        let s = parse_meminfo(content).unwrap();
        assert_eq!(s.zswap_bytes, Some(1024 * 1024));
        assert_eq!(s.zswapped_bytes, Some(4096 * 1024));
    }

    #[test]
    fn omits_zswap_when_absent() {
        let content = "MemTotal: 16000000 kB\nMemAvailable: 8000000 kB\n";
        let s = parse_meminfo(content).unwrap();
        assert_eq!(s.zswap_bytes, None);
        assert_eq!(s.zswapped_bytes, None);
    }
}
