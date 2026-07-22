use anyhow::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tracing::debug;

use crate::gpu::{GpuConfig, GpuInfo, GpuProvider, GpuState, model};

/// Reads Intel iGPU usage from the DRM `fdinfo` interface (the same source
/// `intel_gpu_top`/`nvtop` use). i915 has no `gpu_busy_percent` file like
/// amdgpu, and deriving usage from `rc6_residency_ms` (time spent in the RC6
/// idle power state) overcounts: it tracks package idle state, not render
/// engine occupancy. `fdinfo` instead exposes per-client cumulative busy
/// nanoseconds per engine (`/proc/<pid>/fdinfo/<fd>`), which is summed across
/// clients and diffed against the previous poll to get render-engine
/// utilization.
///
/// Memory, temperature and power are left at 0: Intel iGPUs share system RAM
/// (nvtop's "memory" for this GPU is just total/used system RAM, not
/// GPU-specific) and this driver exposes neither a hwmon power sensor nor a
/// dedicated thermal zone for the GPU.
pub async fn run(tx: watch::Sender<GpuState>, configs: Vec<GpuConfig>, poll_ms: u64) -> Result<()> {
    if configs.is_empty() {
        debug!("No Intel GPU configs, skipping");
        std::future::pending::<()>().await;
        return Ok(());
    }

    let models = model::resolve_all(configs.iter().map(|c| (c.id.as_str(), c.provider)));

    let interval = Duration::from_millis(poll_ms);
    // Per PCI id, per DRM client-id: last-seen cumulative render-busy ns.
    let mut prev_by_gpu: HashMap<String, HashMap<u64, u64>> = configs
        .iter()
        .map(|c| (c.id.clone(), HashMap::new()))
        .collect();
    let mut prev_time = Instant::now();

    loop {
        let now = Instant::now();
        let elapsed_ns = now.duration_since(prev_time).as_nanos().max(1) as f64;

        let samples = tokio::task::spawn_blocking(scan_fdinfo)
            .await
            .unwrap_or_default();

        let mut gpus = Vec::new();
        for cfg in &configs {
            let current = samples.get(&cfg.id).cloned().unwrap_or_default();
            let prev_map = prev_by_gpu.entry(cfg.id.clone()).or_default();

            // A client seen for the first time contributes no delta this
            // tick (its cumulative counter reflects its entire lifetime, not
            // just this poll window) - only known clients count toward busy time.
            let delta_ns: u64 = current
                .iter()
                .filter_map(|(client_id, &render_ns)| {
                    prev_map
                        .get(client_id)
                        .map(|&prev_ns| render_ns.saturating_sub(prev_ns))
                })
                .sum();

            let gpu_usage = (delta_ns as f64 / elapsed_ns * 100.0).clamp(0.0, 100.0);
            *prev_map = current;

            debug!("GPU {}: {gpu_usage:.0}%", cfg.id);

            gpus.push(GpuInfo {
                id: cfg.id.clone(),
                label: "iGPU".to_string(),
                model: models.get(&cfg.id).cloned().unwrap_or_else(|| cfg.id.clone()),
                provider: GpuProvider::Intel,
                gpu_usage,
                mem_used: 0,
                mem_total: 0,
                gtt_used: 0,
                gtt_total: 0,
                temperature: 0.0,
                power_watts: 0.0,
                in_bar: cfg.bar,
            });
        }

        prev_time = now;
        tx.send_replace(GpuState { gpus });
        tokio::time::sleep(interval).await;
    }
}

/// Scans `/proc/*/fdinfo/*` for i915 DRM clients, returning cumulative
/// render-engine busy nanoseconds keyed by PCI id then DRM client-id
/// (deduplicating fds that share the same open file description, which
/// report identical counters).
fn scan_fdinfo() -> HashMap<String, HashMap<u64, u64>> {
    let mut result: HashMap<String, HashMap<u64, u64>> = HashMap::new();

    let Ok(procs) = std::fs::read_dir("/proc") else {
        return result;
    };

    for proc_entry in procs.flatten() {
        let is_pid_dir = proc_entry
            .file_name()
            .to_str()
            .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
        if !is_pid_dir {
            continue;
        }

        let Ok(fds) = std::fs::read_dir(proc_entry.path().join("fdinfo")) else {
            continue;
        };

        for fd_entry in fds.flatten() {
            let Ok(content) = std::fs::read_to_string(fd_entry.path()) else {
                continue;
            };
            if let Some((pci_id, client_id, render_ns)) = parse_i915_fdinfo(&content) {
                result
                    .entry(pci_id)
                    .or_default()
                    .insert(client_id, render_ns);
            }
        }
    }

    result
}

fn parse_i915_fdinfo(content: &str) -> Option<(String, u64, u64)> {
    let mut driver = None;
    let mut pci_id = None;
    let mut client_id = None;
    let mut render_ns = None;

    for line in content.lines() {
        if let Some(v) = line.strip_prefix("drm-driver:\t") {
            driver = Some(v);
        } else if let Some(v) = line.strip_prefix("drm-pdev:\t") {
            pci_id = Some(v.to_string());
        } else if let Some(v) = line.strip_prefix("drm-client-id:\t") {
            client_id = v.trim().parse::<u64>().ok();
        } else if let Some(v) = line.strip_prefix("drm-engine-render:\t") {
            render_ns = v
                .trim()
                .strip_suffix(" ns")
                .and_then(|s| s.parse::<u64>().ok());
        }
    }

    if driver != Some("i915") {
        return None;
    }
    Some((pci_id?, client_id?, render_ns?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_fdinfo_sample() {
        let sample = "pos:\t0\nflags:\t02100002\nmnt_id:\t38\nino:\t390\n\
            drm-driver:\ti915\ndrm-client-id:\t51\ndrm-pdev:\t0000:00:02.0\n\
            drm-total-system0:\t77412 KiB\ndrm-engine-render:\t218989144816 ns\n\
            drm-engine-copy:\t0 ns\n";
        let (pci_id, client_id, render_ns) = parse_i915_fdinfo(sample).unwrap();
        assert_eq!(pci_id, "0000:00:02.0");
        assert_eq!(client_id, 51);
        assert_eq!(render_ns, 218989144816);
    }

    #[test]
    fn ignores_non_i915_driver() {
        let sample = "drm-driver:\tnouveau\ndrm-client-id:\t1\ndrm-pdev:\t0000:01:00.0\n\
            drm-engine-render:\t123 ns\n";
        assert!(parse_i915_fdinfo(sample).is_none());
    }
}
