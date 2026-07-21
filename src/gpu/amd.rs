use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tracing::{debug, warn};

use crate::gpu::{GpuConfig, GpuInfo, GpuProvider, GpuState};
use crate::util::read_trimmed;

/// Scan `/sys/bus/pci/devices/` and return all PCI IDs that expose an `amdgpu`
/// hwmon (used for auto-discovery when no explicit configs are provided).
fn discover_amd_gpu_ids() -> Vec<String> {
    let Ok(pci_devices) = std::fs::read_dir("/sys/bus/pci/devices/") else {
        return vec![];
    };

    pci_devices
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.join("gpu_busy_percent").exists() {
                return None;
            }
            let is_amdgpu = std::fs::read_dir(path.join("hwmon"))
                .ok()
                .into_iter()
                .flatten()
                .flatten()
                .any(|e| read_trimmed(e.path().join("name")).as_deref() == Some("amdgpu"));
            if !is_amdgpu {
                return None;
            }
            Some(entry.file_name().to_string_lossy().into_owned())
        })
        .collect()
}

/// Reads AMD GPU stats straight from the kernel `amdgpu` sysfs interface,
/// avoiding the ~50 MB `amd-smi monitor` python subprocess. Every field the
/// shell consumes is exposed per-GPU under `/sys/bus/pci/devices/<bdf>/`:
///
/// - `gpu_busy_percent`               → usage %
/// - `mem_info_vram_{used,total}`     → VRAM bytes
/// - `hwmon/hwmon*/temp*_{label,input}` → temperature (°C)
/// - `hwmon/hwmon*/power1_{average,input}` → power (W)
///
/// `vk_types` maps a PCI device ID to whether Vulkan reports it as an
/// integrated GPU, used only to compute the `label` field on multi-GPU hosts.
/// Detecting this requires loading libvulkan, which callers may prefer to do
/// out-of-process (e.g. tpx-shell re-execs itself for this) - pass an empty
/// map to skip labeling entirely, which is safe since `label` is cosmetic.
pub async fn run(
    tx: watch::Sender<GpuState>,
    configs: Vec<GpuConfig>,
    poll_ms: u64,
    vk_types: HashMap<u32, bool>,
) -> Result<()> {
    let resolved: Vec<GpuConfig> = if configs.is_empty() {
        let found = discover_amd_gpu_ids();
        if found.is_empty() {
            debug!("No AMD GPU configs or sysfs devices found, skipping");
            std::future::pending::<()>().await;
            return Ok(());
        }
        debug!("Auto-discovered {} AMD GPU(s)", found.len());
        found
            .into_iter()
            .map(|id| GpuConfig {
                id,
                provider: GpuProvider::Amd,
                gpu_temp_sensors: vec![],
                bar: true,
            })
            .collect()
    } else {
        configs
    };

    let resolved = Arc::new(resolved);
    let vk_types = Arc::new(vk_types);
    let interval = Duration::from_millis(poll_ms);
    loop {
        let configs = Arc::clone(&resolved);
        let vk = Arc::clone(&vk_types);
        let state = tokio::task::spawn_blocking(move || read_state(&configs, &vk))
            .await
            .unwrap_or_default();
        tx.send_replace(state);
        tokio::time::sleep(interval).await;
    }
}

fn read_state(configs: &[GpuConfig], vk_types: &HashMap<u32, bool>) -> GpuState {
    let mut gpus = Vec::new();

    for cfg in configs {
        let id = &cfg.id;
        let dir = format!("/sys/bus/pci/devices/{id}");
        let dir = Path::new(&dir);
        if !dir.exists() {
            warn!("AMD GPU {id} sysfs path not found");
            continue;
        }

        let gpu_usage = read_trimmed(dir.join("gpu_busy_percent"))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let mem_used = read_trimmed(dir.join("mem_info_vram_used"))
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let mem_total = read_trimmed(dir.join("mem_info_vram_total"))
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let gtt_used = read_trimmed(dir.join("mem_info_gtt_used"))
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let gtt_total = read_trimmed(dir.join("mem_info_gtt_total"))
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        let (temperature, power_watts) = read_hwmon(dir);

        // Read PCI device ID from sysfs and look it up in the Vulkan map.
        // Falls back to the explicit `bar` config flag if Vulkan wasn't available.
        let pci_device_id = read_trimmed(dir.join("device"))
            .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok());
        let label = if configs.len() <= 1 {
            String::new()
        } else {
            let is_integrated = pci_device_id
                .and_then(|id| vk_types.get(&id).copied())
                .unwrap_or(!cfg.bar);
            if is_integrated { "iGPU" } else { "dGPU" }.to_string()
        };

        debug!("GPU {id}: {gpu_usage:.0}% {temperature:.0}°C {power_watts:.0}W");

        gpus.push(GpuInfo {
            id: id.clone(),
            label,
            provider: GpuProvider::Amd,
            gpu_usage,
            mem_used,
            mem_total,
            gtt_used,
            gtt_total,
            temperature,
            power_watts,
            in_bar: cfg.bar,
        });
    }

    GpuState { gpus }
}

/// Locate the `amdgpu` hwmon directory under the device and read temperature
/// and power from it. Returns `(0.0, 0.0)` if the sensors are unavailable.
fn read_hwmon(device_dir: &Path) -> (f64, f64) {
    let Ok(entries) = std::fs::read_dir(device_dir.join("hwmon")) else {
        return (0.0, 0.0);
    };

    for entry in entries.flatten() {
        let hwmon = entry.path();
        if read_trimmed(hwmon.join("name")).as_deref() != Some("amdgpu") {
            continue;
        }
        return (read_temp(&hwmon), read_power(&hwmon));
    }

    (0.0, 0.0)
}

/// Reads all labelled temperature sensors and returns the most representative
/// one: `junction` (the hotspot) when present, otherwise `edge`, otherwise any.
fn read_temp(hwmon: &Path) -> f64 {
    let mut temps: HashMap<String, f64> = HashMap::new();

    let Ok(entries) = std::fs::read_dir(hwmon) else {
        return 0.0;
    };

    for entry in entries.flatten() {
        let fname = entry.file_name();
        let Some(idx) = fname
            .to_str()
            .and_then(|n| n.strip_prefix("temp"))
            .and_then(|n| n.strip_suffix("_label"))
        else {
            continue;
        };
        let Some(label) = read_trimmed(entry.path()) else {
            continue;
        };
        if let Some(milli) =
            read_trimmed(hwmon.join(format!("temp{idx}_input"))).and_then(|s| s.parse::<f64>().ok())
        {
            temps.insert(label, milli / 1000.0);
        }
    }

    temps
        .get("junction")
        .or_else(|| temps.get("edge"))
        .or_else(|| temps.values().next())
        .copied()
        .unwrap_or(0.0)
}

/// Reads power draw in watts. Prefers the averaged reading, falling back to the
/// instantaneous one (APUs only expose `power1_input`). Sysfs reports µW.
fn read_power(hwmon: &Path) -> f64 {
    for file in ["power1_average", "power1_input"] {
        if let Some(micro) = read_trimmed(hwmon.join(file)).and_then(|s| s.parse::<f64>().ok()) {
            return micro / 1_000_000.0;
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct TmpDir(PathBuf);

    impl TmpDir {
        fn new() -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("tpx-amd-test-{}-{n}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn write(&self, name: &str, contents: &str) {
            fs::write(self.0.join(name), contents).unwrap();
        }
    }

    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn read_temp_prefers_junction_over_edge() {
        let tmp = TmpDir::new();
        tmp.write("temp1_label", "edge\n");
        tmp.write("temp1_input", "44000\n");
        tmp.write("temp2_label", "junction\n");
        tmp.write("temp2_input", "57000\n");
        tmp.write("temp3_label", "mem\n");
        tmp.write("temp3_input", "64000\n");

        assert_eq!(read_temp(&tmp.0), 57.0);
    }

    #[test]
    fn read_temp_falls_back_to_edge_then_any() {
        let edge = TmpDir::new();
        edge.write("temp1_label", "edge\n");
        edge.write("temp1_input", "40000\n");
        assert_eq!(read_temp(&edge.0), 40.0);

        let other = TmpDir::new();
        other.write("temp1_label", "mem\n");
        other.write("temp1_input", "66000\n");
        assert_eq!(read_temp(&other.0), 66.0);
    }

    #[test]
    fn read_temp_missing_is_zero() {
        assert_eq!(read_temp(Path::new("/nonexistent/hwmon")), 0.0);
    }

    #[test]
    fn read_power_prefers_average_and_converts_micro_watts() {
        let tmp = TmpDir::new();
        tmp.write("power1_average", "48000000\n");
        tmp.write("power1_input", "50030000\n");
        assert_eq!(read_power(&tmp.0), 48.0);
    }

    #[test]
    fn read_power_falls_back_to_input() {
        let tmp = TmpDir::new();
        tmp.write("power1_input", "50030000\n");
        assert!((read_power(&tmp.0) - 50.03).abs() < 1e-9);
    }

    #[test]
    fn read_power_missing_is_zero() {
        assert_eq!(read_power(Path::new("/nonexistent/hwmon")), 0.0);
    }
}
