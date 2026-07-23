use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::watch;
use tracing::debug;

/// A logical CPU's marketing model name. Kept per-core rather than collapsed
/// into one string because hybrid/heterogeneous SoCs (ARM big.LITTLE, Intel
/// P-core/E-core) genuinely have more than one CPU model in the same box.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct CoreModel {
    pub cpu: u32,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CpuState {
    /// Per-core marketing model names, read once at startup since they never
    /// change at runtime.
    pub models: Vec<CoreModel>,
    pub usage: f64, // 0.0 - 100.0
}

pub async fn run(tx: watch::Sender<CpuState>) -> Result<()> {
    let models = resolve_models().await;
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
                models: models.clone(),
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

/// Best-effort per-core marketing CPU model names, e.g. `{0: "AMD Ryzen
/// Embedded V1605B"}` or, on a hybrid ARM SoC, `{0: "Cortex-A55", 1:
/// "Cortex-A55", 2: "Cortex-A76", 3: "Cortex-A76"}`. Never changes at
/// runtime, so callers should resolve it once at startup rather than on
/// every poll tick.
pub async fn resolve_models() -> Vec<CoreModel> {
    let from_cpuinfo = read_models_from_cpuinfo().await;
    if !from_cpuinfo.is_empty() {
        return from_cpuinfo;
    }
    read_models_from_lscpu().await
}

async fn read_models_from_cpuinfo() -> Vec<CoreModel> {
    tokio::task::spawn_blocking(|| {
        let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        Some(parse_cpuinfo_models(&content))
    })
    .await
    .ok()
    .flatten()
    .unwrap_or_default()
}

/// `/proc/cpuinfo` lists one block per logical CPU, separated by a blank
/// line; each block carries its own "processor" index and "model name" (x86
/// only - ARM has no such field, see `read_models_from_lscpu`).
fn parse_cpuinfo_models(content: &str) -> Vec<CoreModel> {
    let mut result = Vec::new();
    let mut cpu: Option<u32> = None;
    let mut model: Option<String> = None;

    for line in content.lines() {
        if line.is_empty() {
            if let (Some(cpu), Some(model)) = (cpu.take(), model.take()) {
                result.push(CoreModel {
                    cpu,
                    model: clean_model_name(&model),
                });
            }
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "processor" => cpu = value.trim().parse().ok(),
            "model name" => model = Some(value.trim().to_string()),
            _ => {}
        }
    }
    if let (Some(cpu), Some(model)) = (cpu, model) {
        result.push(CoreModel {
            cpu,
            model: clean_model_name(&model),
        });
    }

    result
}

/// ARM `/proc/cpuinfo` (e.g. on Raspberry Pi) has no "model name" line, only
/// per-core implementer/part IDs (e.g. `0x41`/`0xd08`) that need a database to
/// resolve to something readable ("Cortex-A72"). `lscpu`'s extended,
/// per-logical-CPU format already ships that database and - unlike its
/// summary view - reports each core's own model, which is what correctly
/// identifies heterogeneous (big.LITTLE-style) SoCs.
async fn read_models_from_lscpu() -> Vec<CoreModel> {
    let Ok(output) = tokio::process::Command::new("lscpu")
        .args(["-e=cpu,MODELNAME", "--json"])
        .output()
        .await
    else {
        return Vec::new();
    };
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) else {
        return Vec::new();
    };
    parse_lscpu_extended_models(&json)
}

fn parse_lscpu_extended_models(json: &serde_json::Value) -> Vec<CoreModel> {
    let Some(cpus) = json.get("cpus").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    cpus.iter()
        .filter_map(|entry| {
            let cpu = entry.get("cpu")?.as_u64()? as u32;
            let model = entry.get("modelname")?.as_str()?;
            Some(CoreModel {
                cpu,
                model: clean_model_name(model),
            })
        })
        .collect()
}

/// Strip noise `/proc/cpuinfo` bakes into the "model name" field: registered
/// trademark markers, Intel's trailing clock-speed ("... CPU @ 3.60GHz"), AMD
/// APUs advertising their integrated GPU ("... with Radeon Vega Gfx"), and
/// AMD's trailing core-count marketing suffix ("... 12-Core Processor").
fn clean_model_name(raw: &str) -> String {
    let mut name = raw.trim();
    if let Some(idx) = name.find(" with Radeon") {
        name = &name[..idx];
    }
    if let Some(idx) = name.find(" CPU @") {
        name = &name[..idx];
    }
    let cleaned = name.replace("(R)", "").replace("(TM)", "");
    let mut words: Vec<&str> = cleaned.split_whitespace().collect();
    if words.last() == Some(&"Processor")
        && words.len() >= 2
        && is_core_count(words[words.len() - 2])
    {
        words.truncate(words.len() - 2);
    }
    words.join(" ")
}

/// True for tokens like `12-Core` - a bare digit count followed by `-Core`.
fn is_core_count(word: &str) -> bool {
    word.strip_suffix("-Core")
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// Best-effort per-core current clock speed in MHz, keyed by logical CPU
/// index, matching what Node's `os.cpus()[].speed` reports. Unlike the model
/// name, current clock speed genuinely varies per core (frequency scaling),
/// so callers should re-read this every tick rather than caching it like
/// `resolve_models`.
///
/// Prefers `/proc/cpuinfo`'s `cpu MHz` field (x86); ARM's `/proc/cpuinfo`
/// (e.g. Raspberry Pi) carries no such field at all, so this falls back to
/// sysfs `cpufreq/scaling_cur_freq` - the same source `lscpu` itself reads
/// for its `CPU max/min MHz` output.
pub fn read_speeds() -> HashMap<u32, u32> {
    let from_cpuinfo = read_cpuinfo_speeds();
    if !from_cpuinfo.is_empty() {
        return from_cpuinfo;
    }
    read_sysfs_speeds()
}

fn read_cpuinfo_speeds() -> HashMap<u32, u32> {
    let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") else {
        return HashMap::new();
    };
    parse_cpuinfo_speeds(&content)
}

fn parse_cpuinfo_speeds(content: &str) -> HashMap<u32, u32> {
    let mut result = HashMap::new();
    let mut current_idx: Option<u32> = None;
    let mut current_mhz: Option<f64> = None;

    for line in content.lines() {
        if line.is_empty() {
            if let (Some(idx), Some(mhz)) = (current_idx.take(), current_mhz.take()) {
                result.insert(idx, mhz.round() as u32);
            }
            continue;
        }

        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();

        match key {
            "processor" => current_idx = value.parse().ok(),
            "cpu MHz" => current_mhz = value.parse().ok(),
            _ => {}
        }
    }
    if let (Some(idx), Some(mhz)) = (current_idx, current_mhz) {
        result.insert(idx, mhz.round() as u32);
    }

    result
}

/// Reads current clock speed via sysfs `cpufreq/scaling_cur_freq` (in kHz)
/// for each `cpuN` directory under `/sys/devices/system/cpu/`. Used as the
/// fallback when `/proc/cpuinfo` carries no `cpu MHz` field, e.g. ARM boards
/// such as the Raspberry Pi. `scaling_cur_freq` is world-readable, unlike the
/// root-only `cpuinfo_cur_freq` sibling file.
fn read_sysfs_speeds() -> HashMap<u32, u32> {
    let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") else {
        return HashMap::new();
    };

    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let idx: u32 = name.to_str()?.strip_prefix("cpu")?.parse().ok()?;
            let khz_path = entry.path().join("cpufreq/scaling_cur_freq");
            let khz: u64 = std::fs::read_to_string(khz_path).ok()?.trim().parse().ok()?;
            Some((idx, khz_to_mhz(khz)))
        })
        .collect()
}

fn khz_to_mhz(khz: u64) -> u32 {
    ((khz + 500) / 1000) as u32
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

    #[test]
    fn parses_per_core_model_from_cpuinfo() {
        let content = "processor\t: 0\n\
            model name\t: AMD Ryzen Embedded V1605B with Radeon Vega Gfx\n\
            cpu MHz\t: 2000.0\n\
            \n\
            processor\t: 1\n\
            model name\t: AMD Ryzen Embedded V1605B with Radeon Vega Gfx\n\
            cpu MHz\t: 2100.0\n";
        assert_eq!(
            parse_cpuinfo_models(content),
            vec![
                CoreModel {
                    cpu: 0,
                    model: "AMD Ryzen Embedded V1605B".to_string()
                },
                CoreModel {
                    cpu: 1,
                    model: "AMD Ryzen Embedded V1605B".to_string()
                },
            ],
        );
    }

    #[test]
    fn parses_heterogeneous_cores_from_cpuinfo() {
        // Distinct "model name" values per core, e.g. an Intel P-core/E-core
        // split, must not collapse into a single reported model.
        let content = "processor\t: 0\nmodel name\t: Intel P-Core\n\
            \n\
            processor\t: 1\nmodel name\t: Intel E-Core\n";
        assert_eq!(
            parse_cpuinfo_models(content),
            vec![
                CoreModel {
                    cpu: 0,
                    model: "Intel P-Core".to_string()
                },
                CoreModel {
                    cpu: 1,
                    model: "Intel E-Core".to_string()
                },
            ],
        );
    }

    #[test]
    fn returns_empty_for_arm_cpuinfo_without_model_name() {
        let content = "processor\t: 0\nBogoMIPS\t: 108.00\nCPU implementer\t: 0x41\nCPU part\t: 0xd08\n";
        assert_eq!(parse_cpuinfo_models(content), Vec::new());
    }

    #[test]
    fn parses_per_core_models_from_lscpu_extended_json() {
        let json = serde_json::json!({
            "cpus": [
                {"cpu": 0, "modelname": "Cortex-A72"},
                {"cpu": 1, "modelname": "Cortex-A72"},
            ]
        });
        assert_eq!(
            parse_lscpu_extended_models(&json),
            vec![
                CoreModel {
                    cpu: 0,
                    model: "Cortex-A72".to_string()
                },
                CoreModel {
                    cpu: 1,
                    model: "Cortex-A72".to_string()
                },
            ],
        );
    }

    #[test]
    fn parses_heterogeneous_cores_from_lscpu_extended_json() {
        // A big.LITTLE SoC (e.g. RK3588's 4x A55 + 4x A76) reports a distinct
        // model per cpu row; each core's own model must survive, not just
        // the first one seen.
        let json = serde_json::json!({
            "cpus": [
                {"cpu": 0, "modelname": "Cortex-A55"},
                {"cpu": 1, "modelname": "Cortex-A55"},
                {"cpu": 2, "modelname": "Cortex-A76"},
                {"cpu": 3, "modelname": "Cortex-A76"},
            ]
        });
        assert_eq!(
            parse_lscpu_extended_models(&json),
            vec![
                CoreModel {
                    cpu: 0,
                    model: "Cortex-A55".to_string()
                },
                CoreModel {
                    cpu: 1,
                    model: "Cortex-A55".to_string()
                },
                CoreModel {
                    cpu: 2,
                    model: "Cortex-A76".to_string()
                },
                CoreModel {
                    cpu: 3,
                    model: "Cortex-A76".to_string()
                },
            ],
        );
    }

    #[test]
    fn lscpu_extended_models_cleans_raw_names() {
        let json = serde_json::json!({
            "cpus": [
                {"cpu": 0, "modelname": "AMD Ryzen 9 7900 12-Core Processor"},
            ]
        });
        assert_eq!(
            parse_lscpu_extended_models(&json),
            vec![CoreModel {
                cpu: 0,
                model: "AMD Ryzen 9 7900".to_string()
            }],
        );
    }

    #[test]
    fn lscpu_extended_models_empty_without_cpus_field() {
        assert_eq!(
            parse_lscpu_extended_models(&serde_json::json!({})),
            Vec::new(),
        );
    }

    #[test]
    fn strips_radeon_gpu_suffix() {
        assert_eq!(
            clean_model_name("AMD Ryzen Embedded V1605B with Radeon Vega Gfx"),
            "AMD Ryzen Embedded V1605B",
        );
        assert_eq!(
            clean_model_name("AMD Ryzen 7 5700G with Radeon Graphics"),
            "AMD Ryzen 7 5700G",
        );
    }

    #[test]
    fn strips_amd_core_count_suffix() {
        assert_eq!(
            clean_model_name("AMD Ryzen 9 7900 12-Core Processor"),
            "AMD Ryzen 9 7900",
        );
    }

    #[test]
    fn leaves_processor_suffix_without_core_count_untouched() {
        assert_eq!(
            clean_model_name("Some Weird Processor"),
            "Some Weird Processor",
        );
    }

    #[test]
    fn strips_intel_clock_speed_and_trademark_markers() {
        assert_eq!(
            clean_model_name("Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz"),
            "Intel Core i7-9700K",
        );
    }

    #[test]
    fn leaves_plain_model_names_untouched() {
        assert_eq!(
            clean_model_name("AMD Ryzen 9 7950X3D"),
            "AMD Ryzen 9 7950X3D",
        );
    }

    #[test]
    fn parses_per_core_speeds_from_cpuinfo() {
        let content = "processor\t: 0\n\
            model name\t: AMD Ryzen Embedded V1605B\n\
            cpu MHz\t: 2000.0\n\
            \n\
            processor\t: 1\n\
            model name\t: AMD Ryzen Embedded V1605B\n\
            cpu MHz\t: 2100.0\n";
        let speeds = parse_cpuinfo_speeds(content);
        assert_eq!(speeds.get(&0), Some(&2000));
        assert_eq!(speeds.get(&1), Some(&2100));
    }

    #[test]
    fn rounds_fractional_mhz() {
        let content = "processor\t: 0\ncpu MHz\t: 2399.6\n";
        assert_eq!(parse_cpuinfo_speeds(content).get(&0), Some(&2400));
    }

    #[test]
    fn empty_without_cpu_mhz_field() {
        let content = "processor\t: 0\nBogoMIPS\t: 108.00\n";
        assert!(parse_cpuinfo_speeds(content).is_empty());
    }

    #[test]
    fn khz_to_mhz_rounds_to_nearest() {
        assert_eq!(khz_to_mhz(1_500_000), 1500);
        assert_eq!(khz_to_mhz(2_399_600), 2400);
        assert_eq!(khz_to_mhz(600_000), 600);
    }
}
