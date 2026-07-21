use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::time::Duration;
use tokio::sync::watch;
use tracing::{debug, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorSpec {
    pub device: String,
    pub sensors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SensorsState {
    pub readings: Vec<SensorReading>,
    /// Auto-discovered temperature sensors not covered by any configured spec.
    pub extra: Vec<SensorReading>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReading {
    pub device: String,
    pub sensor: String,
    pub value: f64,
    pub unit: String,
    pub status: SensorStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SensorStatus {
    Normal,
    Warning,
    Critical,
}

pub async fn run(tx: watch::Sender<SensorsState>, specs: Vec<SensorSpec>) -> Result<()> {
    if specs.is_empty() {
        debug!("No sensor specs configured, running auto-discovery only");
    }

    loop {
        match read_sensors(&specs).await {
            Ok(state) => {
                debug!(
                    "Sensors: {} readings, {} extra",
                    state.readings.len(),
                    state.extra.len()
                );
                tx.send_replace(state);
            }
            Err(e) => warn!("sensors read error: {e}"),
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

async fn read_sensors(specs: &[SensorSpec]) -> Result<SensorsState> {
    let output = tokio::process::Command::new("sensors")
        .arg("-J")
        .output()
        .await?;

    let json: Value = serde_json::from_slice(&output.stdout)?;
    let obj = json
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("sensors -J output is not a JSON object"))?;

    let mut readings = Vec::new();
    let mut covered: HashSet<String> = HashSet::new();

    for spec in specs {
        covered.insert(spec.device.clone());

        let device_data = match obj.get(&spec.device) {
            Some(v) => v,
            None => continue,
        };

        for sensor_name in &spec.sensors {
            let sensor_obj = device_data.as_object().and_then(|d| {
                // Match by key name, or by a "label" field if the sensor has one
                d.iter()
                    .find(|(key, val)| {
                        *key == sensor_name
                            || val
                                .get("label")
                                .and_then(|l| l.as_str())
                                .map(|l| l == sensor_name.as_str())
                                .unwrap_or(false)
                    })
                    .map(|(_, v)| v)
            });

            let Some(sensor_obj) = sensor_obj else {
                continue;
            };

            let input = sensor_obj.get("input");
            let value = input
                .and_then(|i| i.get("value"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let unit = input
                .and_then(|i| i.get("unit"))
                .and_then(|u| u.as_str())
                .unwrap_or("")
                .to_string();
            let status = compute_status(sensor_obj, value);

            readings.push(SensorReading {
                device: spec.device.clone(),
                sensor: sensor_name.clone(),
                value,
                unit,
                status,
            });
        }
    }

    let extra = auto_discover(obj, &covered);

    Ok(SensorsState { readings, extra })
}

/// Scan all devices not already in `covered` for temperature sensors.
/// Returns up to 4 readings sorted by temperature descending, filtering
/// out obviously noisy sources (ACPI, USB-C PSY, very low temps).
fn auto_discover(
    obj: &serde_json::Map<String, Value>,
    covered: &HashSet<String>,
) -> Vec<SensorReading> {
    let mut candidates: Vec<SensorReading> = Vec::new();

    for (chip, chip_data) in obj {
        if covered.contains(chip) {
            continue;
        }
        // These chips produce only noise or redundant data.
        if chip.starts_with("acpitz-") || chip.starts_with("ucsi_") {
            continue;
        }

        let Some(chip_obj) = chip_data.as_object() else {
            continue;
        };

        // Pick the single most interesting temperature sensor for this chip.
        let mut best: Option<(String, f64, SensorStatus)> = None;

        for (key, sensor_val) in chip_obj {
            if key == "Adapter" {
                continue;
            }
            let Some(input) = sensor_val.get("input") else {
                continue;
            };
            if input.get("unit").and_then(|u| u.as_str()) != Some("°C") {
                continue;
            }
            let Some(value) = input.get("value").and_then(|v| v.as_f64()) else {
                continue;
            };
            if !(25.0..=150.0).contains(&value) {
                continue;
            }

            // Use the JSON label when present, otherwise the key (e.g. "temp1").
            let display = sensor_val
                .get("label")
                .and_then(|l| l.as_str())
                .unwrap_or(key)
                .to_string();

            let status = compute_status(sensor_val, value);

            let is_better = match &best {
                None => true,
                Some((existing_display, existing_val, _)) => {
                    // Prefer labelled (non-generic) over plain "tempN" keys.
                    let cur_labelled = sensor_val.get("label").is_some();
                    let prev_labelled = !existing_display.starts_with("temp");
                    if cur_labelled && !prev_labelled {
                        true
                    } else if !cur_labelled && prev_labelled {
                        false
                    } else {
                        value > *existing_val
                    }
                }
            };
            if is_better {
                best = Some((display, value, status));
            }
        }

        if let Some((display, value, status)) = best {
            let sensor_label = format!("{}: {}", chip_friendly(chip), display);
            candidates.push(SensorReading {
                device: chip.clone(),
                sensor: sensor_label,
                value,
                unit: "°C".to_string(),
                status,
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.value
            .partial_cmp(&a.value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(4);
    candidates
}

/// Map a chip name like "nvme-pci-0100" to a short friendly prefix ("NVMe").
/// Falls back to the first segment before any "-" separator.
fn chip_friendly(chip: &str) -> &str {
    if chip.starts_with("nvme-") {
        return "NVMe";
    }
    if chip.starts_with("iwlwifi-") || chip.starts_with("iwl-") {
        return "WiFi";
    }
    // MediaTek / Qualcomm WiFi chips (e.g. mt7921_phy0-pci-…, ath11k-…)
    if chip.starts_with("mt") && chip.contains("phy")
        || chip.starts_with("ath")
        || chip.starts_with("rtw")
    {
        return "WiFi";
    }
    if chip.starts_with("pch_") {
        return "PCH";
    }
    chip.split('-').next().unwrap_or(chip)
}

fn compute_status(sensor: &Value, value: f64) -> SensorStatus {
    // Explicit alarm flags take precedence
    for alarm_key in &["crit_alarm", "lcrit_alarm", "max_alarm", "min_alarm"] {
        if sensor
            .get(alarm_key)
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_f64())
            == Some(1.0)
        {
            return SensorStatus::Critical;
        }
    }

    // Threshold comparisons
    if let Some(crit) = sensor
        .get("crit")
        .and_then(|v| v.get("value"))
        .and_then(|v| v.as_f64())
        && value >= crit
    {
        return SensorStatus::Critical;
    }
    if let Some(max) = sensor
        .get("max")
        .and_then(|v| v.get("value"))
        .and_then(|v| v.as_f64())
        && max > 0.0
        && value >= max
    {
        return SensorStatus::Warning;
    }

    // Absolute fallback for silicon temperatures. Chips like k10temp and the
    // amdgpu `edge` sensor advertise no usable max/crit, so a 90 °C CPU would
    // otherwise report Normal. These thresholds also match the shell's coloring.
    if value >= 90.0 {
        return SensorStatus::Critical;
    }
    if value >= 75.0 {
        return SensorStatus::Warning;
    }

    SensorStatus::Normal
}
