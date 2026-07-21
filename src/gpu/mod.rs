pub mod amd;
pub mod intel;
pub mod nvidia;

use serde::{Deserialize, Serialize};

use crate::sensors::SensorSpec;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuState {
    pub gpus: Vec<GpuInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub id: String,
    /// Human-readable label: "iGPU", "dGPU", or "" for single-GPU hosts.
    pub label: String,
    pub provider: GpuProvider,
    pub gpu_usage: f64,   // 0 - 100 %
    pub mem_used: u64,    // bytes
    pub mem_total: u64,   // bytes
    pub gtt_used: u64,    // bytes; AMD only, 0 on other providers
    pub gtt_total: u64,   // bytes; AMD only, 0 on other providers
    pub temperature: f64, // °C (junction/hotspot when available, else edge)
    pub power_watts: f64,
    /// Whether to show this GPU in the bar widget.
    pub in_bar: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    pub id: String,
    pub provider: GpuProvider,
    /// hwmon sensor specs for this GPU's temperature readings.
    pub gpu_temp_sensors: Vec<SensorSpec>,
    /// Whether to show this GPU in the bar widget (default: true).
    /// Set to false for iGPUs that should appear in the popup only.
    #[serde(default = "default_true")]
    pub bar: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GpuProvider {
    Amd,
    Nvidia,
    Intel,
}

fn default_true() -> bool {
    true
}
