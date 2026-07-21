use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use tokio::sync::watch;
use tracing::debug;

use crate::util::read_trimmed;

const POLL_INTERVAL: Duration = Duration::from_secs(30);
const POWER_SUPPLY_DIR: &str = "/sys/class/power_supply";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct BatteryState {
    /// Battery charge in [0.0, 1.0]. Meaningful only when `present`.
    pub percent: f64,
    /// True when at least one battery was found under /sys/class/power_supply.
    pub present: bool,
    /// True when at least one AC adapter reports `online`.
    pub on_ac: bool,
}

pub async fn run(tx: watch::Sender<BatteryState>) -> Result<()> {
    loop {
        let state = tokio::task::spawn_blocking(read_battery)
            .await
            .unwrap_or_default();
        tx.send_if_modified(|current| {
            if *current != state {
                debug!(
                    "Battery: {:.0}% present={} on_ac={}",
                    state.percent * 100.0,
                    state.present,
                    state.on_ac
                );
                *current = state;
                true
            } else {
                false
            }
        });
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

fn read_battery() -> BatteryState {
    read_battery_from(Path::new(POWER_SUPPLY_DIR))
}

fn read_battery_from(dir: &Path) -> BatteryState {
    let mut state = BatteryState::default();

    let Ok(entries) = std::fs::read_dir(dir) else {
        return state;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let kind = read_trimmed(path.join("type"));
        match kind.as_deref() {
            Some("Battery") => {
                // Skip peripheral batteries (wireless mice, controllers) which
                // report scope "Device"; only the system battery gates dimming.
                if read_trimmed(path.join("scope")).as_deref() == Some("Device") {
                    continue;
                }
                if let Some(capacity) = read_trimmed(path.join("capacity"))
                    .and_then(|v| v.parse::<f64>().ok())
                {
                    state.present = true;
                    state.percent = (capacity / 100.0).clamp(0.0, 1.0);
                }
            }
            // Mains / USB / Wireless power delivery all count as AC.
            Some(_) if read_trimmed(path.join("online")).as_deref() == Some("1") => {
                state.on_ac = true;
            }
            _ => {}
        }
    }

    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Minimal self-cleaning temp dir (avoids pulling in the tempfile crate).
    struct TmpDir(PathBuf);

    impl TmpDir {
        fn new() -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "tpx-battery-test-{}-{n}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn supply(&self, name: &str, files: &[(&str, &str)]) {
            let d = self.0.join(name);
            fs::create_dir_all(&d).unwrap();
            for (k, v) in files {
                fs::write(d.join(k), v).unwrap();
            }
        }
    }

    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn on_battery_reports_percent_not_ac() {
        let tmp = TmpDir::new();
        tmp.supply("ACAD", &[("type", "Mains\n"), ("online", "0\n")]);
        tmp.supply("BAT1", &[("type", "Battery\n"), ("capacity", "78\n")]);

        let s = read_battery_from(&tmp.0);
        assert!(s.present);
        assert!(!s.on_ac);
        assert!((s.percent - 0.78).abs() < 1e-9);
    }

    #[test]
    fn online_adapter_reports_ac() {
        let tmp = TmpDir::new();
        tmp.supply("ACAD", &[("type", "Mains\n"), ("online", "1\n")]);
        tmp.supply("BAT0", &[("type", "Battery\n"), ("capacity", "42\n")]);

        let s = read_battery_from(&tmp.0);
        assert!(s.present);
        assert!(s.on_ac);
        assert!((s.percent - 0.42).abs() < 1e-9);
    }

    #[test]
    fn device_scoped_battery_is_ignored() {
        let tmp = TmpDir::new();
        tmp.supply(
            "hidpp_battery",
            &[("type", "Battery\n"), ("scope", "Device\n"), ("capacity", "55\n")],
        );

        let s = read_battery_from(&tmp.0);
        assert!(!s.present);
        assert_eq!(s.percent, 0.0);
    }

    #[test]
    fn missing_dir_is_empty_state() {
        let s = read_battery_from(Path::new("/nonexistent/power_supply"));
        assert_eq!(s, BatteryState::default());
    }
}
