//! Resolves human-readable GPU model names ("Radeon RX 7800 XT") from the PCI
//! address the vendor services already know.
//!
//! AMD marketing names are only disambiguated by device id + revision via
//! libdrm's `amdgpu.ids` - the same source `nvtop`/`amdgpu_top` read - because a
//! shared PCI device id like `747e` covers both the 7700 XT and the 7800 XT.
//! Everything else falls back to the generic `pci.ids` device name (what `lspci`
//! shows), then to the raw PCI address when neither database has an entry.

use std::collections::HashMap;
use std::path::Path;

use crate::gpu::GpuProvider;
use crate::util::read_trimmed;

const AMDGPU_IDS_PATH: &str = "/usr/share/libdrm/amdgpu.ids";
const PCI_IDS_PATH: &str = "/usr/share/hwdata/pci.ids";

/// Best-effort display name for the GPU at `pci_id` (a sysfs BDF such as
/// `0000:03:00.0`). Never empty: falls back to `pci_id` when no database entry
/// matches.
pub(crate) fn resolve(pci_id: &str, provider: GpuProvider) -> String {
    lookup(pci_id, provider).unwrap_or_else(|| pci_id.to_string())
}

/// Resolve display names for every GPU up front - a GPU's model never changes at
/// runtime - so the poll loops needn't re-read the (multi-hundred-KB) PCI
/// databases on every tick. Returns a map from PCI id to resolved name.
pub(crate) fn resolve_all<'a>(
    gpus: impl IntoIterator<Item = (&'a str, GpuProvider)>,
) -> HashMap<String, String> {
    gpus.into_iter()
        .map(|(id, provider)| (id.to_string(), resolve(id, provider)))
        .collect()
}

fn lookup(pci_id: &str, provider: GpuProvider) -> Option<String> {
    let dir = Path::new("/sys/bus/pci/devices").join(pci_id);
    let device_id = read_hex(dir.join("device"))?;

    if provider == GpuProvider::Amd
        && let Some(revision) = read_hex(dir.join("revision"))
        && let Ok(content) = std::fs::read_to_string(AMDGPU_IDS_PATH)
        && let Some(name) = parse_amdgpu_ids(&content, device_id, revision)
    {
        return Some(name);
    }

    let vendor_id = read_hex(dir.join("vendor"))?;
    let content = std::fs::read_to_string(PCI_IDS_PATH).ok()?;
    parse_pci_ids(&content, vendor_id, device_id)
}

/// Parse a `0x`-prefixed hex value from a sysfs file (e.g. `vendor`/`device`).
fn read_hex(path: impl AsRef<Path>) -> Option<u32> {
    let raw = read_trimmed(path)?;
    u32::from_str_radix(raw.trim_start_matches("0x"), 16).ok()
}

/// Look up a marketing name in libdrm's `amdgpu.ids` by device id + revision.
/// Lines are `DEVICE,\tREVISION,\tProduct Name` with uppercase hex ids; `#`
/// comments and the leading version line are ignored. Any redundant leading
/// "AMD " is stripped since the vendor is conveyed separately.
fn parse_amdgpu_ids(content: &str, device_id: u32, revision: u32) -> Option<String> {
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.splitn(3, ',');
        let (Some(dev), Some(rev), Some(name)) = (fields.next(), fields.next(), fields.next())
        else {
            continue; // version line or malformed - not a `dev,rev,name` entry
        };
        let dev = dev.trim();
        let rev = rev.trim();
        if u32::from_str_radix(dev, 16).ok() == Some(device_id)
            && u32::from_str_radix(rev, 16).ok() == Some(revision)
        {
            let name = name.trim();
            return Some(name.strip_prefix("AMD ").unwrap_or(name).to_string());
        }
    }
    None
}

/// Look up a device name in hwdata's `pci.ids`, scoped to its vendor. The file is
/// two levels deep: a vendor line at column 0 (`<vid>  Name`), then its devices
/// indented one tab (`\t<did>  Name`), until the next column-0 line. Device ids
/// are only unique within a vendor, so the vendor block must be matched first.
fn parse_pci_ids(content: &str, vendor_id: u32, device_id: u32) -> Option<String> {
    let vendor_hex = format!("{vendor_id:04x}");
    let device_hex = format!("{device_id:04x}");

    let mut in_vendor = false;
    for line in content.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        if !line.starts_with('\t') {
            // Column-0 line: a vendor (or trailing class) entry. Either the start
            // of our vendor block, or - once we are past it - the end of it.
            if in_vendor {
                return None;
            }
            in_vendor = line.split_once("  ").is_some_and(|(id, _)| id == vendor_hex);
            continue;
        }
        if !in_vendor || line.starts_with("\t\t") {
            continue; // outside our vendor, or a deeper subsystem line
        }
        if let Some((id, name)) = line.trim_start_matches('\t').split_once("  ")
            && id == device_hex
        {
            return Some(name.trim().to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const AMDGPU_IDS_SAMPLE: &str = concat!(
        "# List of AMDGPU IDs\n",
        "#\n",
        "1.0.0\n",
        "747E,\tC8,\tAMD Radeon RX 7800 XT\n",
        "747E,\tFF,\tAMD Radeon RX 7700 XT\n",
        "164E,\tD8,\tAMD Radeon 610M\n",
    );

    #[test]
    fn amdgpu_ids_disambiguates_by_revision_and_strips_amd_prefix() {
        assert_eq!(
            parse_amdgpu_ids(AMDGPU_IDS_SAMPLE, 0x747e, 0xc8).as_deref(),
            Some("Radeon RX 7800 XT"),
        );
        assert_eq!(
            parse_amdgpu_ids(AMDGPU_IDS_SAMPLE, 0x747e, 0xff).as_deref(),
            Some("Radeon RX 7700 XT"),
        );
    }

    #[test]
    fn amdgpu_ids_unknown_revision_is_none() {
        // A device present under other revisions but not this one (e.g. the
        // Raphael iGPU at revision 0xc4) must miss and fall back to pci.ids.
        assert_eq!(parse_amdgpu_ids(AMDGPU_IDS_SAMPLE, 0x164e, 0xc4), None);
    }

    const PCI_IDS_SAMPLE: &str = concat!(
        "# comment\n",
        "10de  NVIDIA Corporation\n",
        "\t2684  AD102 [GeForce RTX 4090]\n",
        "1002  Advanced Micro Devices, Inc. [AMD/ATI]\n",
        "\t164e  Raphael\n",
        "\t747e  Navi 32 [Radeon RX 7700 XT / 7800 XT]\n",
        "\t\t1043 0601  Device 0601\n",
        "14e4  Broadcom Inc. and subsidiaries\n",
        "\t164e  NetXtreme II BCM57710 10-Gigabit PCIe [Everest]\n",
    );

    #[test]
    fn pci_ids_looks_up_device_within_vendor() {
        assert_eq!(
            parse_pci_ids(PCI_IDS_SAMPLE, 0x1002, 0x747e).as_deref(),
            Some("Navi 32 [Radeon RX 7700 XT / 7800 XT]"),
        );
        assert_eq!(
            parse_pci_ids(PCI_IDS_SAMPLE, 0x10de, 0x2684).as_deref(),
            Some("AD102 [GeForce RTX 4090]"),
        );
    }

    #[test]
    fn pci_ids_scopes_device_id_to_vendor() {
        // 164e exists under both AMD (1002) and Broadcom (14e4); the vendor wins.
        assert_eq!(
            parse_pci_ids(PCI_IDS_SAMPLE, 0x1002, 0x164e).as_deref(),
            Some("Raphael"),
        );
        assert_eq!(
            parse_pci_ids(PCI_IDS_SAMPLE, 0x14e4, 0x164e).as_deref(),
            Some("NetXtreme II BCM57710 10-Gigabit PCIe [Everest]"),
        );
    }

    #[test]
    fn pci_ids_missing_device_and_subsystem_lines_ignored() {
        assert_eq!(parse_pci_ids(PCI_IDS_SAMPLE, 0x1002, 0xffff), None);
    }
}
