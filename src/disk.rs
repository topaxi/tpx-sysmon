use serde::{Deserialize, Serialize};
use std::process::Command;

/// Snapshot of mounted volumes and, for btrfs, the subvolumes mounted from
/// each one, with total/used/available space per volume.
///
/// btrfs subvolumes that share a device do NOT get separate usage numbers -
/// btrfs pools free space across every subvolume of a filesystem unless
/// quotas are enabled, which this does not assume or enable. Space is
/// reported once per underlying device instead.
///
/// Enumerating subvolumes that are not separately mounted, and listing
/// snapshots, both require `btrfs subvolume list`/`qgroup show`, which need
/// `CAP_SYS_ADMIN` - not available to this unprivileged poller. Only
/// subvolumes that are already mounted somewhere show up here, read straight
/// out of their mount options.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiskState {
    pub volumes: Vec<Volume>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Volume {
    /// Block device path, e.g. `/dev/mapper/root`. Multiple mountpoints on
    /// the same device (common for btrfs) collapse into one `Volume`.
    pub source: String,
    pub fstype: String,
    /// The shortest (parent-most) mountpoint of this device, used for
    /// display and as the `df` query target.
    pub mountpoint: String,
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub avail_bytes: u64,
    /// Empty for non-btrfs volumes.
    pub subvolumes: Vec<Subvolume>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Subvolume {
    pub id: u64,
    /// Subvolume path within the filesystem, e.g. `/@home`.
    pub path: String,
    pub mountpoint: String,
}

pub fn compute() -> DiskState {
    let mounts = std::fs::read_to_string("/proc/mounts").unwrap_or_default();
    let mut volumes = parse_mounts(&mounts);

    for volume in &mut volumes {
        if let Some((total, used, avail)) = df_bytes(&volume.mountpoint) {
            volume.total_bytes = total;
            volume.used_bytes = used;
            volume.avail_bytes = avail;
        }
    }

    DiskState { volumes }
}

/// Parses `/proc/mounts` into one `Volume` per distinct block device, keeping
/// only real disk-backed filesystems (source starting with `/dev/` - drops
/// tmpfs, proc, sysfs, cgroup2, overlay, fuse.*, efivarfs, etc).
fn parse_mounts(content: &str) -> Vec<Volume> {
    let mut volumes: Vec<Volume> = Vec::new();

    for line in content.lines() {
        let mut fields = line.split_whitespace();
        let (Some(source), Some(raw_mountpoint), Some(fstype), Some(options)) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            continue;
        };

        if !source.starts_with("/dev/") {
            continue;
        }

        let mountpoint = unescape_mount_field(raw_mountpoint);

        let volume = match volumes.iter().position(|v| v.source == source) {
            Some(i) => &mut volumes[i],
            None => {
                volumes.push(Volume {
                    source: source.to_string(),
                    fstype: fstype.to_string(),
                    mountpoint: mountpoint.clone(),
                    ..Default::default()
                });
                volumes.last_mut().unwrap()
            }
        };

        // Prefer the shortest (parent-most) mountpoint for display/df.
        if mountpoint.len() < volume.mountpoint.len() {
            volume.mountpoint = mountpoint.clone();
        }

        if fstype == "btrfs" {
            if let Some(sub) = parse_btrfs_subvol(options, &mountpoint) {
                volume.subvolumes.push(sub);
            }
        }
    }

    volumes
}

/// Pulls `subvolid=N` and `subvol=/path` out of a btrfs mount's comma
/// separated options field - both are always present for a btrfs mount, no
/// privileged call needed.
fn parse_btrfs_subvol(options: &str, mountpoint: &str) -> Option<Subvolume> {
    let mut id = None;
    let mut path = None;

    for opt in options.split(',') {
        if let Some(v) = opt.strip_prefix("subvolid=") {
            id = v.parse().ok();
        } else if let Some(v) = opt.strip_prefix("subvol=") {
            path = Some(v.to_string());
        }
    }

    Some(Subvolume {
        id: id?,
        path: path?,
        mountpoint: mountpoint.to_string(),
    })
}

/// /proc/mounts escapes space/tab/newline/backslash in path fields as octal
/// `\NNN` sequences (see `proc_mounts(5)`).
fn unescape_mount_field(field: &str) -> String {
    let mut out = String::with_capacity(field.len());
    let bytes = field.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 3 < bytes.len() {
            if let Ok(code) = u8::from_str_radix(&field[i + 1..i + 4], 8) {
                out.push(code as char);
                i += 4;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }

    out
}

fn df_bytes(mountpoint: &str) -> Option<(u64, u64, u64)> {
    let output = Command::new("df")
        .env("LC_ALL", "C")
        .args(["-B1", "--output=size,used,avail", mountpoint])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut fields = text.lines().nth(1)?.split_whitespace();
    let total = fields.next()?.parse().ok()?;
    let used = fields.next()?.parse().ok()?;
    let avail = fields.next()?.parse().ok()?;

    Some((total, used, avail))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_multiple_mounts_of_same_device() {
        let content = "\
/dev/mapper/root / btrfs rw,relatime,subvolid=256,subvol=/@ 0 0
/dev/mapper/root /var/log btrfs rw,relatime,subvolid=258,subvol=/@log 0 0
/dev/mapper/root /home btrfs rw,relatime,subvolid=257,subvol=/@home 0 0
/dev/nvme0n1p1 /boot vfat rw,relatime 0 0
tmpfs /run tmpfs rw 0 0
";
        let volumes = parse_mounts(content);
        assert_eq!(volumes.len(), 2);

        let root = volumes.iter().find(|v| v.source == "/dev/mapper/root").unwrap();
        assert_eq!(root.mountpoint, "/");
        assert_eq!(root.subvolumes.len(), 3);
        assert!(
            root.subvolumes
                .iter()
                .any(|s| s.path == "/@home" && s.mountpoint == "/home" && s.id == 257)
        );

        let boot = volumes.iter().find(|v| v.source == "/dev/nvme0n1p1").unwrap();
        assert_eq!(boot.fstype, "vfat");
        assert!(boot.subvolumes.is_empty());
    }

    #[test]
    fn skips_pseudo_filesystems() {
        let content = "\
proc /proc proc rw 0 0
tmpfs /dev/shm tmpfs rw 0 0
overlay /var/lib/docker/overlay2/abc/merged overlay rw 0 0
";
        assert!(parse_mounts(content).is_empty());
    }

    #[test]
    fn unescapes_octal_space_in_mountpoint() {
        let content = "/dev/sdb1 /mnt/My\\040Drive ext4 rw,relatime 0 0\n";
        let volumes = parse_mounts(content);
        assert_eq!(volumes[0].mountpoint, "/mnt/My Drive");
    }

    #[test]
    fn non_btrfs_has_no_subvolumes() {
        let content = "/dev/sda1 / ext4 rw,relatime 0 0\n";
        let volumes = parse_mounts(content);
        assert!(volumes[0].subvolumes.is_empty());
    }
}
