use std::path::Path;

/// Read a sysfs/procfs file and return its contents with surrounding
/// whitespace trimmed. Returns `None` if the file is missing or unreadable.
pub(crate) fn read_trimmed(path: impl AsRef<Path>) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}
