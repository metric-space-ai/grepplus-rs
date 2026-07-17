//! Stateless edit handles.
//!
//! A handle binds everything an edit needs to verify that it still addresses
//! what the agent read: workspace root, path, file hash, byte range, target
//! hash, signature fingerprint, and grammar identity. Every component is
//! re-derivable and re-checked at use; there is no daemon session state
//! behind a handle. If any component no longer holds, the operation fails
//! stale (exit 12) — a handle can fail an edit, never misbind one.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::hash::sha256_hex;
use greppy_core::{Error, Result};

pub const HANDLE_PREFIX: &str = "geh1:";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditHandle {
    pub workspace_root: String,
    pub path: String,
    pub file_sha256: String,
    pub byte_start: usize,
    pub byte_end: usize,
    pub target_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar_version: Option<String>,
}

impl EditHandle {
    /// Create a handle for a byte range of `content` at `path`.
    pub fn for_range(
        workspace_root: &Path,
        path: &Path,
        content: &[u8],
        byte_start: usize,
        byte_end: usize,
    ) -> Result<Self> {
        if byte_end > content.len() || byte_start > byte_end {
            return Err(Error::Invalid(format!(
                "handle range {byte_start}..{byte_end} outside file of {} bytes",
                content.len()
            )));
        }
        Ok(Self {
            workspace_root: workspace_root.to_string_lossy().into_owned(),
            path: path.to_string_lossy().into_owned(),
            file_sha256: sha256_hex(content),
            byte_start,
            byte_end,
            target_sha256: sha256_hex(&content[byte_start..byte_end]),
            signature_fingerprint: None,
            grammar_id: None,
            grammar_version: None,
        })
    }

    /// Serialize to the opaque token form (`geh1:` + base64url JSON).
    pub fn encode(&self) -> String {
        let json = serde_json::to_vec(self).expect("handle serializes");
        format!("{HANDLE_PREFIX}{}", base64url_encode(&json))
    }

    /// Parse a token produced by [`EditHandle::encode`].
    pub fn decode(token: &str) -> Result<Self> {
        let body = token
            .strip_prefix(HANDLE_PREFIX)
            .ok_or_else(|| Error::Invalid(format!("not an edit handle: {token:.16}…")))?;
        let bytes = base64url_decode(body)
            .ok_or_else(|| Error::Invalid("handle is not valid base64url".into()))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| Error::Invalid(format!("handle payload invalid: {e}")))
    }

    /// Verify the handle against live file content. Returns the target slice
    /// bounds on success; a mismatch is a stale handle, never a panic.
    pub fn verify(&self, live_content: &[u8]) -> Result<(usize, usize)> {
        if sha256_hex(live_content) != self.file_sha256 {
            return Err(Error::Workspace(format!(
                "stale handle: file {} changed since the handle was issued",
                self.path
            )));
        }
        // file hash matched, so the range is valid by construction; check anyway
        if self.byte_end > live_content.len() {
            return Err(Error::Workspace(format!(
                "stale handle: range outside {}",
                self.path
            )));
        }
        let target = &live_content[self.byte_start..self.byte_end];
        if sha256_hex(target) != self.target_sha256 {
            return Err(Error::Workspace(format!(
                "stale handle: target span in {} changed",
                self.path
            )));
        }
        Ok((self.byte_start, self.byte_end))
    }
}

// Minimal base64url (no padding) to avoid a new dependency: the alphabet is
// fixed by RFC 4648 §5 and the payload is always well-formed JSON we encode
// ourselves.
const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn base64url_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(B64[(n >> 6) as usize & 63] as char);
        }
        if chunk.len() > 2 {
            out.push(B64[n as usize & 63] as char);
        }
    }
    out
}

fn base64url_decode(text: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        B64.iter().position(|&b| b == c).map(|p| p as u32)
    }
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3 + 2);
    for chunk in bytes.chunks(4) {
        if chunk.len() < 2 {
            return None;
        }
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= val(c)? << (18 - 6 * i as u32);
        }
        out.push((n >> 16) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(n as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn handle(content: &[u8]) -> EditHandle {
        EditHandle::for_range(
            &PathBuf::from("/ws"),
            &PathBuf::from("src/a.rs"),
            content,
            3,
            8,
        )
        .unwrap()
    }

    #[test]
    fn roundtrip_encode_decode() {
        let h = handle(b"fn main() {}\n");
        let token = h.encode();
        assert!(token.starts_with(HANDLE_PREFIX));
        assert_eq!(EditHandle::decode(&token).unwrap(), h);
    }

    #[test]
    fn verify_accepts_unchanged_file() {
        let content = b"fn main() {}\n";
        assert_eq!(handle(content).verify(content).unwrap(), (3, 8));
    }

    #[test]
    fn verify_rejects_any_byte_change() {
        let content = b"fn main() {}\n";
        let h = handle(content);
        let mut mutated = content.to_vec();
        mutated[0] = b'F';
        assert!(h.verify(&mutated).is_err());
    }

    #[test]
    fn decode_rejects_garbage() {
        assert!(EditHandle::decode("geh1:!!!!").is_err());
        assert!(EditHandle::decode("nope").is_err());
    }
}
