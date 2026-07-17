//! greppy-edit: the transactional read/edit surface.
//!
//! Contract: `docs/contracts/EDIT_CONTRACT.md` (binding), schemas
//! `edit-plan.v1` / `edit-certificate.v1` (normative). Design principles:
//! no fuzzy application ever; compare-and-swap end to end; the store
//! addresses, the live file decides; certificates instead of re-reads;
//! failure is a next step; idempotent `ensure-*` verbs.

pub mod certificate;
pub mod ensure;
pub mod handle;
pub mod publish;
pub mod txn;
pub mod verbs;

pub(crate) mod hash {
    use sha2::{Digest, Sha256};

    pub fn sha256_hex(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }
}

pub use certificate::{Certificate, PublishMode, Status};
pub use greppy_parser::{language_for_path, Language};
pub use handle::EditHandle;
