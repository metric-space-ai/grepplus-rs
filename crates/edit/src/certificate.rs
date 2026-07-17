//! Edit certificates: the machine-checkable result of every edit operation.
//!
//! Schema: `docs/contracts/edit-certificate.v1.schema.json` (normative).
//! Guarantee levels are named and reported separately; there is no scalar
//! confidence anywhere in this type.

use serde::{Deserialize, Serialize};

pub const CERTIFICATE_SCHEMA: &str = "greppy.edit-certificate.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    Applied,
    AlreadySatisfied,
    NotFound,
    Ambiguous,
    Stale,
    InvalidResult,
    ValidationFailed,
    PublishFailed,
}

impl Status {
    /// Binding exit-code mapping from `docs/contracts/EDIT_CONTRACT.md`.
    pub fn exit_code(self) -> i32 {
        match self {
            Status::Applied | Status::AlreadySatisfied => 0,
            Status::NotFound => 10,
            Status::Ambiguous => 11,
            Status::Stale => 12,
            Status::InvalidResult => 13,
            Status::ValidationFailed => 14,
            Status::PublishFailed => 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Guarantee {
    Proved,
    NotApplicable,
    WaivedByFormatter,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Guarantees {
    pub addressed_range: Guarantee,
    pub no_clobber: Guarantee,
    pub byte_isolation: Guarantee,
    pub syntax: Guarantee,
    pub validators: Guarantee,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SelectorEngine {
    Symbol,
    TreeSitter,
    Text,
    Regex,
    DataPath,
    Lsp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SelectorClass {
    Resolved,
    Structural,
    ExactText,
    RegexWeak,
    StructuredData,
    Semantic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxDelta {
    pub errors_before: usize,
    pub errors_after: usize,
    pub new_errors: usize,
    pub new_missing_nodes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostconditionResult {
    pub name: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub qualified_name: String,
    pub path: String,
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationReport {
    pub id: String,
    pub file: String,
    pub selector_engine: SelectorEngine,
    pub selector_class: SelectorClass,
    pub scope_matches: usize,
    pub target_matches: usize,
    pub file_sha256_before: String,
    pub file_sha256_after: Option<String>,
    pub target_sha256_before: String,
    pub target_sha256_after: Option<String>,
    pub outside_declared_ranges_unchanged: bool,
    pub changed_byte_ranges: Vec<(usize, usize)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_before: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_after: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unified_diff: Option<String>,
    pub syntax: SyntaxDelta,
    pub postconditions_passed: bool,
    pub postconditions: Vec<PostconditionResult>,
    pub guarantees: Guarantees,
    pub formatter_expanded_change_scope: bool,
    pub store_refreshed: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<Candidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorReport {
    pub argv: Vec<String>,
    pub exit_code: i32,
    pub timed_out: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceReport {
    pub root: String,
    pub git_head_before: Option<String>,
    pub git_head_after: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PublishMode {
    Atomic,
    Journal,
    Patch,
    ShadowWorktree,
    DryRun,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub schema_version: String,
    pub status: Status,
    pub transaction_id: String,
    pub workspace: WorkspaceReport,
    pub operations: Vec<OperationReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validators: Vec<ValidatorReport>,
    pub published: bool,
    pub publish_mode: PublishMode,
}

impl Certificate {
    pub fn exit_code(&self) -> i32 {
        self.status.exit_code()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_match_registered_contract() {
        assert_eq!(Status::Applied.exit_code(), 0);
        assert_eq!(Status::AlreadySatisfied.exit_code(), 0);
        assert_eq!(Status::NotFound.exit_code(), 10);
        assert_eq!(Status::Ambiguous.exit_code(), 11);
        assert_eq!(Status::Stale.exit_code(), 12);
        assert_eq!(Status::InvalidResult.exit_code(), 13);
        assert_eq!(Status::ValidationFailed.exit_code(), 14);
        assert_eq!(Status::PublishFailed.exit_code(), 16);
    }

    #[test]
    fn status_serializes_kebab_case() {
        assert_eq!(
            serde_json::to_string(&Status::AlreadySatisfied).unwrap(),
            "\"already-satisfied\""
        );
    }
}
