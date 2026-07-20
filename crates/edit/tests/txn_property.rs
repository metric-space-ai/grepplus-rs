#![cfg(unix)]

use greppy_edit::plan::{
    apply_plan, Plan, PlanAction, PlanOperation, PlanPreconditions, PlanPublish, PlanPublishMode,
    PlanSelector, PlanValidator, PlanWorkspace, PLAN_SCHEMA,
};
use greppy_edit::Status;
use proptest::prelude::*;

const ORIGINAL: &[u8] = b"target = before\n";

fn concurrent_mutation_plan(root: &std::path::Path, mutation: &std::path::Path) -> Plan {
    let target = root.join("target.py");
    Plan {
        schema_version: PLAN_SCHEMA.to_string(),
        workspace: PlanWorkspace {
            root: root.to_string_lossy().into_owned(),
            expect_git_head: None,
            // This property exercises the unconditional publish-time CAS,
            // independently of the optional plan file-hash precondition.
            require_unchanged_files: false,
        },
        operations: vec![PlanOperation {
            id: "replace-target".into(),
            file: "target.py".into(),
            selector: PlanSelector::Text {
                old_text: "before".into(),
                expect: 1,
            },
            action: PlanAction::Replace {
                content: "after".into(),
            },
            preconditions: PlanPreconditions::default(),
        }],
        validators: vec![PlanValidator {
            // Validators run after the immutable snapshots and in-memory
            // application, but before shadow-worktree journal publication.
            // Copying to the absolute live path deterministically injects the
            // concurrent mutation at exactly that boundary.
            argv: vec![
                "cp".into(),
                mutation.to_string_lossy().into_owned(),
                target.to_string_lossy().into_owned(),
            ],
            timeout_seconds: 10,
        }],
        publish: PlanPublish {
            mode: PlanPublishMode::ShadowWorktree,
        },
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn random_live_byte_mutations_are_stale_and_preserved(
        mutation in prop::collection::vec(any::<u8>(), 0..512)
            .prop_filter("mutation must differ from the planned pre-image", |bytes| bytes != ORIGINAL)
    ) {
        let workspace = tempfile::tempdir().unwrap();
        let target = workspace.path().join("target.py");
        let mutation_file = workspace.path().join("mutation.bin");
        std::fs::write(&target, ORIGINAL).unwrap();
        std::fs::write(&mutation_file, &mutation).unwrap();

        let plan = concurrent_mutation_plan(workspace.path(), &mutation_file);
        let certificate = apply_plan(&plan, false).unwrap();

        prop_assert_eq!(certificate.status, Status::Stale);
        prop_assert_eq!(certificate.exit_code(), 12);
        prop_assert!(!certificate.published);
        prop_assert_eq!(std::fs::read(&target).unwrap(), mutation);

        // Serialization round-trip checks all required typed certificate
        // fields remain present on the stale path.
        let json = serde_json::to_value(&certificate).unwrap();
        prop_assert_eq!(json["schema_version"].as_str(), Some("greppy.edit-certificate.v1"));
        prop_assert_eq!(json["status"].as_str(), Some("stale"));
        let _: greppy_edit::Certificate = serde_json::from_value(json).unwrap();
    }
}
