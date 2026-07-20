# Edit transaction core completion

The prior scope block was lifted: this pass was allowed to change all of `crates/edit/`, so all four audited gaps are now closed without changing any existing public signature.

1. Patch mode now places each computed unified diff directly in its `OperationReport.unified_diff` (`crates/edit/src/plan.rs:264-281`) and remains non-publishing.
2. Shadow validation and publication are separated internally (`crates/edit/src/shadow.rs:120-167`), allowing `apply_plan` to recheck preconditions after validation and map final journal CAS failures to a stale certificate instead of returning an error (`crates/edit/src/plan.rs:325-385`).
3. Publication error classification is centralized in `crates/edit/src/certificate.rs:39-48` and used by plan atomic/journal/shadow publishing plus single-verb atomic and rename journal publishing; only CAS mismatches become `stale`/12, while path, lock, and I/O failures become `publish-failed`/16. Success and stale certificate serialization is covered for all four plan publish modes.
4. `expect_git_head` is checked before planning and again after validators, with live heads recorded in the workspace report; `require_unchanged_files` now requires file hash preconditions and re-verifies all planned file snapshots before publication (`crates/edit/src/plan.rs:130-163,346-357,388-448`). Violations return a stale certificate with exit 12.
5. Journal crash injection now covers lock, CAS, journal-directory/pre-image, commit-marker, per-file publish, and cleanup boundaries, while recovery also removes orphan uncommitted journal directories (`crates/edit/src/journal.rs:113-231,266-327`).
6. `crates/edit/tests/txn_property.rs` runs 64 proptest cases that mutate the live target after planning during shadow validation and byte-checks that stale/12 preserves the concurrent bytes; `crates/edit/tests/journal_crash.rs` injects every journal boundary and verifies recovery leaves an all-before or all-after consistent workspace.

Verification baseline was 63 unit tests. The final suite is 70 unit tests plus 2 integration tests (72 total), all passing; `cargo clippy -p greppy-edit -- -D warnings` is clean.
