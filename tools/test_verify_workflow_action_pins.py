from __future__ import annotations

import pathlib
import tempfile
import unittest

import tools.verify_workflow_action_pins as verifier


class WorkflowActionPinTests(unittest.TestCase):
    def workflow(self, content: str) -> pathlib.Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".yml", delete=False
        )
        self.addCleanup(pathlib.Path(temporary.name).unlink, missing_ok=True)
        with temporary:
            temporary.write(content)
        return pathlib.Path(temporary.name)

    def test_commit_pins_and_local_actions_pass(self):
        path = self.workflow(
            "steps:\n"
            "  - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5\n"
            "  - uses: ./.github/actions/local\n"
        )

        self.assertEqual(verifier.verify([path]), [])

    def test_mutable_tag_fails_with_location(self):
        path = self.workflow("steps:\n  - uses: actions/checkout@v4\n")

        errors = verifier.verify([path])

        self.assertEqual(len(errors), 1)
        self.assertIn(f"{path}:2", errors[0])
        self.assertIn("actions/checkout@v4", errors[0])

    def test_immutable_rust_toolchain_action_requires_explicit_repository_channel(self):
        reference = "dtolnay/rust-toolchain@fa04a1451ff1842e2626ccb99004d0195b455a88"
        missing = self.workflow(f"steps:\n  - uses: {reference}\n")
        pinned = self.workflow(
            f"steps:\n  - uses: {reference}\n"
            "    with:\n"
            "      toolchain: '1.95.0'\n"
        )

        errors = verifier.verify([missing])

        self.assertEqual(len(errors), 1)
        self.assertIn("must declare toolchain '1.95.0'", errors[0])
        self.assertEqual(verifier.verify([pinned]), [])


if __name__ == "__main__":
    unittest.main()
