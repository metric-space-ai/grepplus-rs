import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from bench.agent_efficiency import gen_real_tasks as gen


class EnsureMirrorsTests(unittest.TestCase):
    def manifest(self) -> dict:
        return {
            "repos": {
                name: {"commit": f"{index + 1:040x}"}
                for index, name in enumerate(gen.REPO_ORDER)
            }
        }

    @staticmethod
    def fake_copytree(_source: pathlib.Path, destination: pathlib.Path, **_kwargs) -> None:
        destination.mkdir(parents=True)

    def test_index_failure_preserves_bounded_diagnostic(self) -> None:
        failure = subprocess.CompletedProcess(
            args=["greppy", "index"],
            returncode=73,
            stdout="",
            stderr="prefix-" + "x" * gen.INDEX_DIAGNOSTIC_LIMIT + "-useful-tail",
        )
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(gen, "WORK_DIR", pathlib.Path(directory)),
                mock.patch.object(gen.shutil, "copytree", side_effect=self.fake_copytree),
                mock.patch.object(gen.subprocess, "run", return_value=failure) as run,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"greppy index failed for serde with exit 73: .*useful-tail",
                ) as error:
                    gen.ensure_mirrors(self.manifest())

        self.assertLessEqual(
            len(str(error.exception).split(": ", 1)[1]),
            gen.INDEX_DIAGNOSTIC_LIMIT,
        )
        self.assertEqual(run.call_args.kwargs["timeout"], gen.INDEX_TIMEOUT_SECONDS)
        self.assertEqual(run.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(run.call_args.kwargs["stderr"], subprocess.PIPE)

    def test_index_timeout_identifies_repository(self) -> None:
        timeout = subprocess.TimeoutExpired(cmd=["greppy", "index"], timeout=1)
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(gen, "WORK_DIR", pathlib.Path(directory)),
                mock.patch.object(gen.shutil, "copytree", side_effect=self.fake_copytree),
                mock.patch.object(gen.subprocess, "run", side_effect=timeout),
            ):
                with self.assertRaisesRegex(RuntimeError, r"timed out for serde"):
                    gen.ensure_mirrors(self.manifest())


if __name__ == "__main__":
    unittest.main()
