"""Fast packaging checks; no original data is changed or installations performed."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_script("launcher")
builder = load_script("build_portable")


class LauncherTests(unittest.TestCase):
    def test_version_follows_metadata_and_source(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(builder, "PROJECT", Path(tmp)):
            (Path(tmp) / "pyproject.toml").write_text('[tool.poetry]\nversion = "1.2.3"\n')
            self.assertEqual(builder.application_version(), "1.2.3")
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, "ROOT", Path(tmp)):
            (Path(tmp) / "ENVIRONMENT.json").write_text(json.dumps({"application_version": "0.8.0"}))
            self.assertEqual(launcher.application_version(), "0.8.0")

    def test_layout_errors_and_non_destructive_write_probe(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, "ROOT", Path(tmp)):
            with self.assertRaisesRegex(RuntimeError, "app"):
                launcher.check_layout()
            app = Path(tmp) / "app"; app.mkdir()
            for name in ("gradio_app.py", "game_config.py", "portfolio_engine.py"):
                (app / name).write_text("# test")
            marker = app / f".write_test_{os.getpid()}"
            marker.write_text("user-owned sentinel")
            launcher._probe_writable(app)
            self.assertEqual(marker.read_text(), "user-owned sentinel")
            launcher.check_layout()
            with patch.object(launcher, "_probe_writable", side_effect=PermissionError("test")):
                with self.assertRaisesRegex(RuntimeError, "目录不可写"):
                    launcher.check_layout()

    @unittest.skipUnless(os.name == "nt", "Windows CRT loader")
    def test_crt_probe_uses_loader_not_system32_file_presence(self):
        with patch("ctypes.WinDLL", return_value=object()) as probe:
            launcher.check_runtime_components()
            self.assertEqual(probe.call_count, 3)
        with patch("ctypes.WinDLL", side_effect=OSError("missing")):
            with self.assertRaisesRegex(RuntimeError, "Visual C"):
                launcher.check_runtime_components()

    @unittest.skipUnless(os.name == "nt", "Windows batch exit status")
    def test_missing_runtime_batch_returns_nonzero_with_space_and_bang_path(self):
        with tempfile.TemporaryDirectory(prefix="lottery ! test ") as tmp:
            for name in ("Start.bat", "Check.bat"):
                target = Path(tmp) / name
                shutil.copy2(Path(__file__).parent / name, target)
                result = subprocess.run([os.environ["COMSPEC"], "/d", "/c", f'call "{target}"'],
                    stdin=subprocess.DEVNULL, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 1, (result.stdout, result.stderr))


if __name__ == "__main__":
    unittest.main()
