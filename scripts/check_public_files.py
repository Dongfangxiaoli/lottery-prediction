"""Check tracked files only; never print matching secret values.

This conservative release guard is not a complete secret scanner or legal audit.
Run after git add and before commit/push. Generated local files stay untracked.
"""
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {
    ".venv", "venv", "runtime", "vendor_cp314", "__pycache__", "data", "models",
    "results", "evidence_runs", "candidate_runs", "logs", "cache", "downloads",
    "installers", "dist", "build",
}
FORBIDDEN_SUFFIXES = {".zip", ".exe", ".dll", ".pth", ".pt", ".pkl", ".pickle", ".joblib", ".pem", ".key", ".pyc", ".log"}
CHECKS = {
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github-token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "api-token": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}"),
    "personal-windows-path": re.compile(rb"[A-Za-z]:[\\/]+Users[\\/]+", re.IGNORECASE),
    "personal-unix-path": re.compile(rb"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
}


def main():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
    names = [name.decode("utf-8") for name in result.stdout.split(b"\0") if name]
    if not names:
        raise SystemExit("No tracked files; stage the public whitelist before checking.")
    failures = []
    for name in names:
        relative = Path(name)
        path = ROOT / relative
        if (set(part.lower() for part in relative.parts) & FORBIDDEN_PARTS
                or relative.suffix.lower() in FORBIDDEN_SUFFIXES
                or relative.name == ".env" or relative.name.startswith(".env.")):
            failures.append((name, "private-or-generated-path"))
            continue
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            failures.append((name, "not-a-normal-project-file"))
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            failures.append((name, "unexpected-large-file"))
            continue
        content = path.read_bytes()
        for label, pattern in CHECKS.items():
            if pattern.search(content):
                failures.append((name, label))
    for name, reason in failures:
        print(f"REJECT {name}: {reason}")
    if failures:
        return 1
    print(f"PUBLIC_FILE_CHECK_OK: {len(names)} tracked files; no prohibited paths or detected secrets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
