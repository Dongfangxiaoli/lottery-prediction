"""Build a new Windows x64 offline bundle from the verified local environment.

Never overwrites an existing release; preserves all source models/history.
Usage: python packaging/build_portable.py --output ABSOLUTE_NEW_DIRECTORY
       python packaging/build_portable.py --seal ABSOLUTE_EXISTING_DIRECTORY
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / "packaging"


def application_version():
    """Read the source version; keep the staged manifest in sync with the project."""
    pyproject = PROJECT / "pyproject.toml"
    try:
        import tomllib
        with pyproject.open("rb") as stream:
            version = str(tomllib.load(stream)["tool"]["poetry"]["version"]).strip()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"无法从 {pyproject} 读取应用版本") from exc
    if not version or version == "0.0.0":
        raise RuntimeError(f"项目版本无效：{version!r}")
    return version


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ignore(directory, names):
    return [name for name in names if name == "__pycache__" or name.endswith((".pyc", ".pyo", ".pdb"))
            or (Path(directory).name == "lib" and Path(directory).parent.name == "torch" and name.endswith(".lib"))]


def copy(source, target):
    if source.is_dir():
        shutil.copytree(source, target, ignore=ignore)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def verify_installer(path):
    environment = dict(os.environ, LOTTERY_VC_SIGN_PATH=str(path))
    # A PowerShell 7 host can export incompatible module paths to Windows PowerShell 5.
    environment.pop("PSModulePath", None)
    environment.pop("PSMODULEPATH", None)
    command = ("$ErrorActionPreference = 'Stop'; $s = Get-AuthenticodeSignature -LiteralPath $env:LOTTERY_VC_SIGN_PATH; "
               "[pscustomobject]@{Status=[string]$s.Status; Publisher=$s.SignerCertificate.Subject} | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command],
                            env=environment, capture_output=True, text=True, check=True)
    signature = json.loads(result.stdout)
    if signature["Status"] != "Valid" or "CN=Microsoft Corporation," not in signature["Publisher"]:
        raise RuntimeError(f"Microsoft VC installer signature verification failed: {signature}")
    return signature


def stage(output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output}")
    base = Path(sys.base_prefix)
    site = PROJECT / ".venv" / "Lib" / "site-packages"
    if (sys.version_info[:3] != (3, 12, 14) or not (base / "python312.dll").is_file()
            or Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve()):
        raise RuntimeError("Build using the project's verified .venv Windows Python 3.12.14 environment")
    installer = ASSETS / "downloads" / "VC_redist.x64.exe"
    if not installer.is_file():
        raise FileNotFoundError("Download and verify official Microsoft VC x64 installer first")
    signature = verify_installer(installer)
    version = application_version()
    output.mkdir(parents=True)
    runtime = output / "runtime"
    runtime.mkdir()
    for path in base.iterdir():
        if path.is_file() and path.suffix.lower() in (".exe", ".dll", ".txt"):
            copy(path, runtime / path.name)
    for directory in ("DLLs", "tcl", "include", "libs"):
        if (base / directory).is_dir():
            copy(base / directory, runtime / directory)
    lib = runtime / "Lib"
    lib.mkdir()
    for path in (base / "Lib").iterdir():
        if path.name not in ("site-packages", "__pycache__"):
            copy(path, lib / path.name)
    print("STDLIB_COPIED", flush=True)
    copy(site, lib / "site-packages")
    print("PACKAGES_COPIED", flush=True)
    # _pth disables registry/user/environment Python paths; all entries relative to runtime.
    (runtime / "python312._pth").write_text(".\nLib\nDLLs\nLib/site-packages\n../app\n..\nimport site\n", encoding="ascii")
    app = output / "app"
    app.mkdir()
    for source in PROJECT.glob("*.py"):
        copy(source, app / source.name)
    for name in ("data", "models", "results", "使用说明.md", "requirements.txt", "pyproject.toml"):
        copy(PROJECT / name, app / name)
    research = PROJECT / "evidence_runs"
    if research.is_dir():
        for source in research.iterdir():
            if source.is_dir() and (source.name in ("live_data", "studies") or
                                    (re.fullmatch(r"[0-9]{17}", source.name) and (source / "report.json").is_file())):
                copy(source, app / "evidence_runs" / source.name)
    candidates = PROJECT / "candidate_runs"
    if candidates.is_dir():
        for source in candidates.iterdir():
            if source.is_dir() and re.fullmatch(r"[0-9]{17}", source.name) and (source / "report.json").is_file():
                copy(source, app / "candidate_runs" / source.name)
    for name in ("launcher.py", "Start.bat", "Check.bat", "01_先看使用教程.txt"):
        # Batch requires CRLF; tutorial BOM helps classic Windows Notepad.
        source = (ASSETS / name).read_text(encoding="utf-8-sig")
        with (output / name).open("w", encoding="utf-8-sig" if name.endswith(".txt") else "utf-8", newline="\r\n") as stream:
            stream.write(source)
    copy(installer, output / "installers" / installer.name)
    for folder in ("logs", "cache", "app/docs/report"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    distributions = sorted(({"name": d.metadata["Name"], "version": d.version}
                            for d in importlib.metadata.distributions(path=[str(site)])), key=lambda d: d["name"].lower())
    (output / "ENVIRONMENT.json").write_text(json.dumps({
        "application_version": version, "python_version": sys.version,
        "platform": "Windows 10/11 x64", "cpu_only": True,
        "runtime_provenance": "Copy of the project's verified CPython runtime and dependency distributions; licenses retained.",
        "excluded": ["original .venv launcher and pyvenv.cfg", "vendor_cp314", "Python bytecode caches", "Torch static link libraries and debug symbols"],
        "distributions": distributions,
        "vc_installer_url": "https://aka.ms/vc14/vc_redist.x64.exe",
        "vc_installer_sha256": digest(installer),
        "vc_installer_signature": signature,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "第三方组件说明.txt").write_text(
        "本包供现有项目个人迁移使用。\n"
        "Python 许可证位于 runtime/LICENSE.txt；各依赖许可证保留在 runtime/Lib/site-packages 内相应 .dist-info 或组件目录。\n"
        "CPython、PyTorch、Gradio 等第三方组件未修改；仅移除缓存和不用于本工具运行的静态链接库。\n"
        "应用运行完全使用包内 Python，不需要另装 Python、pip、CUDA 或显卡驱动。\n"
        "微软 Visual C++ x64 安装包从官方链接下载，打包前检查 Microsoft Corporation 数字签名。仅缺少系统组件时手动安装。\n"
        "官方来源：https://aka.ms/vc14/vc_redist.x64.exe\n"
        "说明：https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist\n"
        "Python 相对路径隔离说明：https://docs.python.org/3.12/using/windows.html#finding-modules\n"
        "本包不是 Python 官方发布包，不支持用 pip 随意升级其中组件。\n", encoding="utf-8-sig")
    manifest(output)
    print(f"STAGED {output}", flush=True)


def manifest(output):
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "PACKAGE_MANIFEST.json":
            continue
        relative = path.relative_to(output).as_posix()
        if relative.startswith(("logs/", "cache/")) or "__pycache__" in path.parts:
            continue
        rows.append({"path": relative, "size": path.stat().st_size, "sha256": digest(path),
                     "mutable": relative.startswith(("app/data/", "app/models/", "app/results/", "app/docs/report/",
                                                      "app/evidence_runs/studies/", "app/evidence_runs/live_data/"))})
    result = {"created": datetime.now().isoformat(), "files": rows,
              "total_bytes": sum(row["size"] for row in rows),
              "note": "Hashes describe the release snapshot. Data/models/results can change through normal use."}
    (output / "PACKAGE_MANIFEST.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def seal(output):
    if not (output / "runtime" / "python312._pth").is_file():
        raise RuntimeError("Not a staged portable release")
    verify_installer(output / "installers" / "VC_redist.x64.exe")
    target = output.parent / f"{output.name}.zip"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite: {target}")
    rows = manifest(output)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for row in rows:
            archive.write(output / row["path"], f"{output.name}/{row['path']}")
        archive.write(output / "PACKAGE_MANIFEST.json", f"{output.name}/PACKAGE_MANIFEST.json")
        for empty in ("logs/", "cache/", "app/docs/report/"):
            archive.writestr(f"{output.name}/{empty}", "")
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Zip CRC verification failed: {bad}")
    checksum = digest(target)
    target.with_suffix(".zip.sha256.txt").write_text(f"{checksum}  {target.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(target), "bytes": target.stat().st_size,
                      "sha256": checksum, "zip_crc_check": "passed", "files": len(rows) + 1}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--seal", type=Path)
    arguments = parser.parse_args()
    stage(arguments.output.resolve()) if arguments.output else seal(arguments.seal.resolve())
