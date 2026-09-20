"""Relocatable, local-only entry point. Never installs software or updates data."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parent


def application_version():
    """Read the version embedded by build_portable.py, with source fallback for tests."""
    environment = ROOT / "ENVIRONMENT.json"
    if environment.is_file():
        try:
            version = str(json.loads(environment.read_text(encoding="utf-8"))["application_version"]).strip()
            if version:
                return version
        except (OSError, ValueError, KeyError, TypeError):
            pass
    pyproject = ROOT / "app" / "pyproject.toml"
    try:
        import tomllib
        with pyproject.open("rb") as stream:
            version = str(tomllib.load(stream)["tool"]["poetry"]["version"]).strip()
            if version:
                return version
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return "未知版本"


def _probe_writable(directory):
    directory.mkdir(parents=True, exist_ok=True)
    # A unique, owned probe must never remove a pre-existing user file on collision.
    with tempfile.TemporaryFile(dir=directory, prefix=".lottery_write_test_") as stream:
        stream.write(b"ok")


def check_layout():
    required_files = ("gradio_app.py", "game_config.py", "portfolio_engine.py")
    app = ROOT / "app"
    if not ROOT.is_dir():
        raise RuntimeError("便携包根目录不存在或路径无效，请完整解压 ZIP 后再运行。")
    if not app.is_dir():
        raise RuntimeError("缺少 app 目录，请完整解压 ZIP（不要直接在压缩包内运行）。")
    missing = [name for name in required_files if not (app / name).is_file()]
    if missing:
        raise RuntimeError("app 缺少关键文件：" + ", ".join(missing) + "。请重新完整解压到新目录。")
    for name in required_files:
        try:
            with (app / name).open("rb"):
                pass
        except OSError as exc:
            raise RuntimeError(f"无法读取 app\\{name}，可能没有访问权限：{exc}") from exc
    for relative in ("logs", "cache", "app/docs/report", "app/data", "app/models", "app/results"):
        try:
            _probe_writable(ROOT / relative)
        except OSError as exc:
            raise RuntimeError(f"目录不可写：{relative}。请解压到有写权限的位置（不要覆盖用户文件）：{exc}") from exc


def check_runtime_components():
    if platform.system() != "Windows":
        return
    # Ask the Windows loader, allowing app-local CRT as well as installed CRT.
    import ctypes
    missing = []
    for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        try:
            ctypes.WinDLL(name)
        except OSError:
            missing.append(name)
    if missing:
        raise RuntimeError("系统缺少 Visual C++ 运行组件：" + ", ".join(missing) +
                           "。请运行 installers\\VC_redist.x64.exe 后再自检。")


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def setup():
    check_layout()
    os.chdir(ROOT / "app")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["MPLBACKEND"] = "Agg"
    for env, relative in (("MPLCONFIGDIR", "cache/matplotlib"), ("GRADIO_TEMP_DIR", "cache/gradio")):
        directory = ROOT / relative
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[env] = str(directory)
    (ROOT / "logs").mkdir(exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log = (ROOT / "logs" / f"run_{name}.log").open("x", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)
    print(f"彩票选号工具 {application_version()} 便携版 | Python 和依赖均已内置", flush=True)
    print(f"日志：{log.name}", flush=True)
    if platform.system() != "Windows" or sys.maxsize <= 2**32:
        raise RuntimeError("本包仅支持 Windows 10/11 x64，不支持 32 位或 ARM 原生环境。")
    if not Path(sys.executable).resolve().is_relative_to(ROOT / "runtime"):
        raise RuntimeError("请使用随包的 Start.bat / Check.bat，不要使用电脑上其他 Python。")
    if not sys.flags.no_user_site or any(not Path(p).resolve().is_relative_to(ROOT) for p in sys.path):
        raise RuntimeError("Python 未隔离包外目录，请通过 Start.bat / Check.bat 启动。")


def check():
    print("正在自检；不访问互联网、不训练或覆盖现有模型。", flush=True)
    check_runtime_components()
    manifest = json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    # User data and models are intentionally mutable after first use.
    immutable = [row for row in manifest["files"] if not row["mutable"]]
    for i, row in enumerate(immutable):
        path = ROOT / row["path"]
        if not path.is_file() or path.stat().st_size != row["size"]:
            failures.append(row["path"])
        else:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                    failures.append(row["path"])
        if i % 2000 == 0:
            print(f"文件完整性：{i}/{len(immutable)}", flush=True)
    if failures:
        raise RuntimeError("文件缺失或损坏，请重新完整解压到新目录：" + ", ".join(failures[:10]))
    versions = {}
    for name in ("torch", "numpy", "pandas", "scipy", "sklearn", "xgboost", "matplotlib", "gradio", "requests", "bs4"):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(ROOT / "runtime"):
            raise RuntimeError(f"{name} 错误加载了包外环境")
        versions[name] = getattr(module, "__version__", "installed")
        print(f"依赖通过：{name} {versions[name]}", flush=True)
    import torch
    from lstm_model import build_digit_model
    from portfolio_engine import generate_portfolio
    from game_config import ALL_GAME_CODES, GAME_PLAY_OPTIONS
    torch.set_num_threads(2)
    model = build_digit_model("pls", input_size=4, hidden_size=16, use_attention=False)
    model.train()
    outputs = model(torch.zeros(2, 3, 4))
    loss = sum(value.square().mean() for value in outputs.values())
    loss.backward()
    torch.optim.Adam(model.parameters()).step()
    print("CPU 模型前向、反向与优化步骤通过（仅临时模型）。", flush=True)
    import numpy as np
    from xgboost import XGBClassifier
    tree = XGBClassifier(n_estimators=1, max_depth=1, n_jobs=1, tree_method="hist", random_state=42)
    x = np.array([[0.], [1.], [2.], [3.]], dtype=np.float32)
    tree.fit(x, np.array([0, 1, 0, 1]))
    assert tree.predict_proba(x).shape == (4, 2)
    print("XGBoost CPU 临时训练与预测通过（不写研究模型）。", flush=True)
    for game in ALL_GAME_CODES:
        for play in GAME_PLAY_OPTIONS[game]:
            assert len(generate_portfolio(game, play, 5, "uniform", seed=42)["tickets"]) == 5
    print("五种彩票 / 七种玩法号码生成通过。", flush=True)
    import gradio_app
    ui = gradio_app.build_ui()
    assert ui.config["components"]
    report = {"passed": True, "python": sys.version, "executable": sys.executable,
              "sys_path": sys.path, "versions": versions,
              "immutable_files_verified": len(immutable), "temporary_cpu_training_step": True,
              "temporary_xgboost_training": True, "ui_construction": True, "clean_windows_vm_tested": False}
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (ROOT / "logs" / f"check_{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n自检全部通过！关闭此窗口，双击 Start.bat 即可使用。", flush=True)


def serve(no_browser=False, requested_port=None):
    print("正在加载界面，首次启动可能需要一些时间，请勿关闭窗口……", flush=True)
    import gradio_app
    ports = [requested_port] if requested_port is not None else range(7861, 7881)
    port = None
    for candidate in ports:
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", candidate))
            port = candidate
            break
        except OSError:
            continue
    if port is None:
        raise RuntimeError("本地端口已被占用。请关闭之前启动的本工具窗口，然后重试。")
    print(f"\n浏览器访问：http://127.0.0.1:{port}\n使用期间保留本窗口；退出请按 Ctrl+C。\n", flush=True)
    gradio_app.build_ui().launch(server_name="127.0.0.1", server_port=port,
                                share=False, inbrowser=not no_browser, show_error=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, choices=range(1024, 65536), metavar="PORT")
    args = parser.parse_args()
    try:
        setup()
        check() if args.check else serve(args.no_browser, args.port)
    except KeyboardInterrupt:
        print("已停止。")
    except Exception:
        traceback.print_exc()
        print("\n请保留 logs 中的报错。若提示缺少 MSVCP140/VCRUNTIME，可运行 installers 内的微软运行库安装包。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
