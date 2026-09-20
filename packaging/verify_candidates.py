"""Run using the portable Python -I, validate actual released candidate artifacts."""
import json
from pathlib import Path

import launcher

launcher.setup()
import numpy as np
from candidate_models import XGBoostHeads, MarkovHeads, summarize_windows
from candidate_ui import comparison_view
from evidence_service import checked_study
from game_config import ALL_GAME_CODES, GAME_PLAY_OPTIONS
from jackpot_evaluation import _build_features
from jackpot_selection import select_top_k
from local_history import read_local_history
from train import _labels_to_targets
from run_candidate_experiment import RUNS, _canonical
from candidate_results import read_report
from run_evidence_experiment import digest


def verify():
    completed = sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / "report.json").is_file())
    folder = completed[-1]
    report = read_report(folder)
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    for name, expected in protocol["source_sha256"].items():
        assert digest(launcher.ROOT / "app" / name) == expected, name
    state, reference = checked_study()
    outputs = comparison_view(folder)
    assert len(outputs[1]) == 7 and len(outputs[2]) == 38 and len(outputs[3]) == 20
    for game in ALL_GAME_CODES:
        frame = read_local_history(game, folder / "snapshots")
        old = json.loads((reference / f"{game}_result.json").read_text(encoding="utf-8"))
        saved = json.loads((folder / f"{game}_result.json").read_text(encoding="utf-8"))
        features, labels, _ = _build_features(frame, game, old["feature_fit_end"])
        targets = _labels_to_targets(labels, game)
        indices = np.array([len(frame) - 300, len(frame) - 1])
        models = folder / "models" / game
        xgb = XGBoostHeads.load(models / "xgboost")
        markov = MarkovHeads.load(models / "markov")
        heads = {"xgboost": xgb.predict_heads(summarize_windows(features, indices)),
                 "markov": markov.predict_heads({name: values[indices - 1] for name, values in targets.items()})}
        for strategy, probabilities in heads.items():
            for output_index, saved_index in enumerate((0, 299)):
                for play in GAME_PLAY_OPTIONS[game]:
                    predicted = select_top_k(game, {name: values[output_index] for name, values in probabilities.items()}, 5, play)
                    assert _canonical(predicted, game, play) == _canonical(saved["plays"][play][saved_index]["tickets"][strategy], game, play)
        print("PORTABLE_CANDIDATE_RESTORE_REPLAY_OK", game, flush=True)
    result = {"passed": True, "run": folder.name, "comparisons": len(report["summary"]),
              "bias_tests": len(report["bias_diagnostics"]), "old_prospective_records": len(state["records"]),
              "new_models": "five games x two models; first and last test draws replayed identically",
              "executable": str(Path(__import__("sys").executable)), "clean_windows_vm_tested": False}
    destination = launcher.ROOT / "logs" / "candidate_acceptance.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    verify()
