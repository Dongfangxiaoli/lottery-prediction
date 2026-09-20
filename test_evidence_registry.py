from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import evidence_registry as registry


CONFIG = {"n_tickets": 5, "horizon": 1000, "family_size": 7, "alpha": 0.05}
PLW_MODEL = [{"digits": [1, 2, 3, 4, i]} for i in range(5)]
PLW_BASELINE = [{"digits": [5, 6, 7, 8, i]} for i in range(5)]
PLS_MODEL = [{"digits": item} for item in ([1, 2, 3], [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 4])]
PLS_BASELINE = [{"digits": item} for item in ([4, 5, 6], [4, 5, 7], [4, 5, 8], [4, 5, 9], [4, 6, 7])]


class EvidenceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.study = Path(self.temp.name) / "prospective"
        self.before = datetime(2030, 1, 1, 10, 0, tzinfo=registry.LOCAL_TZ)
        self.after = datetime(2030, 1, 3, 10, 0, tzinfo=registry.LOCAL_TZ)

    def tearDown(self):
        self.temp.cleanup()

    def _create(self, study=None):
        with patch.object(registry, "_now_local", return_value=self.before):
            return registry.create_study(study or self.study, CONFIG)

    def _commit(self, issue="30001", target_date="2030-01-02", cutoff_issue="30000",
                cutoff_date="2030-01-01", study=None, game="plw", play="直选",
                model=PLW_MODEL, baseline=PLW_BASELINE):
        with patch.object(registry, "_now_local", return_value=self.before):
            return registry.commit_prediction(study or self.study, game, play, issue, target_date, cutoff_issue, cutoff_date,
                                              model, baseline, {"method": "test"})

    def test_create_commit_and_score_exact_period(self):
        self._create()
        self._commit()
        history = pd.DataFrame({"issue": ["30001"], "date": ["2030-01-02"],
                                "digit1": [1], "digit2": [2], "digit3": [3], "digit4": [4], "digit5": [0]})
        with patch.object(registry, "_now_local", return_value=self.after):
            scored = registry.score_study(self.study, {"plw": history})
        row = next(item for item in scored["records"] if item["sequence"] == 1)
        self.assertEqual(row["status"], "scored")
        self.assertTrue(row["model_first_prize_win"])
        self.assertFalse(row["baseline_first_prize_win"])
        self.assertFalse(scored["eligibility_for_advantage_claim"])  # fixed 1000-period protocol incomplete
        self.assertFalse(scored["horizon_data_complete"])
        self.assertTrue(scored["local_only"])

    def test_rejects_future_cutoff_duplicate_and_duplicate_tickets(self):
        self._create()
        with patch.object(registry, "_now_local", return_value=self.before):
            with self.assertRaisesRegex(ValueError, "严格晚于"):
                registry.commit_prediction(self.study, "plw", "直选", "30001", "2030-01-01", "30000", "2029-12-31",
                                           PLW_MODEL, PLW_BASELINE, {})
            with self.assertRaisesRegex(ValueError, "cutoff_date"):
                registry.commit_prediction(self.study, "plw", "直选", "30002", "2030-01-03", "30001", "2030-01-02",
                                           PLW_MODEL, PLW_BASELINE, {})
            with self.assertRaisesRegex(ValueError, "重复整票"):
                registry.commit_prediction(self.study, "plw", "直选", "30001", "2030-01-02", "30000", "2030-01-01",
                                           [PLW_MODEL[0]] * 5, PLW_BASELINE, {})
        self._commit()
        with self.assertRaisesRegex(ValueError, "只能提交一次"):
            self._commit()

    def test_rejects_protocol_and_bad_record_time_fields(self):
        self._create()
        protocol_path = self.study / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="ascii"))
        protocol["configuration"]["n_tickets"] = 6
        protocol_path.write_text(json.dumps(protocol), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "n_tickets"):
            registry.read_study(self.study)

        protocol_time_study = Path(self.temp.name) / "bad-protocol-time"
        self._create(protocol_time_study)
        protocol_path = protocol_time_study / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="ascii"))
        protocol["created_at_local"] = "not-a-timestamp"
        protocol["protocol_hash"] = registry._hash_without(protocol, "protocol_hash")
        protocol_path.write_text(json.dumps(protocol), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "ISO"):
            registry.read_study(protocol_time_study)

        time_study = Path(self.temp.name) / "bad-time"
        self._create(time_study)
        self._commit(study=time_study)
        record_path = time_study / "records" / "000001.json"
        record = json.loads(record_path.read_text(encoding="ascii"))
        record["created_at_local"] = "not-a-timestamp"
        record["record_hash"] = registry._hash_without(record, "record_hash")
        record_path.write_text(json.dumps(record), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "记录字段"):
            registry.read_study(time_study)

    def test_rejects_date_conflict_and_middle_chain_deletion(self):
        self._create()
        self._commit()
        with patch.object(registry, "_now_local", return_value=self.before):
            with self.assertRaisesRegex(ValueError, "严格递增"):
                registry.commit_prediction(self.study, "plw", "直选", "30002", "2030-01-02", "30001", "2030-01-01",
                                           PLW_MODEL, PLW_BASELINE, {})
        day_two = datetime(2030, 1, 2, 10, 0, tzinfo=registry.LOCAL_TZ)
        with patch.object(registry, "_now_local", return_value=day_two):
            registry.commit_prediction(self.study, "plw", "直选", "30002", "2030-01-03", "30001", "2030-01-02",
                                       PLW_MODEL, PLW_BASELINE, {})
        (self.study / "records" / "000001.json").unlink()
        with self.assertRaisesRegex(ValueError, "序号不连续"):
            registry.read_study(self.study)

    def test_pl3_group_scores_unordered_actual_direct_draw(self):
        pls_study = Path(self.temp.name) / "pls"
        self._create(pls_study)
        self._commit(study=pls_study, game="pls", play="组选6", model=PLS_MODEL, baseline=PLS_BASELINE)
        history = pd.DataFrame({"issue": ["30001"], "date": ["2030-01-02"],
                                "digit1": [3], "digit2": [1], "digit3": [2]})
        with patch.object(registry, "_now_local", return_value=self.after):
            scored = registry.score_study(pls_study, {"pls": history})
        item = next(row for row in scored["records"] if row["sequence"] == 1)
        self.assertEqual(item["status"], "scored")
        self.assertTrue(item["model_first_prize_win"])

    def test_complete_local_data_is_not_an_advantage_claim(self):
        protocol = {"configuration": {"horizon": 1}}
        records = [{"sequence": index, "game": game, "play": play, "target_issue": "1",
                    "target_date": "2030-01-01", "model_tickets": [], "baseline_tickets": []}
                   for index, (game, play) in enumerate(registry.FIXED_FAMILY, start=1)]
        with patch.object(registry, "_load", return_value=(protocol, records)), \
             patch.object(registry, "_history_row", return_value=("matched", object())), \
             patch.object(registry, "_actual_draw", return_value=()), \
             patch.object(registry, "_best_first_prize", return_value=False), \
             patch.object(registry, "_now_local", return_value=self.after):
            scored = registry.score_study(self.study, {})
        self.assertTrue(scored["horizon_data_complete"])
        self.assertFalse(scored["eligibility_for_advantage_claim"])

    def test_record_hash_tampering_is_rejected(self):
        self._create()
        self._commit()
        record_path = self.study / "records" / "000001.json"
        record = json.loads(record_path.read_text(encoding="ascii"))
        record["target_issue"] = "39999"
        record_path.write_text(json.dumps(record), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "自哈希"):
            registry.read_study(self.study)

    def test_missing_expired_period_never_becomes_success(self):
        self._create()
        self._commit()
        with patch.object(registry, "_now_local", return_value=self.after):
            scored = registry.score_study(self.study, {"plw": pd.DataFrame(columns=["issue", "date", "digit1", "digit2", "digit3", "digit4", "digit5"])})
        row = next(item for item in scored["records"] if item["sequence"] == 1)
        self.assertEqual(row["status"], "missing")
        self.assertFalse(scored["eligibility_for_advantage_claim"])

    def test_history_issue_date_conflict_never_scores(self):
        self._create()
        self._commit()
        conflicting = pd.DataFrame({"issue": ["30001"], "date": ["2030-01-03"],
                                    "digit1": [1], "digit2": [2], "digit3": [3], "digit4": [4], "digit5": [0]})
        with patch.object(registry, "_now_local", return_value=self.after):
            scored = registry.score_study(self.study, {"plw": conflicting})
        row = next(item for item in scored["records"] if item["sequence"] == 1)
        self.assertEqual(row["status"], "conflict")
        self.assertFalse(scored["horizon_data_complete"])


if __name__ == "__main__":
    unittest.main()
