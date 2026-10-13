"""CPU-only P6/P8 model qualification admission and offline planning."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
spec=importlib.util.spec_from_file_location("portable_model_validator",ROOT/"scripts/validate-model.py")
validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)

class ModelValidationTests(unittest.TestCase):
    def test_only_supported_six_and_eight_lane_profiles_are_admitted(self):
        for lanes in (6,8):
            self.assertEqual(validator.qualification_lanes({"parallel":lanes}),lanes)
        for lanes in (1,4,5,7,9,True,False,6.0,"6",None):
            with self.subTest(lanes=lanes),self.assertRaises(RuntimeError):
                validator.qualification_lanes({"parallel":lanes})

    def test_live_idle_health_must_match_the_selected_lane_count(self):
        healthy={"ok":True,"busy":False,"requests_running":0,"requests_total":0,
                 "streams":{"max":6,"decoding":0,"prefilling":0,"filling":0,"paused":0}}
        validator.check_idle(healthy,6)
        for field,value in (("busy",True),("requests_running",1)):
            wrong=copy.deepcopy(healthy);wrong[field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):
                validator.check_idle(wrong,6)
        with self.assertRaises(RuntimeError):
            validator.check_idle(healthy,8)

    def test_offline_plans_use_selected_concurrency_without_host_or_api_actions(self):
        cfg=validator.prepare_config(ROOT/"config.example.json","rocenante")
        for lanes,count in ((6,24),(8,30)):
            selected=copy.deepcopy(cfg);selected["profile"]["parallel"]=lanes
            output=Path(selected["state_root"])/"test-model-plan-unused"
            args=["validate-model.py","--config","unused","--mode","rocenante","--output",str(output),"--plan"]
            stream=io.StringIO()
            with patch.object(sys,"argv",args),patch.object(validator,"prepare_config",return_value=selected), \
                 patch("subprocess.run",side_effect=AssertionError("Host operations forbidden")), \
                 patch("urllib.request.urlopen",side_effect=AssertionError("API calls forbidden")), \
                 patch.object(Path,"write_text",side_effect=AssertionError("Output writes forbidden")), \
                 patch.object(Path,"mkdir",side_effect=AssertionError("Directory writes forbidden")), \
                 contextlib.redirect_stdout(stream):
                self.assertEqual(validator.main(),0)
            plan=json.loads(stream.getvalue())
            self.assertTrue(plan["offline"])
            self.assertEqual(plan["concurrency"],lanes)
            self.assertEqual(plan["chat_requests"],count)
            self.assertEqual(plan["profile"]["parallel"],lanes)

if __name__=="__main__":
    unittest.main()
