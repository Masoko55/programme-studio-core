import asyncio
import contextlib
import io
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import probe_native_retry_series as series

from app.services.comfyui_client import ComfyUIError, build_workflow
from app.services.candidate_spec import build_candidate_spec
from app.services.image_execution import _attempt_prompts
from app.services.sampling_profiles import summarize_failures
from test_candidate_spec_compiler import CASES, document_for


class FakeClient:
    calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def health(self):
        return None

    async def generate_image(self, *args, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise ComfyUIError("Generated background contains colours that were not requested")
        return {"output_path": "/nonexistent/image.png"}


class RetrySeriesTests(unittest.TestCase):
    def test_production_and_probe_retry_workflows_match_for_every_engine(self):
        reason = "PALETTE_ROLE_DRIFT: requested secondary colour is absent"
        previous = {"rejected_attempts": [{"attempt_count": 1, "rejection_reason": reason}]}
        for engine in ("flux-2", "sdxl-1-0", "sd-3-5-medium"):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as temporary:
                document = document_for(CASES[0], "B")
                FakeClient.calls = []
                args = SimpleNamespace(source_reference="ABCDEF-123456", engine=engine,
                                       direction="B", attempts=2, seed=123, working_reference=None)
                with patch.object(series, "_parse_args", return_value=args), \
                     patch.object(series, "settings", SimpleNamespace(
                         programme_data_path=Path(temporary), max_candidate_retries=8)), \
                     patch.object(series, "_new_reference", return_value="FEDCBA-654321"), \
                     patch.object(series, "load_prompts_document", return_value=document), \
                     patch.object(series, "write_json"), \
                     patch.object(series, "ComfyUIClient", FakeClient), \
                     patch.object(series, "_capture_probe_validation", return_value=nullcontext()), \
                     patch.object(series, "_load_record", side_effect=[{}, previous]), \
                     patch.object(series, "_persist_probe_diagnostics", side_effect=[
                         {"status": "rejected", "rejection_reason": reason}, {"status": "complete"}]), \
                     patch.object(series, "select_sampling_profile", return_value=None), \
                     contextlib.redirect_stdout(io.StringIO()):
                    asyncio.run(series.main())
                spec = build_candidate_spec(document, engine, "B")
                expected = _attempt_prompts(spec, None, 2, reason, ("", ""), ("", ""),
                                            (None, None), summarize_failures(previous))[2]
                actual = FakeClient.calls[1]["compiled_prompt"]
                for field in ("failure_category", "retry_stage", "positive", "negative", "clip_l",
                              "clip_g", "t5", "retry_strength_level", "spec_sha256",
                              "semantic_contract_sha256"):
                    self.assertEqual(getattr(actual, field), getattr(expected, field), field)
                probe_workflow = build_workflow(engine, actual.positive, actual.negative, 123,
                                                "offline", compiled_prompt=actual)
                production_workflow = build_workflow(engine, expected.positive, expected.negative,
                                                     123, "offline", compiled_prompt=expected)
                if engine == "flux-2":
                    nodes = (("4", "text"),)
                elif engine == "sdxl-1-0":
                    nodes = (("16", "text_l"), ("16", "text_g"), ("40", "text"))
                else:
                    nodes = tuple((node, field) for node in ("16", "40")
                                  for field in ("clip_l", "clip_g", "t5xxl"))
                for node, field in nodes:
                    self.assertEqual(probe_workflow[node]["inputs"][field],
                                     production_workflow[node]["inputs"][field])

    def test_all_nine_retryable_attempts_use_fresh_seeds(self):
        reason = "PROTECTED_REGION_INTRUSION"

        class RejectingClient(FakeClient):
            async def generate_image(self, *args, **kwargs):
                self.calls.append(kwargs)
                raise ComfyUIError(reason)

        RejectingClient.calls = []
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(source_reference="ABCDEF-123456", engine="sdxl-1-0",
                                   direction="B", attempts=9, seed=20261010, working_reference=None)
            with patch.object(series, "_parse_args", return_value=args), \
                 patch.object(series, "settings", SimpleNamespace(
                     programme_data_path=Path(temporary), max_candidate_retries=8)), \
                 patch.object(series.settings, "engine_1_id", "flux-2", create=True), \
                 patch.object(series, "_new_reference", return_value="FEDCBA-654321"), \
                 patch.object(series, "load_prompts_document",
                              return_value=document_for(CASES[0], "B")), \
                 patch.object(series, "write_json"), \
                 patch.object(series, "ComfyUIClient", RejectingClient), \
                 patch.object(series, "_capture_probe_validation", return_value=nullcontext()), \
                 patch.object(series, "_load_record", return_value={}), \
                 patch.object(series, "_persist_probe_diagnostics",
                              return_value={"status": "rejected", "rejection_reason": reason}), \
                 patch.object(series, "select_sampling_profile", return_value=None):
                with self.assertRaises(SystemExit) as exit_result:
                    asyncio.run(series.main())
        self.assertEqual(exit_result.exception.code, 1)
        self.assertEqual([call["seed_override"] for call in RejectingClient.calls],
                         list(range(20261010, 20261019)))

    def test_retry_keeps_inspiration_and_corrects_the_actual_sdxl_fields(self):
        document = document_for(CASES[0], "B")
        reason = "Generated background contains colours that were not requested"
        previous = {"rejected_attempts": [{"attempt_count": 1, "rejection_reason": reason}]}
        records = iter((
            {"status": "rejected", "rejection_reason": reason},
            {"status": "complete"},
        ))
        FakeClient.calls = []
        with tempfile.TemporaryDirectory() as temporary:
            settings = SimpleNamespace(
                programme_data_path=Path(temporary),
                max_candidate_retries=8,
                engine_1_id="flux-2",
            )
            args = SimpleNamespace(
                source_reference="ABCDEF-123456",
                engine="sdxl-1-0", direction="B", attempts=2, seed=123, working_reference=None,
            )
            with patch.object(series, "_parse_args", return_value=args), \
                 patch.object(series, "settings", settings), \
                 patch.object(series, "_new_reference", return_value="FEDCBA-654321"), \
                 patch.object(series, "load_prompts_document", return_value=document), \
                 patch.object(series, "write_json"), \
                 patch.object(series, "ComfyUIClient", FakeClient), \
                 patch.object(series, "_capture_probe_validation", return_value=nullcontext()), \
                 patch.object(series, "_load_record", side_effect=[{}, previous]), \
                 patch.object(series, "_persist_probe_diagnostics", side_effect=records), \
                 patch.object(series, "select_sampling_profile", return_value=None):
                asyncio.run(series.main())

        self.assertEqual(len(FakeClient.calls), 2)
        first, second = FakeClient.calls
        self.assertEqual(first["spec_sha256"], second["spec_sha256"])
        self.assertEqual(first["seed_override"], 123)
        self.assertEqual(second["seed_override"], 124)
        self.assertEqual(second["retry_stage"], "normal")
        for field in (second["compiled_prompt"].clip_g, second["compiled_prompt"].clip_l):
            self.assertIn(CASES[0][2], field)
        self.assertIn("normal:", second["compiled_prompt"].clip_g)
        spec = build_candidate_spec(document, "sdxl-1-0", "B")
        expected = _attempt_prompts(spec, None, 2, reason, ("", ""), ("", ""),
                                    (None, None), summarize_failures(previous))[2]
        actual = second["compiled_prompt"]
        for field in ("failure_category", "retry_stage", "positive", "negative", "clip_l",
                      "clip_g", "t5", "retry_strength_level", "spec_sha256",
                      "semantic_contract_sha256"):
            with self.subTest(field=field):
                self.assertEqual(getattr(actual, field), getattr(expected, field))
