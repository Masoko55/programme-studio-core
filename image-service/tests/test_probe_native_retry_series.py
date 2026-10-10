import asyncio
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

from app.services.comfyui_client import ComfyUIError
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
    def test_retry_keeps_inspiration_and_corrects_the_actual_sdxl_fields(self):
        document = document_for(CASES[0], "B")
        records = iter((
            {"status": "rejected", "rejection_reason":
             "Generated background contains colours that were not requested"},
            {"status": "complete"},
        ))
        FakeClient.calls = []
        with tempfile.TemporaryDirectory() as temporary:
            settings = SimpleNamespace(
                programme_data_path=Path(temporary),
                max_candidate_retries=8,
            )
            args = SimpleNamespace(
                source_reference="ABCDEF-123456",
                engine="sdxl-1-0", direction="B", attempts=2, seed=123,
            )
            with patch.object(series, "_parse_args", return_value=args), \
                 patch.object(series, "settings", settings), \
                 patch.object(series, "_new_reference", return_value="FEDCBA-654321"), \
                 patch.object(series, "load_prompts_document", return_value=document), \
                 patch.object(series, "write_json"), \
                 patch.object(series, "ComfyUIClient", FakeClient), \
                 patch.object(series, "_capture_probe_validation", return_value=nullcontext()), \
                 patch.object(series, "_load_record", return_value={}), \
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
            self.assertIn("Correction:", field)
