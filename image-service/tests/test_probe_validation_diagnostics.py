"""The probe must distinguish raw QA from production acceptance."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.probe_native_candidate import (
    ProbeValidationCapture, _capture_probe_validation, _diagnostic_summary, _final_validation,
)


class ProbeValidationDiagnosticsTests(unittest.TestCase):
    def test_rejection_retains_distinct_raw_and_full_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "candidate.png"
            source.write_bytes(b"candidate")
            capture = ProbeValidationCapture(Path(directory) / "rejected.png")

            def validator(path, *, enforce_palette=True):
                if not enforce_palette:
                    return {"validation": {"decision": "accepted"}}
                error = ValueError("off-palette ratio 0.16")
                error.diagnostics = {"validation": {"decision": "rejected", "failure_category": "PALETTE_OFF"}}
                raise error

            with patch("scripts.probe_native_candidate.comfyui_module.validate_background", validator):
                with _capture_probe_validation(capture):
                    from app.services import comfyui_client
                    with self.assertRaises(ValueError):
                        comfyui_client.validate_background(source)
            record = {"status": "rejected", "probe_raw_validation": capture.raw_validation,
                      "full_validation": capture.full_validation,
                      "probe_full_validation_error": capture.full_validation_error}
            summary = _diagnostic_summary(reference="test", engine="sdxl-1-0", direction="A", record=record)
            self.assertEqual(summary["raw_non_palette_decision"], "accepted")
            self.assertEqual(summary["full_decision"], "rejected")
            self.assertEqual(summary["full_failure_category"], "PALETTE_OFF")
            self.assertTrue(capture.rejected_image_preserved)

    def test_failed_palette_recovery_overrides_earlier_accepted_diagnostic(self):
        record = {"status": "rejected", "outcome_category": "PALETTE_OFF",
                  "palette_recovery": {"attempted": True, "result": "rejected"},
                  "validation": {"decision": "accepted"}}
        full = _final_validation(record, {"decision": "accepted"})
        self.assertEqual(full["decision"], "rejected")
        self.assertEqual(full["failure_category"], "PALETTE_OFF")
        self.assertTrue(full["recovery_attempted"])

