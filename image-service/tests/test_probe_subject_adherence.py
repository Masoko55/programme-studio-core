"""CLI diagnostics use the production vision checker without rendering."""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

from scripts.probe_subject_adherence import main, parse_args


class ProbeSubjectAdherenceTests(unittest.TestCase):
    def test_argument_parsing_and_output(self):
        args = ["--image", "/tmp/existing.png", "--subject", "city skyline",
                "--anchor", "angular towers", "--anchor", "web geometry", "--model", "qwen3-vl:30b"]
        parsed = parse_args(args)
        self.assertEqual(parsed.anchor, ["angular towers", "web geometry"])
        output = io.StringIO()
        result = {"checked": False, "score": None, "decision": "unavailable",
                  "reason": "Empty response", "model": "qwen3-vl:30b",
                  "vision_debug": {"thinking_present": True, "response_length": 0}}
        with patch("scripts.probe_subject_adherence.assess_subject_adherence", return_value=result) as check:
            with contextlib.redirect_stdout(output):
                main(args)
        self.assertEqual(check.call_args.args[1]["visual_anchors"], parsed.anchor)
        self.assertEqual(json.loads(output.getvalue()), result)

