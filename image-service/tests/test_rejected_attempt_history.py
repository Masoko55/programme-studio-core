import unittest

from app.services.comfyui_client import ComfyUIClient


class RejectedAttemptHistoryTests(unittest.TestCase):
    def test_historical_attempts_keep_separate_validation_evidence(self):
        client = ComfyUIClient()
        first = {
            "attempt_count": 1, "seed": 101, "rejection_reason": "palette drift",
            "primary_failure_category": "PALETTE_OFF",
            "all_failure_categories": ["PALETTE_OFF"],
            "validation": {"decision": "rejected", "palette": {"off_palette_ratio": .3}},
        }
        history, _ = client._prior_attempts(first, True, False)
        second = {
            "attempt_count": 2, "seed": 202, "rejection_reason": "generated text",
            "primary_failure_category": "TEXT", "all_failure_categories": ["TEXT"],
            "validation": {"decision": "rejected", "text": {"confirmed_tokens": ["TITLE"]}},
            "rejected_attempts": history,
        }
        history, _ = client._prior_attempts(second, True, False)
        self.assertEqual(history[0]["validation"]["palette"]["off_palette_ratio"], .3)
        self.assertEqual(history[0]["primary_failure_category"], "PALETTE_OFF")
        self.assertEqual(history[1]["validation"]["text"]["confirmed_tokens"], ["TITLE"])
        self.assertEqual(history[1]["primary_failure_category"], "TEXT")
