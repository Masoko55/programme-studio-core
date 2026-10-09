import unittest

from app.services.creative_direction import deduplicate_negative_prompt


class CreativeDirectionDeduplicationTests(unittest.TestCase):
    def test_negative_prompt_keeps_each_exclusion_once(self):
        result = deduplicate_negative_prompt(
            "people, text, people, text, characters"
        )

        self.assertEqual(result, "people, text, characters")


if __name__ == "__main__":
    unittest.main()
