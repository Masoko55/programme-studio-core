import unittest

from app.schemas.final_selection import FinalSelectionRequest
from app.services.final_selection_service import _selection_layout


LAYOUT = {
    "title_zone": {"x": 0.1, "y": 0.245, "width": 0.8, "height": 0.12},
    "programme_zone": {"x": 0.1, "y": 0.43, "width": 0.8, "height": 0.47},
}


def overlaps(first, second):
    return (
        first["x"] < second["x"] + second["width"]
        and first["x"] + first["width"] > second["x"]
        and first["y"] < second["y"] + second["height"]
        and first["y"] + first["height"] > second["y"]
    )


class FinalSelectionLayoutTests(unittest.TestCase):
    def test_right_logo_does_not_overlap_title_or_programme(self):
        result = _selection_layout(
            LAYOUT,
            FinalSelectionRequest(
                engine_id="flux-2",
                direction_id="B",
                logo_path="/tmp/logo.png",
                logo_placement="right",
                rights_and_consent_confirmed=True,
            ),
        )

        self.assertFalse(overlaps(result["logo_zone"], result["title_zone"]))
        self.assertFalse(overlaps(result["logo_zone"], result["programme_zone"]))

    def test_left_headshot_does_not_overlap_title_or_programme(self):
        result = _selection_layout(
            LAYOUT,
            FinalSelectionRequest(
                engine_id="flux-2",
                direction_id="B",
                headshot_path="/tmp/headshot.png",
                headshot_placement="left",
                rights_and_consent_confirmed=True,
            ),
        )

        self.assertFalse(overlaps(result["headshot_zone"], result["title_zone"]))
        self.assertFalse(overlaps(result["headshot_zone"], result["programme_zone"]))


if __name__ == "__main__":
    unittest.main()
