import unittest

from app.schemas.creative_direction import CreativeDirectionOutput
from app.services.grill_me_semantic import _safe_context
from app.services.validation import validate_direction_diversity


def direction(letter, composition, focal, depth, perimeter, title, programme):
    return CreativeDirectionOutput.model_validate({
        "direction_id": letter, "role": "visual designer",
        "positive_prompt": "Graphic architectural city with web geometry",
        "negative_prompt": "people, text, logos", "title_suggestions": [],
        "short_design_rationale": "Spatial layout",
        "composition_strategy": composition, "focal_strategy": focal,
        "depth_strategy": depth, "border_or_perimeter_strategy": perimeter,
        "title_safe_region_strategy": title, "programme_safe_region_strategy": programme,
        "layout_guidance": {"title_zone": {"x": .1, "y": .08, "width": .8, "height": .12},
                            "programme_zone": {"x": .1, "y": .6, "width": .8, "height": .25}},
    })


class ExactDirectionCases(unittest.TestCase):
    def test_saved_style_three_city_web_plans_need_more_spatial_separation(self):
        plans = [
            direction("A", "Angular Cityscape with Web Geometry", "Dynamic Focal Points with Radial Web Focus",
                      "Layered City Depth with Angular Architecture", "Invisible Perimeter with Radial Web Edges",
                      "Upper-Right Calm Area", "Lower-Left Calm Area"),
            direction("B", "Asymmetrical Skyscraper Cluster with Converging Web Lines",
                      "Off-Center Vertical Cluster with Dynamic Web Focal Point",
                      "Dramatic Atmospheric Perspective with Radial Web Layering",
                      "Fractured Perimeter with Web-Inspired Spikes", "Wide Upper Band", "Right-Side Vertical Column"),
            direction("C", "Central Asymmetrical Tower Cluster with Diagonal Web Patterns",
                      "Diagonal Web Focal Points with Central Dynamic Cluster",
                      "Graduated Depth with Radial Web Layering and Angular Architecture",
                      "Dynamic Perimeter with Web-Inspired Diagonal Lines", "Top Left Quiet Area", "Bottom-Right Quiet Area"),
        ]
        for item in plans[1:]:
            with self.assertRaisesRegex(ValueError, "repeats"):
                validate_direction_diversity(item, [plans[0]])

    def test_paraphrased_central_tower_web_plan_is_duplicate(self):
        plans = [
            direction("A", "radial web around central skyline", "slightly off-centre central skyscraper",
                      "layered city depth", "radial web perimeter", "quiet upper region", "quiet lower programme region"),
            direction("B", "web lines radiating outward from the tower", "tall tower near middle axis",
                      "foreground midground background buildings", "web structure at outer edges", "clean top area", "calm bottom area"),
            direction("C", "circular web around a skyline monument", "primary tower near centre",
                      "layered skyline", "radial filament perimeter", "upper breathing room", "lower breathing room"),
        ]
        for index in (1, 2):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "repeats"):
                validate_direction_diversity(plans[index], plans[:index])

    def test_materially_different_plans_pass(self):
        plans = [
            direction("A", "diagonal asymmetric cluster and dense left mass", "low-angle tower at left",
                      "receding diagonal planes", "open right perimeter", "quiet right area", "quiet lower right"),
            direction("B", "wide panoramic layered skyline with horizontal flow", "distant lateral skyline",
                      "foreground midground distant layers", "open perimeter", "upper sky", "lower edge"),
            direction("C", "street-canyon perspective with vanishing point", "lateral right focal architecture",
                      "receding street perspective", "edge-framed architectural mass", "quiet left opening", "quiet lower left"),
        ]
        for index in (1, 2):
            validate_direction_diversity(plans[index], plans[:index])


class ExactGrillMeBrief(unittest.TestCase):
    def test_complete_superhero_city_brief_needs_no_redundant_questions(self):
        answers = {
            "event_type": "Children birthday party",
            "theme": "Superhero city adventure",
            "age_group": "Children aged 7 to 10",
            "primary_colour": "royal blue", "secondary_colour": "red",
            "creative_description": "A joyful high-energy spiderman-inspired birthday celebration with a dynamic comic-book city atmosphere.",
            "background_inspiration": "A stylised comic-book city skyline with bold angular architecture, spider webs, layered city depth and energetic spatial movement.",
            "theme_reference_treatment": "Use radial web-inspired geometry, angular architecture, speed-line energy and strong blue-red colour relationships without depicting any character.",
        }
        context = _safe_context({}, answers)
        self.assertEqual(context["event_type_status"], "known")
        self.assertEqual(context["theme_status"], "known")
        self.assertTrue(context["reference_detected"])
        self.assertFalse(context["reference_needs_clarification"])
        self.assertTrue(context["background_generation_summary"])
        self.assertFalse({"event_type", "theme", "background_inspiration", "theme_reference_treatment", "timezone"}
                         & {item["field"] for item in context["clarification_questions"]})
        for requirement in ("background only", "no people", "no characters", "no readable text"):
            self.assertIn(requirement, context["generation_constraints"])
