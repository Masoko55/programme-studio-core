"""Metamorphic, contract, quality and reproducibility checks without rendering."""

import importlib.util
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.candidate_quality import hard_validity, rank_candidates, score_candidate, validator_availability
from app.services.candidate_spec import build_candidate_spec
from app.services.prompt_compiler import compile_candidate_prompt, failure_category
from app.services.reproducibility import build_reproducibility_manifest, manifest_hash
from app.services.sampling_profiles import FailureHistorySummary, select_sampling_profile
from app.services.semantic_contract import normalize_contract, semantic_completeness, semantic_consistency
from tests.fixtures.fuzz_cases import cases as fuzz_cases
from tests.fixtures.holdout_cases import CASES as HOLDOUT_CASES
from tests.test_renderer_brief_matrix import fixture


ROOT = Path(__file__).resolve().parents[2]


def document():
    data = fixture(999, "unknown gathering", "floating mineral forms", "navy", "gold",
                   "dominant", "none", "asymmetric")
    data["brief"].update(background_inspiration="floating mineral forms",
                         visual_semantics={"atmosphere_descriptors": ["quietly luminous", "meditative", "weightless"],
                           "audience_descriptors": ["curious visitors"],
                           "motif_descriptors": ["mineral clusters"],
                           "motif_structure_description": "nested translucent mineral clusters",
                           "lighting_descriptors": ["diffuse glow"],
                           "incompatible_treatments": ["harsh metallic glare"],
                           "scene_family": "abstract"})
    return data


class SemanticFoundationTests(unittest.TestCase):
    def test_production_candidate_record_persistence(self):
        from app.services import comfyui_client
        spec = build_candidate_spec(document(), "sdxl-1-0", "A")
        compiled = compile_candidate_prompt(spec, 1)
        base = {
            "reference_number": "TEST", "engine_id": "sdxl-1-0", "direction_id": "A",
            "semantic_contract_version": "1.0",
            "semantic_contract_sha256": spec.semantic_contract_sha256,
            "candidate_spec_sha256": spec.spec_sha256,
            "compiled_prompt": compiled.record(), "workflow_sha256": "workflow",
            "sampling_profile": {"name": "baseline"}, "attempt_count": 1,
        }
        valid = {**base, "status": "complete", "sha256": "image",
                 "validation": {"decision": "accepted", "quality": {"macro_structure_score": .7}},
                 "brief_fidelity": {"decision": "pass", "subject_score": .9,
                                    "atmosphere_score": .8, "motif_score": .8,
                                    "audience_fit_score": .8}}
        for record in (valid, {**base, "status": "rejected", "primary_failure_category": "PALETTE_OFF"},
                       {**base, "status": "runtime_failed", "error": "ComfyUI timeout"}):
            with self.subTest(record=record["status"]), patch.object(comfyui_client, "atomic_write_json") as persist:
                comfyui_client.write_json(Path("candidate.json"), record)
                saved = persist.call_args.args[1]
                self.assertEqual(saved["semantic_contract_sha256"], spec.semantic_contract_sha256)
                self.assertEqual(saved["reproducibility_manifest"]["candidate_spec_sha256"], spec.spec_sha256)
                self.assertNotIn("quality_evaluation", saved)
                self.assertIn("validation", saved)

    def test_version_hash_provenance_confidence_and_change_detection(self):
        spec_path = ROOT / "prompt-service/app/services/semantic_contract.py"
        module_spec = importlib.util.spec_from_file_location("prompt_semantic_contract", spec_path)
        producer = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(producer)
        data = document()
        data["brief"]["grill_me_context"] = {"semantic_provenance": {"atmosphere_descriptors": "SEMANTIC_INFERENCE"},
                                               "semantic_confidence": {"atmosphere_descriptors": .81}}
        contract = producer.build_semantic_contract(data["brief"])
        data["brief"].update(semantic_contract=contract,
                             semantic_contract_sha256=producer.contract_hash(contract))
        spec = build_candidate_spec(data, "sdxl-1-0", "A")
        self.assertEqual(spec.semantic_contract_sha256, producer.contract_hash(contract))
        self.assertEqual(spec.semantic_contract["semantic_contract_version"], "1.0")
        self.assertEqual(spec.semantic_contract["semantic_provenance"]["atmosphere_descriptors"], "SEMANTIC_INFERENCE")
        self.assertEqual(spec.semantic_contract["semantic_confidence"]["atmosphere_descriptors"], .81)
        data["brief"]["visual_semantics"]["atmosphere_descriptors"] = ["somber"]
        updated = build_candidate_spec(data, "sdxl-1-0", "A")
        self.assertEqual(updated.visual_design["visual_semantics"]["atmosphere_descriptors"], ["somber"])
        data["brief"]["visual_semantics"]["atmosphere_descriptors"] = ["quietly luminous", "meditative", "weightless"]
        data["brief"]["primary_colour"] = "red"
        updated = build_candidate_spec(data, "sdxl-1-0", "A")
        self.assertEqual(updated.visual_design["requested_palette"][0]["colour"], "red")

    def test_generic_contradictions_and_completeness(self):
        base = build_candidate_spec(document(), "sdxl-1-0", "A").semantic_contract
        self.assertTrue(semantic_completeness(base)["ready"])
        conflicts = [
            {"atmosphere_descriptors": ["calm chaotic"]},
            {"density_preference": "minimal extremely dense"},
            {"realism_level": "photorealistic", "abstraction_level": "flat vector only"},
            {"palette_mode": "strict", "palette": [{"colour": name} for name in ("black", "white", "orange")]},
            {"primary_visual_subject": "visible character portrait", "incompatible_treatments": ["no characters"]},
        ]
        for update in conflicts:
            with self.subTest(update=update):
                self.assertFalse(semantic_consistency({**base, **update})["valid"])
        absent = semantic_completeness({**base, "motif_descriptors": [], "motif_structure_description": ""})
        self.assertEqual(absent["fields"]["motif_descriptors"], "OPTIONAL_NOT_PROVIDED")

    def test_paraphrase_and_metadata_invariance(self):
        left = document()
        right = json.loads(json.dumps(left))
        right["brief"].update(event_type="a different description of the same gathering",
                              creative_description="A calming display of gently glowing mineral structures",
                              event_date="2035-02-01", venue="another venue", start_time="19:30",
                              timezone="UTC", title_preference="Different title")
        a = build_candidate_spec(left, "sdxl-1-0", "A")
        b = build_candidate_spec(right, "sdxl-1-0", "A")
        self.assertEqual(a.semantic_contract_sha256, b.semantic_contract_sha256)
        self.assertEqual(a.semantic_contract, b.semantic_contract)

    def test_dimension_mutations_are_isolated(self):
        baseline = document()
        base_contract = build_candidate_spec(baseline, "sdxl-1-0", "A").semantic_contract
        mood = json.loads(json.dumps(baseline))
        mood["brief"]["visual_semantics"].update(atmosphere_descriptors=["somber"],
                                                   lighting_descriptors=["dim"],
                                                   incompatible_treatments=["cheerful glare"])
        mood_contract = build_candidate_spec(mood, "sdxl-1-0", "A").semantic_contract
        self.assertNotEqual(base_contract["atmosphere_descriptors"], mood_contract["atmosphere_descriptors"])
        for field in ("primary_visual_subject", "motif_structure_description", "palette", "scene_family"):
            self.assertEqual(base_contract[field], mood_contract[field])
        palette = json.loads(json.dumps(baseline))
        palette["brief"].update(primary_colour="green", secondary_colour="cream")
        other = build_candidate_spec(palette, "sdxl-1-0", "A")
        self.assertNotEqual(base_contract["palette"], other.semantic_contract["palette"])
        self.assertNotEqual(compile_candidate_prompt(build_candidate_spec(baseline, "sdxl-1-0", "A"), 1).clip_g,
                            compile_candidate_prompt(other, 1).clip_g)
        for field in ("primary_visual_subject", "motif_structure_description", "atmosphere_descriptors", "scene_family"):
            self.assertEqual(base_contract[field], other.semantic_contract[field])
        audience = json.loads(json.dumps(baseline))
        audience["brief"]["visual_semantics"]["audience_descriptors"] = ["adults"]
        self.assertEqual(base_contract["primary_visual_subject"],
                         build_candidate_spec(audience, "sdxl-1-0", "A").semantic_contract["primary_visual_subject"])

    def test_engine_and_direction_semantic_invariance_and_retry_lock(self):
        data = document()
        for letter, structure in (("B", "diagonal layered spatial composition"),
                                  ("C", "wide panoramic spatial composition")):
            variant = json.loads(json.dumps(data["directions"][0]))
            variant["direction"]["direction_id"] = letter
            variant["direction"]["composition_strategy"] = structure
            data["directions"].append(variant)
        hashes = set()
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            for direction in "ABC":
                spec = build_candidate_spec(data, engine, direction)
                hashes.add(spec.semantic_contract_sha256)
                initial = compile_candidate_prompt(spec, 1)
                retry = compile_candidate_prompt(spec, 2, "ATMOSPHERE_DRIFT")
                self.assertEqual(initial.semantic_contract_sha256, retry.semantic_contract_sha256)
                self.assertIn("quietly luminous", initial.positive)
                self.assertIn("mineral clusters", initial.positive)
                self.assertIn("navy", initial.positive)
                self.assertIn("curious visitors", initial.positive)
                self.assertIn("abstract", initial.positive)
                self.assertEqual(retry.failure_category, "ATMOSPHERE_DRIFT")
                self.assertEqual(retry.incompatible_treatments, initial.incompatible_treatments)
        self.assertEqual(len(hashes), 1)

    def test_arbitrary_language_survives_each_engine(self):
        words = "ethereal earthy intimate nostalgic ceremonial kinetic tranquil immersive weathered delicate monumental airy"
        data = document()
        data["brief"]["visual_semantics"]["atmosphere_descriptors"] = [words]
        for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
            prompt = compile_candidate_prompt(build_candidate_spec(data, engine, "A"), 1)
            active = prompt.positive if engine == "flux-2" else prompt.clip_g
            for word in words.split():
                self.assertIn(word, active)
            self.assertFalse(prompt.truncated)

    def test_quality_hard_gate_and_missing_soft_signals(self):
        valid = {"reference_number": "A", "validation": {"decision": "accepted", "quality": {"macro_structure_score": .7}},
                 "brief_fidelity": {"decision": "pass", "subject_score": .9, "atmosphere_score": .7,
                                    "motif_score": .65, "audience_fit_score": .6}}
        bad = {**valid, "reference_number": "B", "validation": {"decision": "rejected", "failure_category": "TEXT"}}
        self.assertFalse(hard_validity(bad)["passed"])
        self.assertIsNone(score_candidate(bad)["quality"])
        self.assertIsNotNone(score_candidate(valid)["quality"]["overall_quality"])
        self.assertEqual(rank_candidates([bad, valid])["chosen_reference"], "A")
        tied = {**valid, "reference_number": "C"}
        self.assertEqual(rank_candidates([tied, valid])["chosen_reference"], "A")
        no_signal = {"reference_number": "Z", "validation": {"decision": "accepted"}}
        self.assertIsNone(rank_candidates([no_signal])["chosen_reference"])
        semantically_poor = {**valid, "brief_fidelity": {**valid["brief_fidelity"],
                           "atmosphere_score": .3, "motif_score": .35, "audience_fit_score": .4}}
        self.assertFalse(hard_validity(semantically_poor)["passed"])
        partial = {"validation": {"decision": "accepted", "quality": {"macro_structure_score": .8}}}
        self.assertEqual(score_candidate(partial)["quality"]["availability"]["subject_fidelity"], "UNAVAILABLE")
        self.assertFalse(hard_validity(partial, require_vision=True)["passed"])
        partial["validation"]["text"] = {"ocr_diagnostics": {"decision": "OCR_PARTIAL"}}
        self.assertEqual(validator_availability(partial)["ocr"], "PARTIAL")

    def test_reproducibility_manifest_is_deterministic(self):
        spec = build_candidate_spec(document(), "sdxl-1-0", "A")
        compiled = compile_candidate_prompt(spec, 1)
        record = {"reference_number": spec.reference_number, "engine_id": spec.engine_id,
                  "direction_id": "A", "semantic_contract_version": "1.0",
                  "semantic_contract_sha256": spec.semantic_contract_sha256,
                  "candidate_spec_sha256": spec.spec_sha256, "seed": 123,
                  "workflow_sha256": "abc", "compiled_prompt": compiled.record(),
                  "sampling_profile": {"name": "baseline", "steps": 28}}
        first = build_reproducibility_manifest(record)
        second = build_reproducibility_manifest(json.loads(json.dumps(record)))
        self.assertEqual(first, second)
        self.assertEqual(manifest_hash(first), manifest_hash(second))
        self.assertEqual(first["semantic_contract_sha256"], spec.semantic_contract_sha256)

    def test_holdout_and_fuzz_compile_without_case_rules(self):
        self.assertEqual(len(HOLDOUT_CASES), 50)
        sys.path.insert(0, str(ROOT / "image-service/scripts"))
        from audit_genericity import _document, audit_briefs
        self.assertFalse(audit_briefs(HOLDOUT_CASES))
        self.assertFalse(audit_briefs(fuzz_cases()))
        for index, case in enumerate(fuzz_cases(12)):
            data = _document(index, case)
            if index % 3 == 0:
                data["brief"].pop("symbolic_motifs", None)
            if index % 4 == 0:
                data["brief"]["palette_mode"] = "strict"
            if index % 5 == 0:
                data["brief"]["palette"] = [{"colour": "black", "role": "primary"},
                                               {"colour": "white", "role": "secondary"}]
            spec = build_candidate_spec(data, "flux-2", "A")
            prompt = compile_candidate_prompt(spec, 1)
            self.assertIn(spec.semantic_contract["primary_visual_subject"], prompt.positive)
        for palette in (
            [{"colour": "navy", "role": "primary"}],
            [{"colour": "navy", "role": "primary"}, {"colour": "gold", "role": "secondary"},
             {"colour": "cream", "role": "accent"}],
        ):
            data = document()
            data["brief"].update(palette=palette, palette_mode="dominant",
                                 secondary_colour="gold" if len(palette) > 1 else "",
                                 background_inspiration="unfamiliar mineral noun",
                                 creative_description="strangely iridescent " * 80)
            data["brief"]["visual_semantics"]["atmosphere_descriptors"] = ["strangely iridescent"]
            for engine in ("sdxl-1-0", "sd-3-5-medium", "flux-2"):
                spec = build_candidate_spec(data, engine, "A")
                self.assertEqual(spec.semantic_contract["palette"], palette)
                self.assertIn("strangely iridescent", compile_candidate_prompt(spec, 1).positive)

    def test_ci_gate_and_failure_categories(self):
        self.assertEqual(failure_category("MOTIF_DRIFT: missing lattice"), "MOTIF_DRIFT")
        self.assertEqual(failure_category("AUDIENCE_DRIFT: wrong fit"), "AUDIENCE_DRIFT")
        sys.path.insert(0, str(ROOT / "image-service/scripts"))
        from audit_genericity import classify_source
        branch = classify_source("image-service/app/services/example.py",
                                 "if event_type == 'wedding':\n    treatment = 'flowers'\n")
        self.assertEqual(branch[0]["classification"], "CASE_SPECIFIC_BEHAVIOUR")
        example = classify_source("prompt-service/app/services/grill_me_semantic.py",
                                  "instruction = 'For example, a wedding may involve flowers.'\n")
        self.assertEqual(example[0]["classification"], "EXAMPLE_ONLY")
        result = subprocess.run([sys.executable, "scripts/audit_genericity.py"], cwd=ROOT / "image-service",
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout[-500:])

    def test_profile_selection_uses_failure_history_only(self):
        history = FailureHistorySummary(counts={"RASTER": 2}, consecutive_category="RASTER",
                                        consecutive_count=2)
        profile = select_sampling_profile("sd-3-5-medium", 3, history)
        self.assertEqual(profile.name, "raster_rescue_trial")
        self.assertNotIn("event_type", select_sampling_profile.__code__.co_varnames)

    def test_benchmark_calibration_blind_export_and_generalisation(self):
        sys.path.insert(0, str(ROOT / "image-service/scripts"))
        from calibrate_renderer_quality import calibrate
        from export_blind_review import blind_rows
        from report_generalisation import report
        from report_renderer_benchmark import summarize
        records = [{"reference_number": "A", "engine_id": "sdxl-1-0", "direction_id": "A",
                    "sampling_profile": {"name": "baseline"}, "status": "complete",
                    "brief_fidelity": {"subject_score": .9, "atmosphere_score": .8,
                                       "motif_score": .7, "audience_fit_score": .8},
                    "quality_evaluation": {"hard_gate": {"passed": True},
                                           "quality": {"overall_quality": .8}}}]
        self.assertEqual(calibrate(records, []), "NO HUMAN CALIBRATION DATA")
        self.assertEqual(summarize(records)[0]["attempts"], 1)
        self.assertNotIn("quality_evaluation", blind_rows(records)[0])
        health = report()
        self.assertEqual(health["groups"]["holdout"]["briefs"], 50)
        self.assertEqual(health["case_specific_production_branch_count"], 0)


if __name__ == "__main__":
    unittest.main()
