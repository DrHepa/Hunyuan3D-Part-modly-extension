from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from runtime.semantic_report import (
    FALLBACK_POLICY,
    SEMANTIC_LEVEL,
    SEMANTIC_REPORT_SCHEMA,
    SUPPORTED_CANDIDATE_ROLES,
    build_semantic_report,
    build_xpart_semantic_fallback,
)


class SemanticReportTests(unittest.TestCase):
    def _fabricated_artifacts(self) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        segmentation = {
            "source": "p3-sam",
            "raw_part_count": 7,
            "effective_part_count": 5,
            "max_parts": 5,
            "selection_policy": "face_count_desc",
            "face_ids": list(range(10_000)),
            "effective_parts": [
                {"part_id": 0, "raw_part_id": 10, "face_count": 700},
                {"part_id": 1, "raw_part_id": 11, "face_count": 500},
                {"part_id": 2, "raw_part_id": 12, "face_count": 220},
                {"part_id": 3, "raw_part_id": 13, "face_count": 140},
                {"part_id": 4, "raw_part_id": 14, "face_count": 40},
            ],
        }
        bboxes = {
            "source": "p3-sam",
            "parts": [
                {"part_id": "part-0", "raw_part_id": 10, "min": [-0.35, -0.18, 0.55], "max": [0.35, 0.18, 1.35]},
                {"part_id": "part-1", "raw_part_id": 11, "min": [-0.28, -0.16, 0.00], "max": [0.28, 0.16, 0.55]},
                {"part_id": "part-2", "raw_part_id": 12, "min": [-0.16, -0.12, 1.42], "max": [0.16, 0.12, 1.85]},
                {"part_id": "part-3", "raw_part_id": 13, "min": [0.72, -0.08, 0.45], "max": [0.90, 0.08, 1.26]},
                {"part_id": "part-4", "raw_part_id": 14, "min": [0.98, -0.06, 1.68], "max": [1.08, 0.06, 1.86]},
            ],
        }
        metadata = {
            "adapter": "p3-sam-upstream-subprocess",
            "output_dir": "/tmp/p3",
            "effective_aabb_path": "/tmp/p3/auto_mask_mesh_final_aabb_effective.npy",
            "raw_outputs": {"face_ids_npy": "/tmp/p3/face_ids.npy"},
        }
        return segmentation, bboxes, metadata

    def test_build_semantic_report_schema_roles_and_no_huge_arrays(self) -> None:
        segmentation, bboxes, metadata = self._fabricated_artifacts()

        report = build_semantic_report(
            stage="p3-sam",
            params={"max_parts": 5},
            segmentation=segmentation,
            bboxes=bboxes,
            metadata=metadata,
        )

        self.assertEqual(report["schema"], SEMANTIC_REPORT_SCHEMA)
        self.assertEqual(report["mode"], "analysis")
        self.assertEqual(report["source"], "semantic_decomposition_resolver")
        self.assertEqual(report["stage"], "p3-sam")
        self.assertFalse(report["semantic"])
        self.assertEqual(report["semantic_level"], SEMANTIC_LEVEL)
        self.assertFalse(report["publishable"])
        self.assertEqual(report["fallback_policy"], FALLBACK_POLICY)
        self.assertEqual(report["raw_part_count"], 7)
        self.assertEqual(report["effective_part_count"], 5)
        self.assertEqual(report["max_parts"], 5)
        self.assertEqual(report["selection_policy"], "face_count_desc")
        self.assertEqual(tuple(report["candidate_groups"].keys()), SUPPORTED_CANDIDATE_ROLES)
        self.assertEqual(len(report["parts"]), 5)
        self.assertLessEqual(report["confidence"]["aggregate"], 0.58)
        self.assertFalse(report["confidence"]["publishable"])

        grouped_roles = {part["primary_role"] for part in report["parts"]}
        self.assertIn("head_region", grouped_roles)
        self.assertIn("upper_body_region", grouped_roles)
        self.assertIn("lower_body_region", grouped_roles)
        self.assertIn("limb_region", grouped_roles)
        self.assertIn("accessory_candidate", grouped_roles)

        for part in report["parts"]:
            self.assertIn("bbox", part)
            self.assertIn("center", part["bbox"])
            self.assertIn("extent", part["bbox"])
            self.assertIn("volume", part["bbox"])
            self.assertIn("normalized", part)
            self.assertIn("vertical", part["normalized"])
            self.assertIn("lateral", part["normalized"])
            self.assertIn("face_count", part)
            self.assertNotIn("face_ids", part)
            self.assertNotIn("raw_masks", part)

        encoded = json.dumps(report)
        self.assertNotIn("9999", encoded)
        self.assertIn("raw_arrays_omitted", report["diagnostics"])

    def test_build_semantic_report_is_deterministic_and_allows_unknown(self) -> None:
        segmentation = {
            "source": "p3-sam",
            "raw_part_count": 1,
            "effective_part_count": 1,
            "max_parts": 1,
            "selection_policy": "face_count_desc",
            "effective_parts": [{"part_id": 0, "raw_part_id": 2, "face_count": 1}],
        }
        bboxes = {"source": "p3-sam", "parts": [{"part_id": "part-0", "min": [0, 0, 0], "max": [1, 1, 1]}]}

        first = build_semantic_report(stage="p3-sam", segmentation=segmentation, bboxes=bboxes, metadata={})
        second = build_semantic_report(stage="p3-sam", segmentation=segmentation, bboxes=bboxes, metadata={})

        self.assertEqual(first, second)
        self.assertEqual(first["parts"][0]["primary_role"], "unknown")
        self.assertEqual(first["candidate_groups"]["unknown"], ["part-0"])
        self.assertFalse(first["semantic"])
        self.assertFalse(first["publishable"])

    def test_xpart_fallback_builds_limited_aabb_only_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            aabb_path = Path(temp_dir) / "external_aabb.npy"
            np.save(
                aabb_path,
                np.array(
                    [
                        [[-0.1, -0.1, 0.7], [0.1, 0.1, 1.0]],
                        [[-0.4, -0.2, 0.0], [0.4, 0.2, 0.6]],
                    ],
                    dtype=float,
                ),
            )

            report = build_xpart_semantic_fallback({"max_parts": 1}, aabb_path)

        self.assertEqual(report["stage"], "x-part")
        self.assertFalse(report["semantic"])
        self.assertFalse(report["publishable"])
        self.assertEqual(report["raw_part_count"], 2)
        self.assertEqual(report["effective_part_count"], 1)
        self.assertEqual(report["selection_policy"], "aabb_order_limited")
        self.assertEqual(len(report["parts"]), 1)
        self.assertIn("aabb_only_semantic_report_limited", report["warnings"])
        self.assertEqual(report["diagnostics"]["fallback_reason"], "xpart_aabb_only")

    def test_xpart_fallback_missing_aabb_is_unavailable_without_mutation(self) -> None:
        report = build_xpart_semantic_fallback({"max_parts": 3}, None)

        self.assertEqual(report["stage"], "x-part")
        self.assertFalse(report["semantic"])
        self.assertFalse(report["publishable"])
        self.assertEqual(report["parts"], [])
        self.assertIn("semantic_report_unavailable", report["warnings"])
        self.assertFalse(report["diagnostics"]["mutates_xpart_inputs"])


if __name__ == "__main__":
    unittest.main()
