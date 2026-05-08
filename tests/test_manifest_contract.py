from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ManifestContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))

    def test_planned_identity_and_metadata_are_present(self) -> None:
        self.assertEqual(self.manifest["id"], "hunyuan3d-part")
        self.assertEqual(self.manifest["author"], "DrHepa")
        self.assertEqual(
            self.manifest["source"],
            "https://github.com/DrHepa/Hunyuan3D-Part-modly-extension",
        )
        self.assertEqual(self.manifest["type"], "model")
        self.assertEqual(self.manifest["bucket"], "model-managed-setup")
        self.assertEqual(self.manifest["vram_gb"], 24)
        self.assertEqual(self.manifest["generator_class"], "Hunyuan3DPartGenerator")
        self.assertEqual(self.manifest["live_identity"]["status"], "unconfirmed")
        self.assertEqual(self.manifest["planned_identity"]["type"], "model")
        self.assertEqual(self.manifest["planned_identity"]["bucket"], "model-managed-setup")
        metadata = self.manifest["metadata"]
        self.assertEqual(metadata["resolution"], "host-compat")
        self.assertEqual(metadata["implementation_profile"], "model-managed-generator-adapter")
        self.assertEqual(metadata["setup_contract"], "python-root-setup-py")
        self.assertEqual(metadata["support_state"], "experimental")
        self.assertEqual(metadata["surface_owner"], "electron")
        self.assertFalse(metadata["headless_eligible"])
        self.assertEqual(metadata["linux_arm64_risk"], "elevated")
        self.assertFalse(metadata["plan_only"])
        self.assertTrue(metadata["runtime_shell_ready"])
        self.assertFalse(metadata["runtime_adapter_pending"])
        self.assertEqual(metadata["hf_repo"], "tencent/Hunyuan3D-Part")
        self.assertEqual(metadata["download_check"], "p3sam/p3sam.safetensors")
        self.assertEqual(metadata["weight_owner_id"], "p3sam")
        self.assertIn("Primary workflow input", metadata["workflow_constraint"])

    def test_node_schema_uses_image_primary_and_required_mesh_secondary(self) -> None:
        node = self.manifest["nodes"][0]
        self.assertEqual(node["id"], "decompose-mesh")
        self.assertEqual(node["name"], "Decompose Mesh")
        self.assertEqual(node["input"], "image")
        self.assertEqual(node["output"], "mesh")
        self.assertEqual(node["hf_repo"], "tencent/Hunyuan3D-Part")
        self.assertEqual(node["download_check"], "p3sam/p3sam.safetensors")
        self.assertEqual(node["weight_owner_id"], "p3sam")
        self.assertEqual(len(node["inputs"]), 2)
        self.assertEqual(node["inputs"][0]["name"], "front")
        self.assertEqual(node["inputs"][0]["type"], "image")
        self.assertFalse(node["inputs"][0]["required"])
        self.assertEqual(node["inputs"][1]["name"], "mesh")
        self.assertEqual(node["inputs"][1]["type"], "mesh")
        self.assertTrue(node["inputs"][1]["required"])
        self.assertEqual(node["inputs"][1]["formats"], ["glb", "obj", "stl", "ply"])
        self.assertEqual(len(node["outputs"]), 1)
        self.assertTrue(node["outputs"][0]["primary"])
        self.assertEqual(node["outputs"][0]["formats"], ["glb"])
        self.assertEqual(node["defaults"], {"pipeline_stage": "p3-sam", "export_format": "glb", "output_mode": "primary"})
        self.assertEqual(
            sorted(item["id"] for item in node["params_schema"]),
            ["export_format", "max_parts", "output_mode", "pipeline_stage", "quality_preset", "seed"],
        )
        export_param = next(item for item in node["params_schema"] if item["id"] == "export_format")
        self.assertEqual(export_param["options"], [{"value": "glb", "label": "GLB"}])
        self.assertEqual(node["params"]["export_format"]["enum"], ["glb"])

    def test_output_mode_manifest_contract_exposes_visibility_modes(self) -> None:
        node = self.manifest["nodes"][0]
        output_mode = next(item for item in node["params_schema"] if item["id"] == "output_mode")

        self.assertEqual(output_mode["default"], "primary")
        self.assertEqual(
            output_mode["options"],
            [
                {"value": "primary", "label": "Primary mesh only"},
                {"value": "analysis", "label": "Primary mesh + analysis metadata"},
                {"value": "debug", "label": "Primary mesh + debug part GLBs"},
            ],
        )
        self.assertIn("primary exposes one UI/downstream mesh", output_mode["tooltip"])
        self.assertIn("analysis keeps one mesh plus segmentation/bbox/completion metadata", output_mode["tooltip"])
        self.assertIn("debug additionally exposes per-part GLB sidecars", output_mode["tooltip"])
        self.assertIn("not semantic", output_mode["tooltip"])
        self.assertEqual(node["params"]["output_mode"]["enum"], ["primary", "analysis", "debug"])
        self.assertEqual(node["params"]["output_mode"]["default"], "primary")

    def test_x_part_resource_params_are_documented_as_real_gpu_knobs(self) -> None:
        node = self.manifest["nodes"][0]
        max_parts = next(item for item in node["params_schema"] if item["id"] == "max_parts")
        quality_preset = next(item for item in node["params_schema"] if item["id"] == "quality_preset")

        self.assertEqual(max_parts["default"], 32)
        self.assertEqual(max_parts["max"], 512)
        self.assertIn("Hard runtime cap", max_parts["tooltip"])
        self.assertIn("P3-SAM is post-filtered by face count", max_parts["tooltip"])
        self.assertIn("not semantic labels", max_parts["tooltip"])
        self.assertIn("GPU execution knobs", quality_preset["tooltip"])
        self.assertIn("bf16/fp16 CUDA placement", quality_preset["tooltip"])

    def test_pipeline_stage_manifest_contract_exposes_real_runtime_stages(self) -> None:
        node = self.manifest["nodes"][0]
        pipeline_stage = next(item for item in node["params_schema"] if item["id"] == "pipeline_stage")
        self.assertEqual(pipeline_stage["default"], "p3-sam")
        self.assertEqual(
            pipeline_stage["options"],
            [
                {"value": "p3-sam", "label": "P3-SAM Segmentation"},
                {"value": "x-part", "label": "X-Part Generation"},
                {"value": "full", "label": "Full Pipeline"},
            ],
        )
        self.assertIn("real runtime adapters", pipeline_stage["tooltip"])
        self.assertIn("Full chains P3-SAM into X-Part", pipeline_stage["tooltip"])
        self.assertEqual(node["params"]["pipeline_stage"]["enum"], ["p3-sam", "x-part", "full"])
        self.assertEqual(node["params"]["pipeline_stage"]["default"], "p3-sam")

    def test_non_goals_and_runtime_boundary_are_explicit(self) -> None:
        node = self.manifest["nodes"][0]
        self.assertEqual(
            node["non_goals"],
            [
                "multi-mesh primary routing",
                "headless execution",
                "install/build/release actions from local runtime",
            ],
        )
        self.assertEqual(self.manifest["execution"]["surface_owner"], "electron")
        self.assertEqual(self.manifest["execution"]["backend_readiness"], "fastapi")
        self.assertNotIn("entry", self.manifest)


if __name__ == "__main__":
    unittest.main()
