from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generator import Hunyuan3DPartGenerator
from runtime.config import (
    DEFAULT_MAX_PARTS,
    MAX_PARTS_HARD_CAP,
    HostFacts,
    SUPPORTED_EXPORT_FORMATS,
    SUPPORTED_SEMANTIC_RESOLVERS,
    SUPPORTED_OUTPUT_MODES,
    infer_model_root,
    normalize_params,
    resolve_host_facts,
    resolve_runtime_context,
    resolve_x_part_resource_limits,
)
from runtime.errors import RuntimeFailure, SetupFailure, ValidationError
from runtime.export import DecompositionArtifacts, PartArtifact
from runtime.pipeline import _require_p3_sam_aabb, run_pipeline_stage
from runtime.p3_sam import (
    AdapterReadiness,
    SONATA_CACHE_ENV_VAR,
    AdapterPaths,
    P3_SAM_FAILURE_JSON_NAME,
    P3_SAM_STDERR_LOG_NAME,
    P3_SAM_STDOUT_LOG_NAME,
    build_adapter_readiness,
    build_execution_plan,
    build_subprocess_command,
    collect_artifacts as collect_p3_sam_artifacts,
    decompose_mesh,
    resolve_weight_path,
    run_upstream_p3_sam,
    runtime_source_root,
    sonata_cache_root,
)
from runtime.x_part import (
    _build_subprocess_script,
    _resolve_x_part_min_available_memory_gib,
    _run_with_memory_guard,
    resolve_bundle_root,
    run_upstream_x_part,
    verify_x_part_import_smoke,
    X_PART_MEMORY_GUARD_ENV_VAR,
    X_PART_FAILURE_JSON_NAME,
    X_PART_STDERR_LOG_NAME,
    X_PART_STDOUT_LOG_NAME,
    x_part_import_root,
)
from runtime.validate import validate_inputs, validate_mesh_path


class RuntimeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.injected_model_dir = self.workspace / "managed-models" / "hunyuan3d-part" / "p3sam"
        self.injected_model_dir.mkdir(parents=True)
        self.workflows_dir = self.workspace / "Workflows"
        self.workflows_dir.mkdir()
        self.mesh = self.workspace / "sample.glb"
        self.mesh.write_bytes(b"glTF")
        self.workflow_mesh = self.workflows_dir / "sample.glb"
        self.workflow_mesh.write_bytes(b"glTF-workflow")
        self.part_a = self.workspace / "part-a.glb"
        self.part_a.write_bytes(b"glTF-part-a")
        self.part_b = self.workspace / "part-b.glb"
        self.part_b.write_bytes(b"glTF-part-b")
        ready_host = HostFacts(
            os_name="linux",
            arch="x86_64",
            python_version="3.11.8",
            python_abi="cp311",
            cuda_visible=True,
        )
        self.ready_context = resolve_runtime_context(
            project_root=ROOT,
            host_facts=ready_host,
            dependency_checker=lambda _: object(),
        )
        self.ready_shell_context = resolve_runtime_context(
            project_root=self.workspace,
            host_facts=ready_host,
            dependency_checker=lambda _: object(),
        )
        self._original_workspace_dir = os.environ.get("WORKSPACE_DIR")

    def tearDown(self) -> None:
        if self._original_workspace_dir is None:
            os.environ.pop("WORKSPACE_DIR", None)
        else:
            os.environ["WORKSPACE_DIR"] = self._original_workspace_dir
        self.temp_dir.cleanup()

    def _create_valid_x_part_bundle(self, bundle_root: Path) -> Path:
        (bundle_root / "model").mkdir(parents=True, exist_ok=True)
        (bundle_root / "model" / "model.safetensors").write_bytes(b"weights")
        for dirname in ("conditioner", "shapevae", "scheduler"):
            (bundle_root / dirname).mkdir(parents=True, exist_ok=True)
        (bundle_root / "config.json").write_text("{}\n", encoding="utf-8")
        return bundle_root

    def test_validate_inputs_rejects_unsupported_mesh(self) -> None:
        bad_mesh = self.workspace / "mesh.fbx"
        bad_mesh.write_text("fbx", encoding="utf-8")
        with self.assertRaises(ValidationError):
            validate_inputs({"mesh": bad_mesh})

    def test_validate_mesh_path_resolves_workspace_relative_path_from_workspace_root(self) -> None:
        validated = validate_mesh_path(
            "Workflows/sample.glb",
            search_roots=[self.workspace],
        )
        self.assertEqual(validated.mesh_path, self.workflow_mesh.resolve())

    def test_validate_mesh_path_resolves_collection_relative_path_from_outputs_dir(self) -> None:
        validated = validate_mesh_path(
            "sample.glb",
            search_roots=[self.workflows_dir],
        )
        self.assertEqual(validated.mesh_path, self.workflow_mesh.resolve())

    def test_validate_mesh_path_rejects_parent_traversal(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            validate_mesh_path("../secret.glb", search_roots=[self.workflows_dir, self.workspace])
        self.assertEqual(ctx.exception.code, "invalid_mesh_path")

    def test_host_gate_warns_but_allows_linux_arm64_when_other_requirements_are_ready(self) -> None:
        arm64_context = resolve_runtime_context(
            project_root=ROOT,
            host_facts=HostFacts(
                os_name="linux",
                arch="arm64",
                python_version="3.11.8",
                python_abi="cp311",
                cuda_visible=True,
            ),
            dependency_checker=lambda _: object(),
        )
        plan = build_execution_plan(
            {"mesh": self.mesh},
            runtime_context=arm64_context,
        )
        self.assertEqual(plan.context.support.status, "ready_with_warnings")
        self.assertIn("linux_arm64_risk=elevated", plan.context.support.warnings)

    def test_resolve_host_facts_uses_torch_cuda_probe_without_cuda_env(self) -> None:
        with mock.patch("runtime.config.probe_torch_cuda_availability", return_value=True):
            facts = resolve_host_facts(env={})

        self.assertTrue(facts.cuda_visible)

    def test_host_support_stays_blocked_without_cuda_env_or_torch_cuda(self) -> None:
        with mock.patch("runtime.config.probe_torch_cuda_availability", return_value=False):
            context = resolve_runtime_context(
                project_root=ROOT,
                env={},
                dependency_checker=lambda _: object(),
            )

        self.assertFalse(context.support.ready)
        self.assertEqual(context.support.failure["code"], "cuda_not_visible")

    def test_pipeline_stage_defaults_to_p3_sam(self) -> None:
        plan = build_execution_plan(
            {"mesh": self.mesh},
            runtime_context=self.ready_context,
        )
        self.assertEqual(plan.params.pipeline_stage, "p3-sam")
        self.assertEqual(plan.params.export_format, "glb")
        self.assertEqual(plan.params.max_parts, DEFAULT_MAX_PARTS)
        self.assertEqual(plan.params.quality_preset, "balanced")
        self.assertEqual(plan.params.semantic_resolver, "off")

    def test_supported_export_formats_are_glb_only(self) -> None:
        self.assertEqual(SUPPORTED_EXPORT_FORMATS, ("glb",))

    def test_output_mode_defaults_and_validation(self) -> None:
        self.assertEqual(SUPPORTED_OUTPUT_MODES, ("primary", "analysis", "debug"))

        params = normalize_params({"output_mode": "ANALYSIS"})

        self.assertEqual(params.output_mode, "analysis")
        self.assertEqual(params.to_dict()["output_mode"], "analysis")

        with self.assertRaises(ValidationError) as ctx:
            normalize_params({"output_mode": "semantic"})

        self.assertEqual(ctx.exception.code, "invalid_output_mode")

    def test_semantic_resolver_defaults_normalizes_analysis_and_rejects_guided(self) -> None:
        self.assertEqual(SUPPORTED_SEMANTIC_RESOLVERS, ("off", "analysis"))

        defaults = normalize_params()
        self.assertEqual(defaults.semantic_resolver, "off")
        self.assertEqual(defaults.to_dict()["semantic_resolver"], "off")

        analysis = normalize_params({"semantic_resolver": " ANALYSIS "})
        self.assertEqual(analysis.semantic_resolver, "analysis")
        self.assertEqual(analysis.to_dict()["semantic_resolver"], "analysis")

        for observed in ("guided", "foo"):
            with self.subTest(observed=observed):
                with self.assertRaises(ValidationError) as ctx:
                    normalize_params({"semantic_resolver": observed})

                self.assertEqual(ctx.exception.code, "invalid_semantic_resolver")
                self.assertEqual(ctx.exception.details["allowed"], ["off", "analysis"])
                self.assertEqual(ctx.exception.details["observed"], observed)
                self.assertNotIn("guided", ctx.exception.details["allowed"])

    def test_real_stages_build_execution_plans_without_deferred_rejection(self) -> None:
        for stage in ("x-part", "full"):
            with self.subTest(stage=stage):
                plan = build_execution_plan(
                    {"mesh": self.mesh},
                    {"pipeline_stage": stage},
                    runtime_context=self.ready_context,
                )
                self.assertEqual(plan.params.pipeline_stage, stage)

    def test_x_part_resource_limits_are_explicit_and_keep_quality_profile_high_detail(self) -> None:
        params = normalize_params({"pipeline_stage": "x-part", "quality_preset": "quality", "max_parts": 128})

        limits = resolve_x_part_resource_limits(params)

        self.assertEqual(limits["quality_preset"], "quality")
        self.assertEqual(limits["effective_max_parts"], 128)
        self.assertEqual(limits["num_inference_steps"], 50)
        self.assertEqual(limits["octree_resolution"], 512)
        self.assertEqual(limits["torch_dtype"], "float32")
        self.assertLess(limits["num_chunks"], 400000)
        self.assertEqual(limits["surface_point_count"], 8192)
        self.assertEqual(limits["requested_bbox_point_count"], 8192)
        self.assertEqual(limits["bbox_point_count"], 4096)
        self.assertEqual(limits["point_budget_policy"], "cap_total_bbox_points_by_effective_max_parts")

    def test_x_part_balanced_max_parts_eight_uses_capped_point_budget(self) -> None:
        params = normalize_params({"pipeline_stage": "x-part", "quality_preset": "balanced", "max_parts": 8})

        limits = resolve_x_part_resource_limits(params)

        self.assertEqual(limits["quality_preset"], "balanced")
        self.assertEqual(limits["effective_max_parts"], 8)
        self.assertEqual(limits["surface_point_count"], 8192)
        self.assertEqual(limits["bbox_point_count"], 8192)
        self.assertEqual(limits["total_bbox_point_budget"], 65536)

    def test_x_part_resource_limits_fail_closed_above_hard_part_cap(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            normalize_params({"pipeline_stage": "x-part", "max_parts": MAX_PARTS_HARD_CAP + 1})

        self.assertEqual(ctx.exception.code, "resource_guardrail_exceeded")
        self.assertEqual(ctx.exception.details["maximum"], MAX_PARTS_HARD_CAP)

    def test_runtime_requires_adapter_and_does_not_fake_inference(self) -> None:
        with self.assertRaises(SetupFailure):
            decompose_mesh(
                {"mesh": self.mesh},
                output_dir=self.workspace / "out",
                runtime_context=self.ready_context,
            )

    def test_dispatch_accepts_x_part_and_full_with_mocked_runtime_adapter(self) -> None:
        def fake_adapter(plan):
            return DecompositionArtifacts(
                primary_mesh_path=self.mesh,
                parts=(),
                segmentation={"source": plan.params.pipeline_stage},
                bboxes={"parts": []},
                completion={"status": "completed"},
                metadata={"adapter": "unit-test"},
            )

        for stage in ("x-part", "full"):
            with self.subTest(stage=stage):
                result = decompose_mesh(
                    {"mesh": self.mesh},
                    {"pipeline_stage": stage},
                    output_dir=self.workspace / f"out-{stage}",
                    runtime_adapter=fake_adapter,
                    runtime_context=self.ready_context,
                )
                self.assertTrue(Path(result["primary_mesh"]).exists())

    def test_generator_reports_separate_shell_and_weight_readiness(self) -> None:
        generator = Hunyuan3DPartGenerator(
            self.injected_model_dir,
            self.workspace,
            project_root=self.workspace,
            runtime_context=self.ready_shell_context,
        )
        readiness = generator.readiness_status()
        self.assertTrue(readiness["shell_ready"])
        self.assertFalse(readiness["execution_ready"])
        self.assertEqual(readiness["setup"]["status"], "prepared_shell")
        self.assertFalse(readiness["setup"]["ready"])
        self.assertEqual(readiness["setup"]["model_dir"], str(self.injected_model_dir))
        self.assertEqual(readiness["setup"]["venv_python"], str(self.workspace / "venv" / "bin" / "python"))
        self.assertEqual(readiness["runtime_adapter"]["status"], "blocked")
        self.assertEqual(readiness["weights"]["status"], "missing")
        self.assertEqual(readiness["weights"]["model_dir"], str(self.injected_model_dir))
        self.assertEqual(
            readiness["weights"]["path"],
            str(self.injected_model_dir / "p3sam.safetensors"),
        )

    def test_generator_setup_is_ready_when_extension_venv_exists(self) -> None:
        managed_python = self.workspace / "venv" / "bin" / "python"
        managed_python.parent.mkdir(parents=True)
        managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        generator = Hunyuan3DPartGenerator(
            self.injected_model_dir,
            self.workspace,
            project_root=self.workspace,
            runtime_context=self.ready_shell_context,
        )

        readiness = generator.readiness_status()
        self.assertTrue(readiness["setup"]["ready"])

    def test_generator_readiness_exposes_modly_contract_fields_when_ready(self) -> None:
        managed_python = self.workspace / "venv" / "bin" / "python"
        managed_python.parent.mkdir(parents=True, exist_ok=True)
        managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        weight_path = self.injected_model_dir / "p3sam.safetensors"
        weight_path.write_bytes(b"weights")
        adapter_readiness = AdapterReadiness(
            ready=True,
            status="ready",
            paths=AdapterPaths(
                managed_python=managed_python,
                runtime_root=self.workspace / ".upstream" / "hunyuan3d-part",
                entrypoint=self.workspace / ".upstream" / "hunyuan3d-part" / "P3-SAM" / "demo" / "auto_mask.py",
                weights=weight_path,
            ),
            components={
                "managed_python": {"ready": True, "status": "ready"},
                "runtime_source": {"ready": True, "status": "ready"},
                "entrypoint": {"ready": True, "status": "ready"},
                "import_smoke": {"ready": True, "status": "ready"},
                "weights": {"ready": True, "status": "ready", "path": str(weight_path)},
            },
            message="Runtime adapter ready.",
        )

        generator = Hunyuan3DPartGenerator(
            self.injected_model_dir,
            self.workspace,
            project_root=self.workspace,
            runtime_context=self.ready_shell_context,
        )

        with mock.patch("generator.build_adapter_readiness", return_value=adapter_readiness):
            readiness = generator.readiness_status()

        self.assertTrue(readiness["ok"])
        self.assertEqual(readiness["machine_code"], "ready")
        self.assertEqual(readiness["label_hint"], "Ready")
        self.assertIsNone(readiness["reason"])
        self.assertEqual(readiness["details"]["blocker_count"], 0)
        self.assertIn("host_support", readiness)
        self.assertIn("runtime_adapter", readiness)
        self.assertIn("weights", readiness)

    def test_generator_readiness_refreshes_stale_cuda_gate_from_runner_torch_probe(self) -> None:
        stale_context = resolve_runtime_context(
            project_root=self.workspace,
            host_facts=HostFacts(
                os_name="linux",
                arch="x86_64",
                python_version="3.11.8",
                python_abi="cp311",
                cuda_visible=False,
            ),
            dependency_checker=lambda _: object(),
        )
        managed_python = self.workspace / "venv" / "bin" / "python"
        managed_python.parent.mkdir(parents=True, exist_ok=True)
        managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        weight_path = self.injected_model_dir / "p3sam.safetensors"
        weight_path.write_bytes(b"weights")
        adapter_readiness = AdapterReadiness(
            ready=True,
            status="ready",
            paths=None,
            components={
                "managed_python": {"ready": True, "status": "ready"},
                "runtime_source": {"ready": True, "status": "ready"},
                "entrypoint": {"ready": True, "status": "ready"},
                "import_smoke": {"ready": True, "status": "ready"},
                "weights": {"ready": True, "status": "ready", "path": str(weight_path)},
            },
            message="Runtime adapter ready.",
        )
        generator = Hunyuan3DPartGenerator(
            self.injected_model_dir,
            self.workspace,
            project_root=self.workspace,
            runtime_context=stale_context,
        )

        with (
            mock.patch("generator.probe_torch_cuda_availability", return_value=True),
            mock.patch("generator.build_adapter_readiness", return_value=adapter_readiness),
        ):
            readiness = generator.readiness_status()

        self.assertTrue(readiness["host_support"]["ready"])
        self.assertEqual(readiness["host_support"]["source"], "current_process_torch")
        self.assertTrue(readiness["ok"])

    def test_generator_uses_injected_model_dir_for_weight_path_and_download_status(self) -> None:
        generator = Hunyuan3DPartGenerator(
            self.injected_model_dir,
            self.workspace,
            project_root=self.workspace,
            runtime_context=self.ready_shell_context,
        )
        weight_path = self.injected_model_dir / generator.download_check
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        weight_path.write_bytes(b"weights")

        self.assertEqual(generator._weight_path(), weight_path)
        self.assertTrue(generator.is_downloaded())
        self.assertEqual(generator.readiness_status()["weights"]["status"], "downloaded")

    def test_weight_resolution_supports_nested_managed_model_tree(self) -> None:
        managed_model_root = self.workspace / "managed-models" / "hunyuan3d-part"
        weight_path = managed_model_root / "p3sam" / "p3sam.safetensors"
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        weight_path.write_bytes(b"weights")

        self.assertEqual(resolve_weight_path(managed_model_root), weight_path)

    def test_infer_model_root_uses_modly_central_cache_for_extensions_layout(self) -> None:
        extension_root = self.workspace / "Modly" / "extensions" / "hunyuan3d-part"
        extension_root.mkdir(parents=True)

        self.assertEqual(
            infer_model_root(extension_root),
            self.workspace / "Modly" / "models" / "hunyuan3d-part",
        )

    def test_infer_model_root_falls_back_to_local_runtime_models_outside_modly_layout(self) -> None:
        self.assertEqual(
            infer_model_root(self.workspace),
            self.workspace / ".runtime" / "models",
        )

    def test_weight_resolution_supports_nested_central_modly_model_tree(self) -> None:
        model_root = self.workspace / "Modly" / "models" / "hunyuan3d-part"
        weight_path = model_root / "p3sam" / "p3sam" / "p3sam.safetensors"
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        weight_path.write_bytes(b"weights")

        self.assertEqual(resolve_weight_path(model_root), weight_path)

    def test_generator_modly_layout_uses_central_model_cache_for_readiness(self) -> None:
        project_root = self.workspace / "Modly" / "extensions" / "hunyuan3d-part"
        project_root.mkdir(parents=True)
        managed_python = project_root / "venv" / "bin" / "python"
        managed_python.parent.mkdir(parents=True, exist_ok=True)
        managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        weight_path = self.workspace / "Modly" / "models" / "hunyuan3d-part" / "p3sam" / "p3sam" / "p3sam.safetensors"
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        weight_path.write_bytes(b"weights")
        runtime_context = resolve_runtime_context(
            project_root=project_root,
            host_facts=HostFacts(
                os_name="linux",
                arch="x86_64",
                python_version="3.11.8",
                python_abi="cp311",
                cuda_visible=True,
            ),
            dependency_checker=lambda _: object(),
        )

        generator = Hunyuan3DPartGenerator(
            project_root=project_root,
            runtime_context=runtime_context,
        )

        readiness = generator.readiness_status()
        self.assertEqual(runtime_context.paths.model_root, self.workspace / "Modly" / "models" / "hunyuan3d-part")
        self.assertTrue(generator.is_downloaded())
        self.assertEqual(readiness["weights"]["status"], "downloaded")
        self.assertEqual(readiness["weights"]["path"], str(weight_path))
        self.assertNotIn(str(project_root / ".runtime" / "models"), readiness["weights"]["path"])
        self.assertEqual(
            readiness["runtime_adapter"]["components"]["weights"]["path"],
            str(weight_path),
        )

    def test_injected_model_dir_remains_authoritative_over_inferred_model_root(self) -> None:
        project_root = self.workspace / "Modly" / "extensions" / "hunyuan3d-part"
        project_root.mkdir(parents=True)
        central_weight = self.workspace / "Modly" / "models" / "hunyuan3d-part" / "p3sam" / "p3sam" / "p3sam.safetensors"
        central_weight.parent.mkdir(parents=True, exist_ok=True)
        central_weight.write_bytes(b"central-weights")
        injected_model_dir = self.workspace / "override-model-dir"
        injected_weight = injected_model_dir / "p3sam" / "p3sam.safetensors"
        injected_weight.parent.mkdir(parents=True, exist_ok=True)
        injected_weight.write_bytes(b"override-weights")
        runtime_context = resolve_runtime_context(
            project_root=project_root,
            host_facts=HostFacts(
                os_name="linux",
                arch="x86_64",
                python_version="3.11.8",
                python_abi="cp311",
                cuda_visible=True,
            ),
            dependency_checker=lambda _: object(),
        )

        generator = Hunyuan3DPartGenerator(
            injected_model_dir,
            self.workspace,
            project_root=project_root,
            runtime_context=runtime_context,
        )

        self.assertEqual(generator._weight_path(), injected_weight)

    def test_adapter_readiness_reports_missing_runtime_source(self) -> None:
        managed_python = self.workspace / "venv" / "bin" / "python"
        managed_python.parent.mkdir(parents=True)
        managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        managed_python.chmod(stat.S_IRWXU)
        weight_path = self.injected_model_dir / "p3sam.safetensors"
        weight_path.write_bytes(b"weights")

        with mock.patch("runtime.p3_sam.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(
                returncode=1,
                stdout='__P3SAM_IMPORT_SMOKE__={"error":"No module named \'missing_dep\'","error_type":"ModuleNotFoundError","module":"/tmp/model.py","python_exe":"/tmp/python","ready":false,"traceback_tail":["ModuleNotFoundError: No module named \'missing_dep\'"]}\n',
                stderr="ModuleNotFoundError: No module named 'missing_dep'\n",
            )
            readiness = build_adapter_readiness(
                project_root=self.workspace,
                managed_python=managed_python,
                model_root=self.injected_model_dir,
            )

        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.status, "blocked")
        self.assertFalse(readiness.components["runtime_source"]["ready"])
        self.assertEqual(readiness.components["import_smoke"]["status"], "missing")

    def test_adapter_readiness_reports_import_smoke_failure_when_dependency_is_missing(self) -> None:
        managed_python = self.workspace / "venv" / "bin" / "python"
        managed_python.parent.mkdir(parents=True)
        managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        weight_path = self.injected_model_dir / "p3sam.safetensors"
        weight_path.write_bytes(b"weights")
        runtime_root = runtime_source_root(self.workspace)
        (runtime_root / "P3-SAM" / "demo").mkdir(parents=True, exist_ok=True)
        (runtime_root / "XPart" / "partgen" / "models").mkdir(parents=True, exist_ok=True)
        (runtime_root / "P3-SAM" / "model.py").write_text("import missing_dep\n", encoding="utf-8")
        (runtime_root / "P3-SAM" / "demo" / "auto_mask.py").write_text("print('ok')\n", encoding="utf-8")

        with mock.patch("runtime.p3_sam.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(
                returncode=1,
                stdout='__P3SAM_IMPORT_SMOKE__={"error":"No module named \'missing_dep\'","error_type":"ModuleNotFoundError","module":"/tmp/model.py","python_exe":"/tmp/python","ready":false,"traceback_tail":["ModuleNotFoundError: No module named \'missing_dep\'"]}\n',
                stderr="ModuleNotFoundError: No module named 'missing_dep'\n",
            )
            readiness = build_adapter_readiness(
                project_root=self.workspace,
                managed_python=managed_python,
                model_root=self.injected_model_dir,
            )

        self.assertFalse(readiness.ready)
        smoke = readiness.components["import_smoke"]
        self.assertFalse(smoke["ready"])
        self.assertEqual(smoke["status"], "blocked")
        self.assertIn("missing_dep", smoke["missing_modules"])

    def test_subprocess_command_is_constructed_without_running_inference(self) -> None:
        command = build_subprocess_command(
            managed_python=self.workspace / "venv" / "bin" / "python",
            entrypoint=self.workspace / ".upstream" / "hunyuan3d-part" / "P3-SAM" / "demo" / "auto_mask.py",
            weights=self.injected_model_dir / "p3sam.safetensors",
            mesh_path=self.mesh,
            output_dir=self.workspace / "out",
            params=build_execution_plan({"mesh": self.mesh}, runtime_context=self.ready_context).params,
        )

        self.assertIn("--ckpt_path", command)
        self.assertIn("--mesh_path", command)
        self.assertIn("--output_path", command)
        self.assertIn("--save_mid_res", command)
        self.assertIn("--post_process", command)
        self.assertEqual(command[command.index("--save_mid_res") + 1], "1")

    def test_p3_sam_artifact_collection_requires_save_mid_res_outputs(self) -> None:
        output_dir = self.workspace / "p3sam-out"
        output_dir.mkdir()
        for name in (
            "auto_mask_mesh_final.glb",
            "auto_mask_mesh_final.ply",
            "auto_mask_mesh_final_aabb.glb",
            "auto_mask_mesh_final_aabb.npy",
            "auto_mask_mesh_final_face_ids.npy",
        ):
            path = output_dir / name
            if path.suffix == ".npy":
                import numpy as np

                np.save(path, [[0, 0, 0], [1, 1, 1]] if "aabb" in name else [1, 2, 3])
            else:
                path.write_bytes(b"glTF")

        artifacts = collect_p3_sam_artifacts(output_dir, export_format="glb")
        self.assertEqual(artifacts.primary_mesh_path, output_dir / "auto_mask_mesh_final.glb")
        self.assertEqual(artifacts.metadata["aabb_path"], str(output_dir / "auto_mask_mesh_final_aabb.npy"))

    def test_p3_sam_artifact_collection_filters_top_k_parts_and_maps_raw_aabbs(self) -> None:
        output_dir = self.workspace / "p3sam-filtered"
        output_dir.mkdir()
        for name in (
            "auto_mask_mesh_final.glb",
            "auto_mask_mesh_final.ply",
            "auto_mask_mesh_final_aabb.glb",
        ):
            (output_dir / name).write_bytes(b"glTF")

        import numpy as np

        np.save(
            output_dir / "auto_mask_mesh_final_face_ids.npy",
            [-1, 5, 5, 2, 9, 5, 2, 9, 9, 9],
        )
        np.save(
            output_dir / "auto_mask_mesh_final_aabb.npy",
            [
                [[2, 0, 0], [2, 1, 1]],
                [[5, 0, 0], [5, 1, 1]],
                [[9, 0, 0], [9, 1, 1]],
            ],
        )

        artifacts = collect_p3_sam_artifacts(
            output_dir,
            export_format="glb",
            params=normalize_params({"max_parts": 2}),
        )

        self.assertEqual(artifacts.segmentation["source"], "p3-sam")
        self.assertFalse(artifacts.segmentation["semantic"])
        self.assertEqual(artifacts.segmentation["raw_part_count"], 3)
        self.assertEqual(artifacts.segmentation["effective_part_count"], 2)
        self.assertEqual(artifacts.segmentation["selection_policy"], "face_count_desc")
        self.assertEqual(
            artifacts.segmentation["effective_parts"],
            [
                {"part_id": 0, "raw_part_id": 9, "face_count": 4},
                {"part_id": 1, "raw_part_id": 5, "face_count": 3},
            ],
        )
        self.assertEqual(artifacts.segmentation["face_ids"], [-1, 1, 1, -1, 0, 1, -1, 0, 0, 0])
        self.assertEqual([part["raw_part_id"] for part in artifacts.bboxes["parts"]], [9, 5])
        self.assertEqual(artifacts.bboxes["parts"][0]["min"], [9, 0, 0])
        effective_aabb_path = Path(str(artifacts.metadata["effective_aabb_path"]))
        self.assertTrue(effective_aabb_path.is_file())
        self.assertEqual(np.load(effective_aabb_path).tolist(), [[[9, 0, 0], [9, 1, 1]], [[5, 0, 0], [5, 1, 1]]])

    def test_run_upstream_p3_sam_sets_extension_local_sonata_cache_env(self) -> None:
        managed_python = self.workspace / "venv" / "bin" / "python"
        managed_python.parent.mkdir(parents=True)
        managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        runtime_root = runtime_source_root(self.workspace)
        (runtime_root / "P3-SAM" / "demo").mkdir(parents=True, exist_ok=True)
        (runtime_root / "XPart" / "partgen" / "models").mkdir(parents=True, exist_ok=True)
        (runtime_root / "P3-SAM" / "model.py").write_text("import os\n", encoding="utf-8")
        (runtime_root / "P3-SAM" / "demo" / "auto_mask.py").write_text("print('ok')\n", encoding="utf-8")
        weight_path = self.injected_model_dir / "p3sam.safetensors"
        weight_path.write_bytes(b"weights")
        output_dir = self.workspace / "out"
        plan = build_execution_plan({"mesh": self.mesh}, runtime_context=self.ready_shell_context)

        def fake_runner(command, **kwargs):
            self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertEqual(kwargs["env"]["PYTHONUNBUFFERED"], "1")
            self.assertEqual(kwargs["env"][SONATA_CACHE_ENV_VAR], str(sonata_cache_root(self.workspace)))
            self.assertEqual(kwargs["cwd"], str(runtime_root / "P3-SAM" / "demo"))
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "auto_mask_mesh_final.glb").write_bytes(b"glTF")
            (output_dir / "auto_mask_mesh_final.ply").write_bytes(b"ply")
            (output_dir / "auto_mask_mesh_final_aabb.glb").write_bytes(b"glTF-aabb")
            import numpy as np

            np.save(output_dir / "auto_mask_mesh_final_aabb.npy", [[[0, 0, 0], [1, 1, 1]]])
            np.save(output_dir / "auto_mask_mesh_final_face_ids.npy", [1, 2, 3])
            return mock.Mock(returncode=0, stdout="ok", stderr="")

        with mock.patch(
            "runtime.p3_sam.verify_upstream_import_smoke",
            return_value={"ready": True, "status": "ready"},
        ):
            artifacts = run_upstream_p3_sam(
                plan,
                project_root=self.workspace,
                managed_python=managed_python,
                model_root=self.injected_model_dir,
                output_dir=output_dir,
                runner=fake_runner,
            )

        self.assertEqual(artifacts.primary_mesh_path, output_dir / "auto_mask_mesh_final.glb")
        self.assertEqual(artifacts.metadata["aabb_path"], str(output_dir / "auto_mask_mesh_final_aabb.npy"))

    def test_run_upstream_p3_sam_persists_failure_diagnostics(self) -> None:
        managed_python = self.workspace / "venv" / "bin" / "python"
        managed_python.parent.mkdir(parents=True)
        managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        runtime_root = runtime_source_root(self.workspace)
        (runtime_root / "P3-SAM" / "demo").mkdir(parents=True, exist_ok=True)
        (runtime_root / "XPart" / "partgen" / "models").mkdir(parents=True, exist_ok=True)
        (runtime_root / "P3-SAM" / "model.py").write_text("import os\n", encoding="utf-8")
        (runtime_root / "P3-SAM" / "demo" / "auto_mask.py").write_text("print('ok')\n", encoding="utf-8")
        weight_path = self.injected_model_dir / "p3sam.safetensors"
        weight_path.write_bytes(b"weights")
        output_dir = self.workspace / "out"
        plan = build_execution_plan({"mesh": self.mesh}, runtime_context=self.ready_shell_context)

        def failing_runner(_command, **_kwargs):
            return mock.Mock(returncode=2, stdout="stdout details\n", stderr="stderr details\n")

        with mock.patch(
            "runtime.p3_sam.verify_upstream_import_smoke",
            return_value={"ready": True, "status": "ready"},
        ):
            with self.assertRaises(RuntimeFailure) as ctx:
                run_upstream_p3_sam(
                    plan,
                    project_root=self.workspace,
                    managed_python=managed_python,
                    model_root=self.injected_model_dir,
                    output_dir=output_dir,
                    runner=failing_runner,
                )

        self.assertEqual(ctx.exception.code, "p3_sam_subprocess_failed")
        diagnostics = ctx.exception.details["diagnostics"]
        stdout_log = Path(diagnostics["stdout"])
        stderr_log = Path(diagnostics["stderr"])
        failure_json = Path(diagnostics["failure"])
        self.assertEqual(stdout_log.name, P3_SAM_STDOUT_LOG_NAME)
        self.assertEqual(stderr_log.name, P3_SAM_STDERR_LOG_NAME)
        self.assertEqual(failure_json.name, P3_SAM_FAILURE_JSON_NAME)
        self.assertEqual(stdout_log.read_text(encoding="utf-8"), "stdout details\n")
        self.assertEqual(stderr_log.read_text(encoding="utf-8"), "stderr details\n")
        failure_payload = json.loads(failure_json.read_text(encoding="utf-8"))
        self.assertEqual(failure_payload["returncode"], 2)
        self.assertEqual(failure_payload["stdout_log"], str(stdout_log))
        self.assertEqual(failure_payload["stderr_log"], str(stderr_log))

    def test_x_part_bundle_resolution_supports_nested_central_model_cache(self) -> None:
        model_root = self.workspace / "Modly" / "models" / "hunyuan3d-part"
        bundle_root = self._create_valid_x_part_bundle(model_root / "p3sam")
        nested_weights_only = model_root / "p3sam" / "p3sam"
        nested_weights_only.mkdir(parents=True, exist_ok=True)
        (nested_weights_only / "config.json").write_text("{}\n", encoding="utf-8")
        (nested_weights_only / "p3sam.safetensors").write_bytes(b"nested-weights")

        self.assertEqual(resolve_bundle_root(model_root), bundle_root)

    def test_x_part_bundle_resolution_rejects_nested_p3_sam_weights_directory(self) -> None:
        model_root = self.workspace / "Modly" / "models" / "hunyuan3d-part" / "p3sam"
        nested_weights_only = model_root / "p3sam"
        nested_weights_only.mkdir(parents=True, exist_ok=True)
        (nested_weights_only / "config.json").write_text("{}\n", encoding="utf-8")
        (nested_weights_only / "p3sam.safetensors").write_bytes(b"nested-weights")
        self._create_valid_x_part_bundle(model_root)

        self.assertEqual(resolve_bundle_root(model_root), model_root)

    def test_x_part_import_root_targets_xpart_directory(self) -> None:
        pipeline_path = self.workspace / ".upstream" / "hunyuan3d-part" / "XPart" / "partgen" / "partformer_pipeline.py"

        self.assertEqual(x_part_import_root(pipeline_path), pipeline_path.parents[1])
        self.assertEqual(x_part_import_root(pipeline_path).name, "XPart")

    def test_x_part_import_smoke_adds_xpart_root_for_partgen_import(self) -> None:
        runtime_root = runtime_source_root(self.workspace)
        pipeline_path = runtime_root / "XPart" / "partgen" / "partformer_pipeline.py"
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        (pipeline_path.parent / "__init__.py").write_text("", encoding="utf-8")
        pipeline_path.write_text(
            "class PartFormerPipeline:\n    pass\n",
            encoding="utf-8",
        )
        bundle_root = self._create_valid_x_part_bundle(self.workspace / "managed-models" / "hunyuan3d-part" / "p3sam")

        smoke = verify_x_part_import_smoke(
            project_root=self.workspace,
            managed_python=Path(sys.executable),
            model_root=self.workspace / "managed-models" / "hunyuan3d-part",
        )

        self.assertTrue(smoke["ready"], smoke)
        self.assertEqual(smoke["status"], "ready")
        self.assertEqual(smoke["class_name"], "PartFormerPipeline")
        self.assertEqual(smoke["import_root"], str(runtime_root / "XPart"))
        self.assertEqual(smoke["bundle_root"], str(bundle_root))
        self.assertTrue(smoke["bundle_validation"]["model_present"])
        self.assertIn("conditioner", smoke["bundle_validation"]["marker_dirs_present"])

    def test_x_part_subprocess_script_uses_same_xpart_import_root(self) -> None:
        runtime_root = runtime_source_root(self.workspace)
        plan = build_execution_plan(
            {"mesh": self.mesh},
            {"pipeline_stage": "x-part"},
            runtime_context=self.ready_context,
        )

        script = _build_subprocess_script(
            runtime_root=runtime_root,
            bundle_root=self.workspace / "managed-models" / "hunyuan3d-part" / "p3sam",
            mesh_path=self.mesh,
            output_dir=self.workspace / "xpart-out",
            params=plan.params,
            aabb_path=None,
        )

        self.assertIn(f"x_part_root = Path({str(runtime_root / 'XPart')!r})", script)
        self.assertIn("sys.path.insert(0, str(x_part_root))", script)
        self.assertNotIn("pipeline_path.parents[2]", script)

    def test_x_part_subprocess_script_imports_torch_and_moves_pipeline_to_cuda(self) -> None:
        runtime_root = runtime_source_root(self.workspace)
        plan = build_execution_plan(
            {"mesh": self.mesh},
            {"pipeline_stage": "x-part"},
            runtime_context=self.ready_context,
        )

        script = _build_subprocess_script(
            runtime_root=runtime_root,
            bundle_root=self.workspace / "managed-models" / "hunyuan3d-part" / "p3sam",
            mesh_path=self.mesh,
            output_dir=self.workspace / "xpart-out",
            params=plan.params,
            aabb_path=None,
        )

        self.assertIn("import torch", script)
        self.assertIn("if not torch.cuda.is_available():", script)
        self.assertIn("device = torch.device('cuda', torch.cuda.current_device())", script)
        self.assertIn("torch.backends.cuda.matmul.allow_tf32 = True", script)
        self.assertIn("if requested_dtype == 'float32':", script)
        self.assertIn("dtype = torch.float32", script)
        self.assertIn("elif requested_dtype == 'bfloat16'", script)
        self.assertIn("def sm121_safe_sdp_kernel", script)
        self.assertIn("enable_math=True", script)
        self.assertIn("resource_limits['sdp_kernel_policy'] = 'math_only_sm121_safe'", script)
        self.assertIn("sdp_kernel_policy='math_only_sm121_safe'", script)
        self.assertIn("memory_trace = []", script)
        self.assertIn("def capture_memory(stage, **extra):", script)
        self.assertIn("capture_memory('before_partgen_import')", script)
        self.assertIn("capture_memory('after_partgen_import')", script)
        self.assertIn("capture_memory('before_from_pretrained')", script)
        self.assertIn("capture_memory('after_from_pretrained')", script)
        self.assertIn("capture_memory('before_pipeline_call')", script)
        self.assertIn("pipeline = PartFormerPipeline.from_pretrained(str(bundle_root), device=device, dtype=dtype)", script)
        self.assertIn("pipeline.to(device=device, dtype=dtype)", script)
        self.assertLess(
            script.index("pipeline = PartFormerPipeline.from_pretrained(str(bundle_root), device=device, dtype=dtype)"),
            script.index("pipeline.to(device=device, dtype=dtype)"),
        )

    def test_x_part_subprocess_script_applies_real_resource_limits_not_ignored_max_parts_kwarg(self) -> None:
        runtime_root = runtime_source_root(self.workspace)
        plan = build_execution_plan(
            {"mesh": self.mesh},
            {"pipeline_stage": "x-part", "quality_preset": "quality", "max_parts": 64},
            runtime_context=self.ready_context,
        )

        script = _build_subprocess_script(
            runtime_root=runtime_root,
            bundle_root=self.workspace / "managed-models" / "hunyuan3d-part" / "p3sam",
            mesh_path=self.mesh,
            output_dir=self.workspace / "xpart-out",
            params=plan.params,
            aabb_path=self.workspace / "auto_mask_mesh_final_aabb.npy",
        )

        self.assertIn("resource_limits_path = output_dir / 'x_part_resource_limits.json'", script)
        self.assertIn("aabb = aabb[: int(resource_limits['effective_max_parts'])]", script)
        self.assertIn("original_predict_bbox = pipeline.predict_bbox", script)
        self.assertIn("pipeline.predict_bbox = guarded_predict_bbox", script)
        self.assertIn("num_inference_steps=int(resource_limits['num_inference_steps'])", script)
        self.assertIn("octree_resolution=int(resource_limits['octree_resolution'])", script)
        self.assertIn("num_chunks=int(resource_limits['num_chunks'])", script)
        self.assertIn("xpart_pipeline_module.load_surface_points = guarded_load_surface_points", script)
        self.assertIn("resource_limits['surface_point_count']", script)
        self.assertIn("xpart_pipeline_module.sample_bbox_points_from_trimesh = guarded_sample_bbox_points_from_trimesh", script)
        self.assertIn("resource_limits['bbox_point_count']", script)
        self.assertIn("capture_memory('before_sample_bbox_points'", script)
        self.assertIn("patched_encoder_pc_sizes = []", script)
        self.assertIn("target_pc_size = int(resource_limits['bbox_point_count']) if 'geo_encoder' in module_name else int(resource_limits['surface_point_count'])", script)
        self.assertIn("module.pc_size = min(old_pc_size, target_pc_size)", script)
        self.assertIn("module.pc_sharpedge_size = 0", script)
        self.assertIn("module.downsample_ratio = min(old_downsample_ratio, max_downsample_ratio)", script)
        self.assertIn("enable_pbar=False", script)
        self.assertIn("generator = torch.Generator(device=device).manual_seed(seed)", script)
        self.assertIn("scene = result[0] if isinstance(result, (list, tuple)) else result", script)
        self.assertNotIn("max_parts=max_parts", script)

    def test_x_part_adapter_fails_closed_when_bundle_is_missing(self) -> None:
        plan = build_execution_plan(
            {"mesh": self.mesh},
            {"pipeline_stage": "x-part"},
            runtime_context=self.ready_context,
        )
        managed_python = self.workspace / "venv" / "bin" / "python"
        managed_python.parent.mkdir(parents=True)
        managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with self.assertRaises(SetupFailure) as ctx:
            run_upstream_x_part(
                plan,
                project_root=self.workspace,
                managed_python=managed_python,
                model_root=self.workspace / "missing-model-root",
                output_dir=self.workspace / "xpart-out",
            )

        self.assertEqual(ctx.exception.code, "missing_x_part_bundle")
        self.assertEqual(ctx.exception.details["required_model_path"], "model/model.safetensors")
        self.assertEqual(ctx.exception.details["model_root"], str((self.workspace / "missing-model-root").resolve()))
        self.assertTrue(ctx.exception.details["candidates"])

    def test_x_part_bundle_resolution_reports_candidate_validation_details_when_missing(self) -> None:
        model_root = self.workspace / "managed-models" / "hunyuan3d-part"
        nested_weights_only = model_root / "p3sam" / "p3sam"
        nested_weights_only.mkdir(parents=True, exist_ok=True)
        (nested_weights_only / "config.json").write_text("{}\n", encoding="utf-8")
        (nested_weights_only / "p3sam.safetensors").write_bytes(b"nested-weights")

        with self.assertRaises(SetupFailure) as ctx:
            resolve_bundle_root(model_root)

        self.assertEqual(ctx.exception.code, "missing_x_part_bundle")
        candidates = ctx.exception.details["candidates"]
        self.assertEqual(candidates[0]["path"], str(model_root.resolve()))
        nested_candidate = next(item for item in candidates if item["path"] == str(nested_weights_only.resolve()))
        self.assertFalse(nested_candidate["valid"])
        self.assertFalse(nested_candidate["model_present"])
        self.assertTrue(nested_candidate["p3_sam_weight_present"])

    def test_x_part_adapter_fails_closed_when_dependencies_are_missing(self) -> None:
        plan = build_execution_plan(
            {"mesh": self.mesh},
            {"pipeline_stage": "x-part"},
            runtime_context=self.ready_context,
        )
        managed_python = self.workspace / "venv" / "bin" / "python"
        managed_python.parent.mkdir(parents=True)
        managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        runtime_root = runtime_source_root(self.workspace)
        (runtime_root / "XPart" / "partgen").mkdir(parents=True, exist_ok=True)
        (runtime_root / "XPart" / "partgen" / "partformer_pipeline.py").write_text("print('ok')\n", encoding="utf-8")
        bundle_root = self._create_valid_x_part_bundle(self.workspace / "managed-models" / "hunyuan3d-part" / "p3sam")

        with mock.patch(
            "runtime.x_part.verify_x_part_import_smoke",
            return_value={"ready": False, "status": "blocked", "missing_modules": ["partgen"], "stderr": "ModuleNotFoundError"},
        ):
            with self.assertRaises(SetupFailure) as ctx:
                run_upstream_x_part(
                    plan,
                    project_root=self.workspace,
                    managed_python=managed_python,
                    model_root=self.workspace / "managed-models" / "hunyuan3d-part",
                    output_dir=self.workspace / "xpart-out",
                )

        self.assertEqual(ctx.exception.code, "x_part_adapter_unavailable")
        self.assertEqual(ctx.exception.details["missing_modules"], ["partgen"])

    def test_run_upstream_x_part_persists_failure_diagnostics(self) -> None:
        plan = build_execution_plan(
            {"mesh": self.mesh},
            {"pipeline_stage": "x-part"},
            runtime_context=self.ready_context,
        )
        managed_python = self.workspace / "venv" / "bin" / "python"
        managed_python.parent.mkdir(parents=True)
        managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        output_dir = self.workspace / "xpart-out"
        bundle_root = self._create_valid_x_part_bundle(self.workspace / "managed-models" / "hunyuan3d-part" / "p3sam")

        def failing_runner(_command, **_kwargs):
            return mock.Mock(returncode=7, stdout="xpart stdout\n", stderr="xpart stderr\n")

        with mock.patch(
            "runtime.x_part.verify_x_part_import_smoke",
            return_value={"ready": True, "status": "ready"},
        ):
            with self.assertRaises(RuntimeFailure) as ctx:
                run_upstream_x_part(
                    plan,
                    project_root=self.workspace,
                    managed_python=managed_python,
                    model_root=self.workspace / "managed-models" / "hunyuan3d-part",
                    output_dir=output_dir,
                    aabb_path=self.workspace / "auto_mask_mesh_final_aabb.npy",
                    runner=failing_runner,
                )

        self.assertEqual(ctx.exception.code, "x_part_subprocess_failed")
        self.assertEqual(ctx.exception.details["mesh_path"], str(self.mesh))
        self.assertEqual(ctx.exception.details["output_dir"], str(output_dir))
        diagnostics = ctx.exception.details["diagnostics"]
        stdout_log = Path(diagnostics["stdout"])
        stderr_log = Path(diagnostics["stderr"])
        failure_json = Path(diagnostics["failure"])
        self.assertEqual(stdout_log.name, X_PART_STDOUT_LOG_NAME)
        self.assertEqual(stderr_log.name, X_PART_STDERR_LOG_NAME)
        self.assertEqual(failure_json.name, X_PART_FAILURE_JSON_NAME)
        self.assertEqual(stdout_log.read_text(encoding="utf-8"), "xpart stdout\n")
        self.assertEqual(stderr_log.read_text(encoding="utf-8"), "xpart stderr\n")
        failure_payload = json.loads(failure_json.read_text(encoding="utf-8"))
        self.assertEqual(failure_payload["returncode"], 7)
        self.assertEqual(failure_payload["bundle_root"], str(bundle_root))
        self.assertEqual(failure_payload["mesh_path"], str(self.mesh))
        self.assertEqual(failure_payload["output_dir"], str(output_dir))
        self.assertEqual(failure_payload["aabb_path"], str(self.workspace / "auto_mask_mesh_final_aabb.npy"))
        self.assertEqual(failure_payload["stdout_log"], str(stdout_log))
        self.assertEqual(failure_payload["stderr_log"], str(stderr_log))
        self.assertEqual(failure_payload["script_summary"]["entrypoint"], "XPart/partgen/partformer_pipeline.py")

    def test_run_upstream_x_part_timeout_persists_failure_diagnostics(self) -> None:
        plan = build_execution_plan(
            {"mesh": self.mesh},
            {"pipeline_stage": "x-part"},
            runtime_context=self.ready_context,
        )
        managed_python = self.workspace / "venv" / "bin" / "python"
        managed_python.parent.mkdir(parents=True)
        managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        output_dir = self.workspace / "xpart-timeout"
        self._create_valid_x_part_bundle(self.workspace / "managed-models" / "hunyuan3d-part" / "p3sam")

        def timeout_runner(command, **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs["timeout"], output="partial stdout", stderr="partial stderr")

        with mock.patch(
            "runtime.x_part.verify_x_part_import_smoke",
            return_value={"ready": True, "status": "ready"},
        ):
            with self.assertRaises(RuntimeFailure) as ctx:
                run_upstream_x_part(
                    plan,
                    project_root=self.workspace,
                    managed_python=managed_python,
                    model_root=self.workspace / "managed-models" / "hunyuan3d-part",
                    output_dir=output_dir,
                    runner=timeout_runner,
                )

        self.assertEqual(ctx.exception.code, "x_part_subprocess_timeout")
        diagnostics = ctx.exception.details["diagnostics"]
        failure_payload = json.loads(Path(diagnostics["failure"]).read_text(encoding="utf-8"))
        self.assertTrue(failure_payload["timed_out"])
        self.assertEqual(failure_payload["returncode"], -9)
        self.assertIn("partial stdout", Path(diagnostics["stdout"]).read_text(encoding="utf-8"))
        self.assertIn("timed out", Path(diagnostics["stderr"]).read_text(encoding="utf-8"))

    def test_x_part_memory_guard_threshold_is_configurable(self) -> None:
        self.assertEqual(_resolve_x_part_min_available_memory_gib({}), 16.0)
        self.assertEqual(_resolve_x_part_min_available_memory_gib({X_PART_MEMORY_GUARD_ENV_VAR: "24"}), 24.0)

        with self.assertRaises(SetupFailure) as ctx:
            _resolve_x_part_min_available_memory_gib({X_PART_MEMORY_GUARD_ENV_VAR: "not-a-number"})

        self.assertEqual(ctx.exception.code, "invalid_x_part_memory_guard")

    def test_x_part_memory_guard_terminates_subprocess_before_host_oom(self) -> None:
        with mock.patch("runtime.x_part._read_mem_available_gib", return_value=0.5):
            result, guard = _run_with_memory_guard(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=str(self.workspace),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                timeout_seconds=60,
                min_available_gib=16.0,
            )

        self.assertTrue(guard and guard["triggered"])
        self.assertIn("memory guard terminated", result.stderr.lower())

    def test_full_stage_chains_p3_sam_aabb_into_x_part(self) -> None:
        plan = build_execution_plan(
            {"mesh": self.mesh},
            {"pipeline_stage": "full"},
            runtime_context=self.ready_context,
        )
        aabb_path = self.workspace / "chain" / "auto_mask_mesh_final_aabb.npy"
        aabb_path.parent.mkdir(parents=True, exist_ok=True)
        aabb_path.write_bytes(b"npy")
        p3_artifacts = DecompositionArtifacts(
            primary_mesh_path=self.mesh,
            parts=(),
            segmentation={"source": "p3-sam"},
            bboxes={"parts": []},
            completion={"status": "completed"},
            metadata={"aabb_path": str(aabb_path)},
        )
        x_part_artifacts = DecompositionArtifacts(
            primary_mesh_path=self.mesh,
            parts=(),
            segmentation={"source": "x-part"},
            bboxes={"parts": []},
            completion={"status": "completed"},
            metadata={"adapter": "x-part"},
        )

        with (
            mock.patch("runtime.pipeline.run_upstream_p3_sam", return_value=p3_artifacts) as p3_mock,
            mock.patch("runtime.pipeline.run_upstream_x_part", return_value=x_part_artifacts) as x_part_mock,
        ):
            result = run_pipeline_stage(
                plan,
                project_root=self.workspace,
                managed_python=self.workspace / "venv" / "bin" / "python",
                model_root=self.workspace / "models",
                output_dir=self.workspace / "out",
            )

        self.assertIs(result, x_part_artifacts)
        p3_mock.assert_called_once()
        self.assertEqual(x_part_mock.call_args.kwargs["aabb_path"], aabb_path)

    def test_p3_sam_stage_analysis_attaches_semantic_report_after_effective_artifacts(self) -> None:
        plan = build_execution_plan(
            {"mesh": self.mesh},
            {"pipeline_stage": "p3-sam", "semantic_resolver": "analysis", "max_parts": 1},
            runtime_context=self.ready_context,
        )
        effective_aabb_path = self.workspace / "p3" / "auto_mask_mesh_final_aabb_effective.npy"
        effective_aabb_path.parent.mkdir(parents=True, exist_ok=True)
        effective_aabb_path.write_bytes(b"effective")
        p3_artifacts = DecompositionArtifacts(
            primary_mesh_path=self.mesh,
            parts=(),
            segmentation={
                "source": "p3-sam",
                "raw_part_count": 3,
                "effective_part_count": 1,
                "max_parts": 1,
                "selection_policy": "face_count_desc",
                "effective_parts": [{"part_id": 0, "raw_part_id": 2, "face_count": 42}],
            },
            bboxes={"source": "p3-sam", "parts": [{"part_id": "part-0", "raw_part_id": 2, "min": [0, 0, 0], "max": [1, 1, 1]}]},
            completion={"status": "completed"},
            metadata={"effective_aabb_path": str(effective_aabb_path)},
        )

        with mock.patch("runtime.pipeline.run_upstream_p3_sam", return_value=p3_artifacts):
            result = run_pipeline_stage(
                plan,
                project_root=self.workspace,
                managed_python=self.workspace / "venv" / "bin" / "python",
                model_root=self.workspace / "models",
                output_dir=self.workspace / "p3-out",
            )

        report = result.metadata["semantic_report"]
        self.assertEqual(report["stage"], "p3-sam")
        self.assertEqual(report["raw_part_count"], 3)
        self.assertEqual(report["effective_part_count"], 1)
        self.assertEqual(report["inputs"]["effective_aabb_path"], str(effective_aabb_path))

    def test_p3_sam_stage_off_does_not_attach_semantic_report(self) -> None:
        plan = build_execution_plan(
            {"mesh": self.mesh},
            {"pipeline_stage": "p3-sam", "semantic_resolver": "off"},
            runtime_context=self.ready_context,
        )
        p3_artifacts = DecompositionArtifacts(
            primary_mesh_path=self.mesh,
            parts=(),
            segmentation={"source": "p3-sam"},
            bboxes={"parts": []},
            completion={"status": "completed"},
            metadata={"source": "unit-test"},
        )

        with mock.patch("runtime.pipeline.run_upstream_p3_sam", return_value=p3_artifacts):
            result = run_pipeline_stage(
                plan,
                project_root=self.workspace,
                managed_python=self.workspace / "venv" / "bin" / "python",
                model_root=self.workspace / "models",
                output_dir=self.workspace / "p3-off-out",
            )

        self.assertIs(result, p3_artifacts)
        self.assertNotIn("semantic_report", result.metadata)

    def test_full_stage_analysis_preserves_same_x_part_aabb_path_as_off(self) -> None:
        raw_aabb_path = self.workspace / "chain-analysis" / "auto_mask_mesh_final_aabb.npy"
        effective_aabb_path = self.workspace / "chain-analysis" / "auto_mask_mesh_final_aabb_effective.npy"
        raw_aabb_path.parent.mkdir(parents=True, exist_ok=True)
        raw_aabb_path.write_bytes(b"raw")
        effective_aabb_path.write_bytes(b"effective")
        p3_artifacts = DecompositionArtifacts(
            primary_mesh_path=self.mesh,
            parts=(),
            segmentation={
                "source": "p3-sam",
                "raw_part_count": 2,
                "effective_part_count": 1,
                "selection_policy": "face_count_desc",
                "effective_parts": [{"part_id": 0, "raw_part_id": 1, "face_count": 7}],
            },
            bboxes={"parts": [{"part_id": "part-0", "raw_part_id": 1, "min": [0, 0, 0], "max": [1, 1, 1]}]},
            completion={"status": "completed"},
            metadata={"aabb_path": str(raw_aabb_path), "effective_aabb_path": str(effective_aabb_path)},
        )
        x_part_artifacts = DecompositionArtifacts(
            primary_mesh_path=self.mesh,
            parts=(),
            segmentation={"source": "x-part"},
            bboxes={"parts": []},
            completion={"status": "completed"},
            metadata={"adapter": "x-part"},
        )
        observed_paths: list[Path | None] = []

        for resolver in ("off", "analysis"):
            plan = build_execution_plan(
                {"mesh": self.mesh},
                {"pipeline_stage": "full", "semantic_resolver": resolver},
                runtime_context=self.ready_context,
            )
            with (
                mock.patch("runtime.pipeline.run_upstream_p3_sam", return_value=p3_artifacts),
                mock.patch("runtime.pipeline.run_upstream_x_part", return_value=x_part_artifacts) as x_part_mock,
            ):
                result = run_pipeline_stage(
                    plan,
                    project_root=self.workspace,
                    managed_python=self.workspace / "venv" / "bin" / "python",
                    model_root=self.workspace / "models",
                    output_dir=self.workspace / f"full-{resolver}",
                )
            observed_paths.append(x_part_mock.call_args.kwargs["aabb_path"])
            if resolver == "analysis":
                self.assertEqual(result.metadata["semantic_report"]["stage"], "full")

        self.assertEqual(observed_paths, [effective_aabb_path, effective_aabb_path])

    def test_x_part_stage_analysis_attaches_aabb_only_fallback_without_changing_execution(self) -> None:
        import numpy as np

        aabb_path = self.workspace / "external-aabb.npy"
        np.save(aabb_path, np.array([[[0, 0, 0], [1, 1, 1]]], dtype=float))
        plan = build_execution_plan(
            {"mesh": self.mesh},
            {"pipeline_stage": "x-part", "semantic_resolver": "analysis", "aabb_path": str(aabb_path)},
            runtime_context=self.ready_context,
        )
        x_part_artifacts = DecompositionArtifacts(
            primary_mesh_path=self.mesh,
            parts=(),
            segmentation={"source": "x-part"},
            bboxes={"parts": []},
            completion={"status": "completed"},
            metadata={"adapter": "x-part"},
        )

        with mock.patch("runtime.pipeline.run_upstream_x_part", return_value=x_part_artifacts) as x_part_mock:
            result = run_pipeline_stage(
                plan,
                project_root=self.workspace,
                managed_python=self.workspace / "venv" / "bin" / "python",
                model_root=self.workspace / "models",
                output_dir=self.workspace / "xpart-analysis-out",
            )

        self.assertEqual(x_part_mock.call_args.kwargs["aabb_path"], aabb_path)
        report = result.metadata["semantic_report"]
        self.assertEqual(report["stage"], "x-part")
        self.assertIn("aabb_only_semantic_report_limited", report["warnings"])

    def test_full_stage_prefers_effective_p3_sam_aabb_when_present(self) -> None:
        raw_aabb_path = self.workspace / "chain" / "auto_mask_mesh_final_aabb.npy"
        effective_aabb_path = self.workspace / "chain" / "auto_mask_mesh_final_aabb_effective.npy"
        raw_aabb_path.parent.mkdir(parents=True, exist_ok=True)
        raw_aabb_path.write_bytes(b"raw")
        effective_aabb_path.write_bytes(b"effective")
        artifacts = DecompositionArtifacts(
            primary_mesh_path=self.mesh,
            parts=(),
            segmentation={"source": "p3-sam"},
            bboxes={"parts": []},
            completion={"status": "completed"},
            metadata={"aabb_path": str(raw_aabb_path), "effective_aabb_path": str(effective_aabb_path)},
        )

        self.assertEqual(_require_p3_sam_aabb(artifacts), effective_aabb_path)

    def test_generator_load_is_shell_only_and_does_not_claim_inference_ready(self) -> None:
        generator = Hunyuan3DPartGenerator(
            self.injected_model_dir,
            self.workspace,
            project_root=self.workspace,
            runtime_context=self.ready_shell_context,
        )
        loaded = generator.load()
        self.assertIs(loaded, generator)
        self.assertTrue(generator._shell_loaded)
        self.assertFalse(generator.readiness_status()["inference_ready"])

    def test_generator_generate_requires_mesh_path_param(self) -> None:
        generator = Hunyuan3DPartGenerator(
            self.injected_model_dir,
            self.workspace,
            project_root=self.workspace,
            runtime_context=self.ready_shell_context,
        )
        with self.assertRaises(ValidationError) as ctx:
            generator.generate(b"image-bytes", {"pipeline_stage": "p3-sam"})
        self.assertEqual(ctx.exception.code, "missing_mesh_input")

    def test_generator_generate_refuses_to_fake_inference(self) -> None:
        generator = Hunyuan3DPartGenerator(
            self.injected_model_dir,
            self.workspace,
            project_root=self.workspace,
            runtime_context=self.ready_shell_context,
        )
        with self.assertRaises(RuntimeFailure) as ctx:
            generator.generate(b"image-bytes", {"mesh_path": str(self.mesh)})
        self.assertEqual(ctx.exception.code, "runtime_unavailable")
        blockers = {item["component"] for item in ctx.exception.details["blockers"]}
        self.assertEqual(blockers, {"runtime_adapter", "weights"})
        self.assertEqual(ctx.exception.details["mesh_path"], str(self.mesh.resolve()))

    def test_generator_generate_does_not_block_on_stale_cuda_gate_when_torch_reports_cuda(self) -> None:
        stale_context = resolve_runtime_context(
            project_root=self.workspace,
            host_facts=HostFacts(
                os_name="linux",
                arch="x86_64",
                python_version="3.11.8",
                python_abi="cp311",
                cuda_visible=False,
            ),
            dependency_checker=lambda _: object(),
        )
        managed_python = self.workspace / "venv" / "bin" / "python"
        managed_python.parent.mkdir(parents=True, exist_ok=True)
        managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        weight_path = self.injected_model_dir / "p3sam.safetensors"
        weight_path.write_bytes(b"weights")
        adapter_readiness = AdapterReadiness(
            ready=True,
            status="ready",
            paths=None,
            components={
                "managed_python": {"ready": True, "status": "ready"},
                "runtime_source": {"ready": True, "status": "ready"},
                "entrypoint": {"ready": True, "status": "ready"},
                "import_smoke": {"ready": True, "status": "ready"},
                "weights": {"ready": True, "status": "ready", "path": str(weight_path)},
            },
            message="Runtime adapter ready.",
        )
        generator = Hunyuan3DPartGenerator(
            self.injected_model_dir,
            self.workspace,
            project_root=self.workspace,
            runtime_context=stale_context,
        )

        with (
            mock.patch("generator.probe_torch_cuda_availability", return_value=True),
            mock.patch("generator.build_adapter_readiness", return_value=adapter_readiness),
            mock.patch("generator.decompose_mesh", return_value={"primary_mesh": str(self.mesh), "status": "ok"}) as decompose_mock,
        ):
            result = generator.generate(b"image-bytes", {"mesh_path": str(self.mesh)})

        self.assertEqual(result, str(self.mesh))
        decompose_mock.assert_called_once()

    def test_generator_progress_callback_prefers_modly_two_arg_contract(self) -> None:
        generator = Hunyuan3DPartGenerator(
            self.injected_model_dir,
            self.workspace,
            project_root=self.workspace,
            runtime_context=self.ready_shell_context,
        )
        calls: list[tuple[int, str]] = []

        def progress_cb(percent: int, message: str) -> None:
            calls.append((percent, message))

        with self.assertRaises(ValidationError):
            generator.generate(b"image-bytes", {"pipeline_stage": "p3-sam"}, progress_cb=progress_cb)

        self.assertEqual(calls, [(5, "Generator shell prepared; validating mesh input and runtime readiness.")])
        self.assertIsInstance(calls[0][0], int)
        self.assertIsInstance(calls[0][1], str)

    def test_generator_progress_callback_falls_back_to_single_payload_contract(self) -> None:
        generator = Hunyuan3DPartGenerator(
            self.injected_model_dir,
            self.workspace,
            project_root=self.workspace,
            runtime_context=self.ready_shell_context,
        )
        payloads: list[dict[str, object]] = []

        def progress_cb(payload: dict[str, object]) -> None:
            payloads.append(payload)

        with self.assertRaises(ValidationError):
            generator.generate(b"image-bytes", {"pipeline_stage": "p3-sam"}, progress_cb=progress_cb)

        self.assertEqual(
            payloads,
            [
                {
                    "stage": "shell",
                    "percent": 5,
                    "message": "Generator shell prepared; validating mesh input and runtime readiness.",
                }
            ],
        )

    def test_generator_fails_closed_when_runtime_source_is_missing_even_if_weights_exist(self) -> None:
        managed_python = self.workspace / "venv" / "bin" / "python"
        managed_python.parent.mkdir(parents=True)
        managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        weight_path = self.injected_model_dir / "p3sam.safetensors"
        weight_path.write_bytes(b"weights")

        generator = Hunyuan3DPartGenerator(
            self.injected_model_dir,
            self.workspace,
            project_root=self.workspace,
            runtime_context=self.ready_shell_context,
        )
        with self.assertRaises(RuntimeFailure) as ctx:
            generator.generate(b"image-bytes", {"mesh_path": str(self.mesh)})
        self.assertEqual(ctx.exception.code, "runtime_unavailable")
        blockers = {item["component"] for item in ctx.exception.details["blockers"]}
        self.assertEqual(blockers, {"runtime_adapter"})

    def test_generator_generate_resolves_workspace_relative_mesh_path_before_runtime_gate(self) -> None:
        os.environ["WORKSPACE_DIR"] = str(self.workspace)
        generator = Hunyuan3DPartGenerator(
            self.injected_model_dir,
            self.workspace,
            project_root=self.workspace,
            runtime_context=self.ready_shell_context,
        )
        generator.outputs_dir = self.workflows_dir
        with self.assertRaises(RuntimeFailure) as ctx:
            generator.generate(b"image-bytes", {"mesh_path": "Workflows/sample.glb"})
        self.assertEqual(ctx.exception.code, "runtime_unavailable")
        self.assertEqual(ctx.exception.details["mesh_path"], str(self.workflow_mesh.resolve()))

    def test_repo_has_no_processor_contract_surface(self) -> None:
        self.assertFalse((ROOT / "processor.py").exists())

    def test_export_bundle_primary_keeps_part_glbs_out_of_return_payload(self) -> None:
        def fake_adapter(_plan):
            return DecompositionArtifacts(
                primary_mesh_path=self.mesh,
                parts=(
                    PartArtifact(part_id="arm", mesh_path=self.part_a, bbox={"min": [0, 0, 0], "max": [1, 1, 1]}),
                    PartArtifact(part_id="leg", mesh_path=self.part_b, bbox={"min": [1, 1, 1], "max": [2, 2, 2]}),
                ),
                segmentation={"parts": [{"id": "arm"}, {"id": "leg"}]},
                bboxes={"arm": {"min": [0, 0, 0], "max": [1, 1, 1]}},
                completion={"status": "deferred-runtime-adapter"},
                metadata={"source": "unit-test"},
            )

        result = decompose_mesh(
            {"mesh": self.mesh},
            output_dir=self.workspace / "out",
            runtime_adapter=fake_adapter,
            runtime_context=self.ready_context,
        )

        bundle = json.loads(Path(result["bundle_manifest"]).read_text(encoding="utf-8"))
        self.assertTrue(Path(result["primary_mesh"]).exists())
        self.assertEqual(bundle["routing_contract"]["single_mesh_primary"], True)
        self.assertEqual(bundle["routing_contract"]["output_mode"], "primary")
        self.assertEqual(bundle["routing_contract"]["parts_visibility"], "hidden_from_primary_outputs")
        self.assertTrue(bundle["routing_contract"]["parts_are_debug_only"])
        self.assertFalse(bundle["routing_contract"]["semantic"])
        self.assertEqual(bundle["observability"]["pipeline_stage"], "p3-sam")
        self.assertEqual(result["parts"], [])
        self.assertEqual(bundle["sidecars"]["parts"], [])
        self.assertFalse((self.workspace / "out" / "parts").exists())

    def test_export_bundle_writes_semantic_report_sidecar_and_observability_without_changing_primary(self) -> None:
        semantic_report = {
            "schema": "hunyuan3d.semantic_report.v1",
            "mode": "analysis",
            "stage": "p3-sam",
            "semantic": False,
            "publishable": False,
            "effective_part_count": 1,
            "confidence": {"aggregate": 0.2, "level": "low", "publishable": False},
            "warnings": ["insufficient_parts_for_regional_inference"],
        }

        def fake_adapter(_plan):
            return DecompositionArtifacts(
                primary_mesh_path=self.mesh,
                parts=(),
                segmentation={"source": "p3-sam", "semantic": False},
                bboxes={"parts": []},
                completion={"status": "completed"},
                metadata={"source": "unit-test", "semantic_report": semantic_report},
            )

        result = decompose_mesh(
            {"mesh": self.mesh},
            {"semantic_resolver": "analysis", "output_mode": "primary"},
            output_dir=self.workspace / "out-semantic",
            runtime_adapter=fake_adapter,
            runtime_context=self.ready_context,
        )

        bundle = json.loads(Path(result["bundle_manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(result["primary_mesh"], str(self.workspace / "out-semantic" / Path(result["primary_mesh"]).name))
        self.assertEqual(result["semantic_report"], str(self.workspace / "out-semantic" / "semantic_report.json"))
        self.assertTrue(Path(result["semantic_report"]).is_file())
        self.assertEqual(bundle["sidecars"]["semantic_report"], "semantic_report.json")
        self.assertEqual(bundle["observability"]["artifact_paths"]["semantic_report"], "semantic_report.json")
        self.assertEqual(bundle["observability"]["artifact_paths"]["semantic_summary"]["effective_part_count"], 1)
        self.assertEqual(result["parts"], [])

    def test_export_bundle_analysis_keeps_metadata_but_not_part_glbs(self) -> None:
        def fake_adapter(_plan):
            return DecompositionArtifacts(
                primary_mesh_path=self.mesh,
                parts=(PartArtifact(part_id="arm", mesh_path=self.part_a, bbox={"min": [0, 0, 0], "max": [1, 1, 1]}),),
                segmentation={"source": "x-part", "parts": [{"id": "arm"}], "semantic": False},
                bboxes={"parts": [{"part_id": "arm"}]},
                completion={"status": "completed"},
                metadata={"source": "unit-test"},
            )

        result = decompose_mesh(
            {"mesh": self.mesh},
            {"output_mode": "analysis"},
            output_dir=self.workspace / "out-analysis",
            runtime_adapter=fake_adapter,
            runtime_context=self.ready_context,
        )

        self.assertEqual(result["parts"], [])
        self.assertTrue(Path(result["segmentation"]).is_file())
        self.assertTrue(Path(result["bboxes"]).is_file())
        self.assertTrue(Path(str(result["completion"])).is_file())
        self.assertFalse((self.workspace / "out-analysis" / "parts").exists())

    def test_export_bundle_debug_copies_and_lists_part_glbs(self) -> None:
        def fake_adapter(_plan):
            return DecompositionArtifacts(
                primary_mesh_path=self.mesh,
                parts=(
                    PartArtifact(part_id="arm", mesh_path=self.part_a, bbox={"min": [0, 0, 0], "max": [1, 1, 1]}),
                    PartArtifact(part_id="leg", mesh_path=self.part_b, bbox={"min": [1, 1, 1], "max": [2, 2, 2]}),
                ),
                segmentation={"source": "x-part", "semantic": False},
                bboxes={"parts": []},
                completion={"status": "completed"},
                metadata={"source": "unit-test"},
            )

        result = decompose_mesh(
            {"mesh": self.mesh},
            {"output_mode": "debug"},
            output_dir=self.workspace / "out-debug",
            runtime_adapter=fake_adapter,
            runtime_context=self.ready_context,
        )

        bundle = json.loads(Path(result["bundle_manifest"]).read_text(encoding="utf-8"))
        self.assertEqual([part["part_id"] for part in result["parts"]], ["arm", "leg"])
        self.assertEqual([part["part_id"] for part in bundle["sidecars"]["parts"]], ["arm", "leg"])
        self.assertEqual(bundle["routing_contract"]["output_mode"], "debug")
        self.assertEqual(bundle["routing_contract"]["parts_visibility"], "debug_sidecars")
        self.assertTrue((self.workspace / "out-debug" / "parts" / "arm.glb").is_file())

    def test_x_part_contract_does_not_expose_fake_semantic_labels(self) -> None:
        runtime_root = runtime_source_root(self.workspace)
        plan = build_execution_plan(
            {"mesh": self.mesh},
            {"pipeline_stage": "x-part"},
            runtime_context=self.ready_context,
        )

        script = _build_subprocess_script(
            runtime_root=runtime_root,
            bundle_root=self.workspace / "managed-models" / "hunyuan3d-part" / "p3sam",
            mesh_path=self.mesh,
            output_dir=self.workspace / "xpart-out",
            params=plan.params,
            aabb_path=None,
        )

        self.assertNotIn("semantic_targets", script)
        self.assertNotIn("garment", script)
        self.assertNotIn("hair", script)

if __name__ == "__main__":
    unittest.main()
