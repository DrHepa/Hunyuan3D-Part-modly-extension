# Hunyuan3D-Part Modly Extension

Hunyuan3D-Part is a Modly model extension/adaptor for Tencent's Hunyuan3D-Part project. It exposes mesh-guided part decomposition in Modly through a managed model setup flow, using the upstream `tencent/Hunyuan3D-Part` assets and runtime components when they are available on the host.

This repository contains the Modly extension metadata, setup/readiness integration, runtime adapter code, and local tests for that adapter.

## What this repository is not

This repository does **not** include, vendor, or relicense Tencent Hunyuan3D-Part, its model weights, P3-SAM, X-Part, or their upstream dependencies. Those projects, assets, and models remain under their own licenses and terms.

The extension is an integration layer for Modly. It is not a standalone redistribution of the upstream model stack.

## Requirements

- Modly with extension support.
- Managed Python setup through Modly/Electron. The extension declares `python-root-setup-py` with explicit managed setup owned by Electron.
- NVIDIA CUDA-capable hardware and compatible native dependencies for real inference.
- Upstream Hunyuan3D-Part weights and assets from `tencent/Hunyuan3D-Part`.
- Sufficient GPU memory for the selected pipeline and quality preset. The manifest currently declares a 24 GB VRAM target.

## Support status

- **Linux NVIDIA:** validated locally for the current adapter/runtime path. Recent local validation reported `214 passed, 12 subtests passed` after Windows-readiness promotion.
- **Windows:** setup and readiness metadata are compatible and fail closed. Full Windows NVIDIA inference is **not** advertised until CUDA, native dependency, runtime, and weight probes pass on a real target host.
- **Linux ARM64:** experimental and locally validated in limited conditions, with elevated native dependency risk. Treat this path as platform-sensitive until reproduced on the target environment.

## Usage in Modly

Install or add this extension from GitHub or from a local checkout using the Modly UI flow for extensions. The manifest source is:

```text
https://github.com/DrHepa/Hunyuan3D-Part-modly-extension
```

After the extension is added, use Modly's managed setup flow to prepare the Python environment and dependencies. Real inference requires the upstream assets and a CUDA-ready host.

This README intentionally does not claim a CLI/headless install path. Follow the supported Modly UI workflow for extension setup and execution.

## Model node

The extension declares one model node:

- `decompose-mesh` — decomposes a required input mesh. The manifest exposes this as a single mesh-primary renderer node; optional image evidence is runtime-only and is not declared as a renderer input.

Inputs:

- `mesh` — required mesh input: `glb`, `obj`, `stl`, or `ply`.

Output:

- Primary mesh output in `glb` format.

## Main parameters

| Parameter | Values / purpose |
| --- | --- |
| `pipeline_stage` | `p3-sam`, `x-part`, or `full`. Selects P3-SAM segmentation, X-Part generation, or the chained pipeline. |
| `max_parts` | Positive integer part cap. The runtime enforces a hard safety cap. Parts are geometric decomposition units, not semantic labels. |
| `output_mode` | `primary`, `analysis`, or `debug`. Controls sidecar exposure in addition to the primary mesh. |
| `semantic_resolver` | `off` or `analysis`. `analysis` writes diagnostic reports only; guided semantic decomposition is reserved and not accepted in this MVP. |
| `seed` | Deterministic seed forwarded to the upstream runtime. |
| `export_format` | `glb`. The primary output format is intentionally constrained for safe Modly routing. |
| `quality_preset` | `fast`, `balanced`, or `quality`. Controls X-Part execution knobs such as diffusion steps, octree resolution, chunking, and CUDA dtype placement. |

## Outputs and sidecars

Every successful run produces a primary GLB mesh for Modly routing. Depending on the selected mode and resolver, additional run-scoped artifacts may be written:

- `analysis/<output_stem>/...` — analysis metadata scoped to the run output stem.
- `semantic_report.json` — diagnostic semantic resolver report when `semantic_resolver=analysis`.
- `comparison_report.json` — non-authoritative comparison/diagnostic report for analysis workflows.
- `parts/` — per-part GLB sidecars in `debug` output mode.
- Additional adapter metadata such as segmentation, bounding boxes, completion, and bundle manifests may be present as implementation sidecars.

## Semantic limitations

Current parts are geometric decomposition results. They are **not** authoritative garment, body, hair, material, or object labels.

`semantic_resolver=analysis` is diagnostic only. It does not mutate generation, `max_parts`, `output_mode`, routing, or X-Part inputs. Guided semantic decomposition is reserved for future work and is not accepted by the current manifest.

## Readiness and setup diagnostics

Real installation is managed by Modly/Electron. For local diagnostics from a checkout, the setup script supports readiness probes:

```bash
python setup.py readiness
python setup.py readiness --json
```

These commands report whether the host, CUDA visibility, native dependencies, adapter imports, and weights are ready. The readiness contract is fail-closed: if required platform/runtime conditions are missing, inference should not be treated as available.

## Local testing

The local Python test suite can be run with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src pytest tests -q
```

Do not treat passing local tests as proof that a new host can run full inference. CUDA, native wheels, upstream assets, and platform-specific runtime probes still matter.

## Security and privacy

Diagnostic reports summarize optional image input as metadata only. They do not persist image bytes, base64 payloads, image hashes, or sensitive input paths in the semantic image-evidence sections.

## References and acknowledgements

- Tencent Hunyuan3D-Part: <https://huggingface.co/tencent/Hunyuan3D-Part>
- Upstream projects and dependencies, including Hunyuan3D-Part, P3-SAM, X-Part, model weights, CUDA/native libraries, and Python packages, retain their own licenses and terms.

## License

This repository and Modly extension adapter code are released under the MIT License. See [`LICENSE`](LICENSE).

Upstream assets, code, models, and dependencies are governed by their own licenses and terms and are not relicensed by this repository.
