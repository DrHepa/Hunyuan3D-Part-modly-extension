# Manual Verification Plan

## Goal
Validate the intended `mesh -> Hunyuan3D-Part -> UniRig -> Kimodo` workflow once an Electron-managed runtime adapter exists.

## Current Status
- **Contract-complete now**: local MVP planning covers manifest contract, input validation, deterministic parameter resolution, readiness gating, deferred-stage rejection, and primary-mesh-plus-sidecars export behavior.
- **Real adapter still pending**: this repo does **not** contain a live Electron-managed P3-SAM inference adapter, so successful segmentation/inference cannot be claimed locally.
- **Execution status for task 5.3**: prepared, not executed on this host unless the preflight checks below all pass.

## Safe Preflight (read-only)
1. Run `python3 -m unittest discover -s tests -v` and expect contract tests to pass.
2. Run `python3 setup.py readiness --json` and execute the live workflow only if `support.status == "ready"`.
3. Confirm a real Electron-managed extension execution surface is actually available; do **not** invent headless Electron support.
4. Stop immediately if the host remains blocked (for example Linux ARM64 risk, missing CUDA, or missing dependency matrix).

## Preconditions
- Use a supported host with visible CUDA and the required native Python dependency matrix.
- Keep execution inside Modly surfaces: Electron owns setup/workflows/live execution, FastAPI only reports backend readiness.
- Do not enable headless mode; this extension declares `headless_eligible=false`.

## Workflow Check
1. Import one `.glb`, `.obj`, `.stl`, or `.ply` mesh into Modly.
2. Invoke `decompose-mesh` with no explicit `pipeline_stage` and confirm it resolves to `p3-sam`.
3. Confirm setup readiness blocks unsupported Linux ARM64 or missing CUDA/dependency matrices before execution.
4. Run the Electron-managed adapter and inspect the output bundle only when preflight is green:
   - one primary mesh output
   - `parts/`
   - `segmentation.json`
   - `bboxes.json`
   - optional `completion.json`
   - `bundle_manifest.json`
5. Feed the primary mesh only into UniRig and confirm downstream routing does not require multi-mesh awareness.
6. Continue from UniRig into Kimodo and confirm sidecars are ignored safely unless a future workflow consumes them explicitly.

## Expected Non-Goals
- No X-Part or `full` stage execution in MVP.
- No multi-mesh primary routing.
- No build/install/release mutation from the local runtime repo.
- No claim of successful inference without a live Electron-managed adapter.
- No attempt to run Electron-owned setup/workflows from an unsupported headless path.

## Deferred Scope
- Add a real X-Part adapter boundary and `full` stage behavior.
- Decide whether the primary mesh can change topology or must remain topology-preserving.
- Revisit bundle-first routing only after UniRig/Kimodo compatibility is proven.
