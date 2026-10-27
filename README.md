# Bounded native CPU validation for speech PRs

This branch only runs the unmodified native test module and the three previously failing ComfyUI workflow suites at the exact published source commits:

- 8883: `91d733385e8b7acf9808888c87ee84174763208c`
- 8884: `685e9893c9328f406976ca2606837e009cf96e49`
- vLLM CPU wheel: `0.32.0+cpu`, source `5df63a7203cdb9328d824a27b682e77c7b9928be`

Ubuntu 24.04, Python 3.12, 35-minute maximum job. No models, GPU/NPU devices, repository secrets, PR writes, or production deployment. Test files and assertions are not patched. Existing repository fixtures remain enabled. Model network access is disabled during testing; download-dependent cases must be reported, not silently reclassified as passed.

Logs, source identities, installed versions and JUnit are retained as an Actions artifact even on failure. This fork CPU run does not substitute for upstream CUDA L2 admission or maintainer approval.

## Verified outcome — 2026-10-27

Run [40512882950](https://github.com/Liuchenbing-2026/vllm-omni/actions/runs/40512882950), validation code commit `0bf5ce3ee77f1e1871ed0cdb71fbd6ae66988682`, passed. JUnit contains no failures/errors/skips. Both source working trees remained clean.

| Source | Speech module | Affected ComfyUI files | Total |
|---|---:|---:|---:|
| 8883 / 91d733385 | 388 | 19 | 407 |
| 8884 / 685e9893 | 386 | 19 | 405 |

Counts represent two branch executions (812 passes), not 812 unique cases. They include all four streaming regression variants and both timestamp variants.

Native tests use the repository's own mocks/fixtures. They do not perform real model generation, audio-quality assessment, accelerator execution, or end-to-end HTTP serving. `PYTHONPATH` and the current directory select each corresponding source tree; the editable distribution metadata was built once from 8883, so both logs contain its generated distribution version. The shallow checkout has no version tags, yielding an Omni `0.1.dev1` version warning. The vLLM wheel commit matches the source Dockerfile.ci pin; warnings are preserved and this does not establish complete runtime compatibility.

First run [40511553562](https://github.com/Liuchenbing-2026/vllm-omni/actions/runs/40511553562) failed before collecting tests: missing pytest-xdist meant `--dist=loadgroup` was unrecognized. Only the runner dependency was corrected to pytest-xdist 3.8.0; original failure logs are retained. No product or assertion changes were made in this validation round.

At 2026-10-27 13:39 UTC both published heads had green visible GitHub checks/docs, no current-head Buildkite result, and approval only at old commits. Upstream current-tip CUDA L2 and human review remain pending. The older CUDA/AMD failures concerned ComfyUI timeline schema; both heads contain upstream fix 6a798699d4af4bf56b755e7c7ed8ae3304742659, and the affected whole test files passed here.

Intel old build 10757 failed during Z-Image LoRA warmup with UR_RESULT_ERROR_OUT_OF_RESOURCES and ultimately timed out; exact resource/driver/kernel cause needs runner diagnostics. Build 10758 completed its four pytest groups (106/28/3/2 passed), then exited 125 around its 30-minute container budget; the precise exit source is unresolved. Neither is proven to be caused by the speech guard, and neither is declared fixed.

### Reproduction and evidence

The workflow defines the full install commands, exact public source commits, CPU wheel URL, pytest dependencies, runtime flags and test commands. `run-native.sh` is the execution entry point from the workflow workspace. No container image was used (digest not applicable); the hosted image is ubuntu-24.04, runtime and complete installed versions are in each evidence directory. See `requirements.lock` for actual installed versions, including the editable path to the source checkout. For offline interpretation no test rerun is necessary. Rebuilding the hosted environment was not independently tested.

`evidence/40512882950/` holds original logs, JUnit, source identities, exit codes, clean-tree records and dependency lock; `evidence/40511553562/` retains the failed attempt. `evidence/sha256.json` binds the archived evidence bytes. GitHub Actions artifacts have 14-day retention; these Git copies preserve the selected evidence beyond that period. No raw chats, private host records, signed artifact URLs or Buildkite HTML are included.
