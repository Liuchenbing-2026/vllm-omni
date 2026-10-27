# Bounded native CPU validation for speech PRs

This branch only runs the unmodified native test module and the three previously failing ComfyUI workflow suites at the exact published source commits:

- 8883: `91d733385e8b7acf9808888c87ee84174763208c`
- 8884: `685e9893c9328f406976ca2606837e009cf96e49`
- vLLM CPU wheel: `0.32.0+cpu`, source `5df63a7203cdb9328d824a27b682e77c7b9928be`

Ubuntu 24.04, Python 3.12, 35-minute maximum job. No models, GPU/NPU devices, repository secrets, PR writes, or production deployment. Test files and assertions are not patched. Existing repository fixtures remain enabled. Model network access is disabled during testing; download-dependent cases must be reported, not silently reclassified as passed.

Logs, source identities, installed versions and JUnit are retained as an Actions artifact even on failure. This fork CPU run does not substitute for upstream CUDA L2 admission or maintainer approval.
