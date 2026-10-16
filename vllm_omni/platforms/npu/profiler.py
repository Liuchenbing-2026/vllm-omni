# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import Any

from typing_extensions import override
from vllm.config import ProfilerConfig
from vllm.logger import init_logger

from vllm_omni.profiler.omni_torch_profiler import (
    OmniTorchProfilerWrapper,
    TorchProfilerActivity,
)

logger = init_logger(__name__)


class _ClosableRecordSchedule:
    """Schedule that records continuously until it is explicitly closed.

    ``torch_npu.profiler.profile`` falls back to its internal
    ``_default_schedule_fn`` when no ``schedule`` is supplied, and that
    function returns ``ProfilerAction.RECORD`` forever. ``stop()`` therefore
    always leaves through ``transit_action(RECORD, None)`` -- the profiler's
    "Incorrect schedule: Stop profiler while current state is RECORD" branch,
    whose trace export crashes the process on Ascend (issue #6620).

    This schedule keeps the documented "record until ``/stop_profile``"
    semantics (no ``active`` window is imposed) and only switches to
    ``RECORD_AND_SAVE`` once :meth:`close` has been called, so a single
    ``step()`` moves the state machine onto the regular exit branch.

    Only ``_closed`` lives in ``__dict__`` because torch_npu serialises the
    schedule object into ``profiler_info.json`` via ``__dict__``.
    """

    def __init__(self) -> None:
        self._closed = False

    def close(self) -> None:
        """Leave the recording window on the next ``step()``."""
        self._closed = True

    def __call__(self, step: int) -> Any:
        from torch_npu.profiler import ProfilerAction

        if self._closed:
            return ProfilerAction.RECORD_AND_SAVE
        return ProfilerAction.RECORD


class NPUTorchProfilerWrapper(OmniTorchProfilerWrapper):
    """NPU-specific profiler using torch_npu.profiler.

    Key differences from base profiler:
    - Uses torch_npu.profiler instead of torch.profiler
    - Different experimental_config options (AiCMetrics, profiler_level, etc.)
    - Uses tensorboard_trace_handler for trace output
    - No key_averages() support - requires offline parsing with
      torch_npu.profiler.profiler.analyse()
    """

    def __init__(
        self,
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
        activities: list[TorchProfilerActivity] | None = None,
    ) -> None:
        # Number of profiling windows already started by this wrapper; used to
        # decide whether the torch_npu profiler instance has to be rebuilt.
        self._windows_started = 0
        super().__init__(profiler_config, worker_name, local_rank, activities)

    @override
    def _get_default_activities(self) -> list[TorchProfilerActivity]:
        return ["CPU", "NPU"]

    @override
    def _create_profiler(
        self,
        profiler_config: ProfilerConfig,
        activities: list[TorchProfilerActivity],
    ):
        import torch_npu

        # Map activity names to torch_npu profiler activities
        npu_activities = []
        for activity in activities:
            if activity == "CPU":
                npu_activities.append(torch_npu.profiler.ProfilerActivity.CPU)
            elif activity == "NPU":
                npu_activities.append(torch_npu.profiler.ProfilerActivity.NPU)

        # NPU-specific experimental config for detailed profiling
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        # Set up trace directory for NPU - tensorboard_trace_handler creates
        # its own subdirectory structure. Use worker_name which includes stage_id.
        npu_trace_dir = os.path.join(self._trace_dir, self._worker_name)
        os.makedirs(npu_trace_dir, exist_ok=True)
        self._trace_path = npu_trace_dir

        # Never let torch_npu fall back to ``_default_schedule_fn``: that
        # schedule can only ever be stopped mid-RECORD. ``_stop`` closes this
        # schedule so the profiler leaves the capture window on a known
        # boundary before the trace is exported.
        self._schedule = _ClosableRecordSchedule()

        return torch_npu.profiler.profile(
            activities=npu_activities,
            schedule=self._schedule,
            with_stack=False,
            profile_memory=profiler_config.torch_profiler_with_memory,
            # Forward the shape-recording flag; without it the NPU profiler
            # silently ignores ``torch_profiler_record_shapes=True``.
            record_shapes=profiler_config.torch_profiler_record_shapes,
            # NOTE: torch_npu.profiler.with_modules is equivalent to
            # torch.profiler.with_stack. The with_stack option in
            # torch_npu.profiler introduces significant time overhead.
            with_modules=profiler_config.torch_profiler_with_stack,
            experimental_config=experimental_config,
            # Use tensorboard_trace_handler directly - NPU profiler expects
            # this specific handler format
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(npu_trace_dir),
        )

    @override
    def _start(self) -> None:
        # ``torch_npu.profiler.profile`` cannot be restarted once stopped: its
        # action state machine keeps the terminal action and the profiling
        # directory is released, so a second window on the same instance emits
        # no trace. Rebuild profiler and schedule per window instead, mirroring
        # vLLM's own ``TorchProfilerWrapper``. The instance created by
        # ``__init__`` is still fresh, so it is reused for the first window and
        # never discarded unstarted.
        if self._windows_started:
            self.profiler = self._create_profiler(self.profiler_config, self._activities)
        self._windows_started += 1
        super()._start()

    @override
    def _stop(self) -> None:
        """Close the recording window before stopping (issue #6620).

        ``stop_profile`` is an out-of-band call, so without this the profiler
        is always stopped mid-RECORD on the default configuration and the NPU
        trace export runs on an inconsistent capture state.
        """
        self._schedule.close()
        self.profiler.step()
        super()._stop()

    @override
    def _on_stop_hook(self) -> None:
        """NPU profiler doesn't support key_averages() - log offline parsing hint."""
        if self.local_rank == 0:
            logger.info(
                "NPU profiler stopped. Use offline parsing to analyze: "
                "from torch_npu.profiler.profiler import analyse; "
                "analyse('%s')",
                self._trace_path or self._trace_dir,
            )
