# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for StagePool.collective_rpc EngineCore control dispatch."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from vllm.lora.request import LoRARequest

from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model]


def _make_pool(*, stage_type: str = "llm", **client_methods: AsyncMock) -> tuple[StagePool, SimpleNamespace]:
    client = SimpleNamespace(stage_type=stage_type, **client_methods)
    if "collective_rpc_async" not in client_methods:
        client.collective_rpc_async = AsyncMock(return_value={"via": "collective"})
    pool = StagePool(0, [client])  # type: ignore[arg-type]
    return pool, client


@pytest.mark.cpu
def test_collective_rpc_normalizes_none_args_on_control_helper():
    async def run() -> None:
        pause = AsyncMock(return_value="paused")
        pool, client = _make_pool(pause_scheduler_async=pause)

        result = await pool.collective_rpc(0, "pause_scheduler", args=None, kwargs={"mode": "abort"})

        assert result == "paused"
        pause.assert_awaited_once_with(mode="abort")
        client.collective_rpc_async.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_unrelated_async_helper_uses_collective_path():
    async def run() -> None:
        other = AsyncMock(return_value="should-not-run")
        pool, client = _make_pool(is_sleeping_async=other)

        result = await pool.collective_rpc(0, "is_sleeping", timeout=1.5, args=("x",), kwargs={"k": 1})

        assert result == {"via": "collective"}
        other.assert_not_awaited()
        client.collective_rpc_async.assert_awaited_once_with(
            method="is_sleeping",
            timeout=1.5,
            args=("x",),
            kwargs={"k": 1},
        )

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_control_helper_honors_timeout():
    async def run() -> None:
        async def slow_sleep(*_args, **_kwargs):
            await asyncio.sleep(1.0)
            return "slept"

        pool, _client = _make_pool(sleep_async=AsyncMock(side_effect=slow_sleep))
        with pytest.raises(asyncio.TimeoutError):
            await pool.collective_rpc(0, "sleep", timeout=0.01, args=(1,))

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_control_method_reraises_worker_error():
    async def run() -> None:
        pool, _client = _make_pool(wake_up_async=AsyncMock(side_effect=RuntimeError("worker died")))
        with pytest.raises(RuntimeError, match="worker died"):
            await pool.collective_rpc(0, "wake_up")

    asyncio.run(run())


@pytest.mark.cpu
def test_kv_memory_release_uses_engine_core_and_propagates_pause_errors():
    async def run() -> None:
        release = AsyncMock(return_value=None)
        pool, client = _make_pool(release_kv_cache_memory_async=release)
        await pool.collective_rpc(0, "release_kv_cache_memory", timeout=2.0)
        release.assert_awaited_once_with()
        client.collective_rpc_async.assert_not_awaited()

        release.side_effect = RuntimeError("requires a completed pause first")
        with pytest.raises(RuntimeError, match="completed pause"):
            await pool.collective_rpc(0, "release_kv_cache_memory")

        del client.release_kv_cache_memory_async
        result = await pool.collective_rpc(0, "release_kv_cache_memory")
        assert result["supported"] is False
        client.collective_rpc_async.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_reset_caches_use_control_helpers():
    async def run() -> None:
        reset_prefix = AsyncMock(return_value=True)
        reset_encoder = AsyncMock(return_value=None)
        reset_mm = AsyncMock(return_value=None)
        pool, client = _make_pool(
            reset_prefix_cache_async=reset_prefix,
            reset_encoder_cache_async=reset_encoder,
            reset_mm_cache_async=reset_mm,
        )

        result = await pool.collective_rpc(0, "reset_prefix_cache", args=(True, True), timeout=2.0)
        assert result is True
        reset_prefix.assert_awaited_once_with(True, True)

        await pool.collective_rpc(0, "reset_encoder_cache", timeout=2.0)
        reset_encoder.assert_awaited_once()

        await pool.collective_rpc(0, "reset_mm_cache", timeout=2.0)
        reset_mm.assert_awaited_once()

        client.collective_rpc_async.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_non_control_method_still_returns_error_dict():
    async def run() -> None:
        pool, _client = _make_pool(collective_rpc_async=AsyncMock(side_effect=RuntimeError("probe failed")))
        result = await pool.collective_rpc(0, "is_sleeping")
        assert result["supported"] is False
        assert "probe failed" in result["error"]

    asyncio.run(run())


@pytest.mark.cpu
def test_abort_requests_does_not_commit_op_state_when_engine_abort_fails():
    async def run() -> None:
        class RecordingOutputProcessor:
            def __init__(self) -> None:
                self.collected = False
                self.committed = False

            def abort_requests_collecting_outputs(self, request_ids, *, internal=False, commit_state=True):
                del internal
                self.collected = True
                assert commit_state is False
                return list(request_ids), [SimpleNamespace(request_id=request_ids[0])]

            def commit_aborted_request_state(self, request_ids, *, internal=False):
                del request_ids, internal
                self.committed = True

        abort = AsyncMock(side_effect=RuntimeError("engine abort failed"))
        output_processor = RecordingOutputProcessor()
        client = SimpleNamespace(stage_type="llm", abort_requests_async=abort)
        pool = StagePool(0, [client], output_processor=output_processor)  # type: ignore[arg-type]
        pool._request_bindings["req-1"] = 0

        with pytest.raises(RuntimeError, match="engine abort failed"):
            await pool.abort_requests(["req-1"])
        assert output_processor.collected is True
        assert output_processor.committed is False
        abort.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.cpu
def test_release_request_resources_skips_without_async_chunk():
    async def run() -> None:
        call = AsyncMock()
        pool = StagePool(
            0,
            [SimpleNamespace(call_utility_async=call)],  # type: ignore[list-item]
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(async_chunk=False)),
        )

        pool.schedule_release_request_resources(["req-1"])
        await asyncio.sleep(0.01)
        call.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.parametrize("method", ["reset_prefix_cache", "reset_encoder_cache", "reset_mm_cache"])
@pytest.mark.parametrize("failure", ["error", "timeout", "missing"])
def test_cache_reset_failure_is_serialized_and_pool_remains_usable(method, failure):
    async def run():
        async def reset():
            if failure == "timeout":
                await asyncio.Event().wait()
            raise RuntimeError("cache reset failed")

        pool, client = _make_pool()
        if failure != "missing":
            setattr(client, f"{method}_async", AsyncMock(side_effect=reset))
        result = await pool.collective_rpc(0, method, timeout=0.01)
        assert result["supported"] is False
        assert result["error"]
        if failure == "timeout":
            assert "timed out" in result["error"]
        client.collective_rpc_async.assert_not_awaited()
        # A subsequent control request still completes on the same pool.
        assert await pool.collective_rpc(0, "is_sleeping") == {"via": "collective"}

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_add_lora_uses_typed_engine_helper():
    async def run() -> None:
        request = LoRARequest(lora_name="adapter", lora_int_id=1, lora_path="/unused/adapter")
        add_lora = AsyncMock(return_value=True)
        pool, client = _make_pool(add_lora_async=add_lora)
        assert await pool.collective_rpc(0, "add_lora", args=(request,)) is True
        add_lora.assert_awaited_once_with(request)
        client.collective_rpc_async.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_add_lora_missing_helper_does_not_fall_back():
    async def run() -> None:
        pool, client = _make_pool()
        result = await pool.collective_rpc(0, "add_lora", args=("unused",))
        assert result["supported"] is False
        assert "add_lora_async" in result["error"]
        client.collective_rpc_async.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_add_lora_preserves_failure_result():
    async def run() -> None:
        add_lora = AsyncMock(side_effect=RuntimeError("invalid adapter"))
        pool, client = _make_pool(add_lora_async=add_lora)
        result = await pool.collective_rpc(0, "add_lora", args=("unused",))
        assert result["supported"] is False
        assert result["error"] == "invalid adapter"
        client.collective_rpc_async.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_add_lora_honors_timeout_without_raising():
    async def slow_add(_request: object) -> bool:
        await asyncio.sleep(1.0)
        return True

    async def run() -> None:
        pool, client = _make_pool(add_lora_async=AsyncMock(side_effect=slow_add))
        result = await pool.collective_rpc(0, "add_lora", timeout=0.01, args=("unused",))
        assert result["supported"] is False
        assert "TimeoutError" in result["error"]
        client.collective_rpc_async.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_diffusion_add_lora_keeps_collective_path():
    async def run() -> None:
        add_lora = AsyncMock(return_value=True)
        pool, client = _make_pool(stage_type="diffusion", add_lora_async=add_lora)
        assert await pool.collective_rpc(0, "add_lora", args=("request",)) == {"via": "collective"}
        add_lora.assert_not_awaited()
        client.collective_rpc_async.assert_awaited_once_with(
            method="add_lora", timeout=None, args=("request",), kwargs={}
        )

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.parametrize("method,args", [("remove_lora", (1,)), ("pin_lora", (1,)), ("list_loras", ())])
def test_collective_rpc_other_lora_methods_keep_collective_path(method, args):
    async def run() -> None:
        helper = AsyncMock(return_value=True)
        pool, client = _make_pool(**{f"{method}_async": helper})
        assert await pool.collective_rpc(0, method, args=args) == {"via": "collective"}
        helper.assert_not_awaited()
        client.collective_rpc_async.assert_awaited_once_with(method=method, timeout=None, args=args, kwargs={})

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_add_lora_survives_real_utility_codec():
    from vllm.v1.engine.core import EngineCore, EngineCoreProc
    from vllm.v1.engine.core_client import AsyncMPClient
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, run_method

    class Worker:
        received = None

        def add_lora(self, request: LoRARequest) -> bool:
            self.received = request
            return request.lora_int_id > 0

    worker = Worker()

    class InProcessExecutor:
        add_lora = Executor.add_lora

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            return [run_method(worker, method, args, kwargs or {})]

    class UtilityRoundTripClient:
        # The process, ZMQ transport and model executor are the explicit test
        # boundaries; vLLM's utility client, codec and argument conversion run.
        stage_type = "llm"
        collective_rpc_async = AsyncMPClient.collective_rpc_async
        add_lora_async = AsyncMPClient.add_lora_async

        def __init__(self):
            self.encoder = MsgpackEncoder()
            self.decoder = MsgpackDecoder()
            self.core = EngineCore.__new__(EngineCore)
            self.core.model_executor = InProcessExecutor()

        async def call_utility_async(self, method, *args):
            _, _, decoded_method, decoded_args = self.decoder.decode(self.encoder.encode((0, 1, method, args)))
            target = getattr(self.core, decoded_method)
            return target(*EngineCoreProc._convert_msgspec_args(target, decoded_args))

    async def run() -> None:
        client = UtilityRoundTripClient()
        pool = StagePool(0, [client])  # type: ignore[arg-type]
        request = LoRARequest(lora_name="adapter", lora_int_id=7, lora_path="/unused/adapter")
        assert await pool.collective_rpc(0, "add_lora", args=(request,)) is True
        assert isinstance(worker.received, LoRARequest)
        assert worker.received.lora_int_id == 7

    asyncio.run(run())
