# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import os
import time
from argparse import Namespace
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI

from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationStatus, VideoResponse
from vllm_omni.entrypoints.openai.storage import LocalStorageManager, LocalStorageTTLManager
from vllm_omni.entrypoints.openai.stores import AsyncDictStore

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Socket:
    def close(self) -> None:
        pass


@pytest.mark.parametrize("serve_error", [False, True])
def test_worker_expires_metadata_and_stops_storage(tmp_path, monkeypatch, serve_error):
    storage = LocalStorageTTLManager(storage_path=str(tmp_path), ttl_seconds=1, sweep_interval_seconds=60)
    store: AsyncDictStore[VideoResponse] = AsyncDictStore()
    app = FastAPI()

    @asynccontextmanager
    async def build_engine(*_args, **_kwargs):
        yield object()

    async def init_app_state(engine, state, _args):
        state.engine_client = engine

    async def get_config(_engine):
        return None

    async def serve_http(_app, **_kwargs):
        if serve_error:
            raise RuntimeError("serve_http failed")
        await storage._sweep_once(time.time() - 1)
        assert not storage.exists("expired")
        assert await store.get("expired") is None
        assert [job.id for job in await store.list_values()] == ["fresh"]
        return asyncio.create_task(asyncio.sleep(0))

    monkeypatch.setattr(api_server, "build_async_omni", build_engine)
    monkeypatch.setattr(api_server, "build_openai_app", lambda *_args: app)
    monkeypatch.setattr(api_server, "omni_init_app_state", init_app_state)
    monkeypatch.setattr(api_server, "_register_omni_exception_handlers", lambda _app: None)
    monkeypatch.setattr(api_server, "_should_enable_profiler_endpoints", lambda _stages: False)
    monkeypatch.setattr(api_server, "get_uvicorn_log_config", lambda _args: None)
    monkeypatch.setattr(api_server, "_limit_api_server_torch_threads", lambda: None)
    monkeypatch.setattr(api_server.openai_app_state, "_get_vllm_config", get_config)
    monkeypatch.setattr(api_server, "STORAGE_MANAGER", storage)
    monkeypatch.setattr(api_server, "VIDEO_STORE", store)
    monkeypatch.setattr(api_server, "serve_http", serve_http)
    args = Namespace(
        tool_parser_plugin=None,
        reasoning_parser_plugin=None,
        enable_ssl_refresh=False,
        host="127.0.0.1",
        port=8000,
        uvicorn_log_level="info",
        disable_uvicorn_access_log=True,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_ca_certs=None,
        ssl_cert_reqs=0,
        ssl_ciphers=None,
        h11_max_incomplete_event_size=None,
        h11_max_header_count=None,
    )

    async def run():
        for key in ("expired", "fresh"):
            await storage.save(b"video-bytes", key)
            await store.upsert(
                key,
                VideoResponse(id=key, model="test", prompt="test", status=VideoGenerationStatus.COMPLETED),
            )
        os.utime(tmp_path / "expired", (time.time() - 10,) * 2)
        try:
            if serve_error:
                with pytest.raises(RuntimeError, match="serve_http failed"):
                    await api_server.omni_run_server_worker("127.0.0.1:8000", _Socket(), args)
            else:
                await api_server.omni_run_server_worker("127.0.0.1:8000", _Socket(), args)
            assert storage._sweeper_task is None
        finally:
            await storage.stop()

    asyncio.run(run())


@pytest.mark.parametrize("delete_error", [False, True])
def test_ttl_callback_only_receives_successfully_deleted_keys(tmp_path, monkeypatch, delete_error):
    storage = LocalStorageTTLManager(storage_path=str(tmp_path), ttl_seconds=1, sweep_interval_seconds=60)
    batches: list[list[str]] = []

    async def on_expire(keys: list[str]) -> None:
        batches.append(keys)

    original_remove = os.remove

    def remove(path):
        if delete_error and os.path.basename(path) == "expired":
            raise PermissionError("file is busy")
        original_remove(path)

    async def run():
        await storage.start(expiration_callback=on_expire)
        try:
            await storage.save(b"video", "expired")
            await storage.save(b"video", "fresh")
            os.utime(tmp_path / "expired", (time.time() - 10,) * 2)
            monkeypatch.setattr(os, "remove", remove)
            assert await storage._sweep_once(time.time() - 1) == (0 if delete_error else 1)
            assert batches == ([] if delete_error else [["expired"]])
            assert storage.exists("fresh")
            assert storage.exists("expired") is delete_error
        finally:
            await storage.stop()

    asyncio.run(run())


def test_ttl_callback_rebinds_on_restart(tmp_path):
    storage = LocalStorageTTLManager(storage_path=str(tmp_path), ttl_seconds=1, sweep_interval_seconds=60)
    callbacks: list[str] = []

    async def first(_keys: list[str]) -> None:
        callbacks.append("first")

    async def second(_keys: list[str]) -> None:
        callbacks.append("second")

    async def run():
        for callback in (first, second):
            await storage.start(expiration_callback=callback)
            try:
                await storage.save(b"video", "expired")
                os.utime(tmp_path / "expired", (time.time() - 10,) * 2)
                assert await storage._sweep_once(time.time() - 1) == 1
            finally:
                await storage.stop()
        assert callbacks == ["first", "second"]

    asyncio.run(run())


@pytest.mark.parametrize("keys", [[], ["expired", "missing", "expired"]])
def test_store_pop_many_preserves_other_jobs(keys):
    store: AsyncDictStore[VideoResponse] = AsyncDictStore()

    async def run():
        for key in ("expired", "fresh"):
            await store.upsert(key, VideoResponse(id=key, model="test", prompt="test"))
        await store.pop_many(keys)
        assert await store.get("fresh") is not None
        assert (await store.get("expired") is None) is bool(keys)

    asyncio.run(run())


def test_storage_without_ttl_accepts_worker_callback(tmp_path):
    storage = LocalStorageManager(storage_path=str(tmp_path))
    store: AsyncDictStore[VideoResponse] = AsyncDictStore()

    async def run():
        await storage.start(expiration_callback=store.pop_many)
        try:
            await storage.save(b"video", "fresh")
            assert await storage.open("fresh") is not None
        finally:
            await storage.stop()

    asyncio.run(run())
