"""
Tests for the connection-scoped execution profile policy stack (async end).

Async mirror of `test_profile.py`: every acceptance behaviour must hold on
`AsyncConnection`/`AsyncCursor`/`AsyncTransaction`/`AsyncPipeline` too, in
addition to the sync side.
"""

# Copyright (C) 2026 The Psycopg Team

from __future__ import annotations

import asyncio

import pytest

import psycopg
from psycopg import AsyncClientCursor, AsyncPipeline
from psycopg import errors as e
from psycopg import pq
from psycopg import (
    AutocommitOverlay,
    CancelSurface,
    ExecProfile,
    PipelineAdmit,
    PrepareBind,
)

from ._test_profile import TABLE

IDLE = pq.TransactionStatus.IDLE
ACTIVE = pq.TransactionStatus.ACTIVE
INTRANS = pq.TransactionStatus.INTRANS


@pytest.fixture
async def table(aconn):
    await aconn.execute(f"drop table if exists {TABLE}")
    await aconn.execute(f"create table {TABLE} (name text, i int)")
    await aconn.commit()
    yield TABLE
    await aconn.execute(f"drop table if exists {TABLE}")
    await aconn.commit()


def observer(dsn):
    return psycopg.AsyncConnection.connect(dsn, autocommit=True)


def autocommit_conn(dsn):
    return psycopg.AsyncConnection.connect(dsn, autocommit=True)


async def rows(obs, where=""):
    cur = await obs.execute(f"select count(*) from {TABLE} {where}")
    return (await cur.fetchone())[0]


# --- stack mechanics --------------------------------------------------------


async def test_empty_stack_is_pin(aconn):
    assert aconn.profiles.depth == 0
    assert not aconn.profiles
    assert aconn.profiles.top is None
    assert aconn.profiles.autocommit_overlay is AutocommitOverlay.PIN
    assert aconn.profiles.pipeline_admit is PipelineAdmit.PIN
    assert aconn.profiles.prepare_bind is PrepareBind.PIN
    assert aconn.profiles.cancel_surface is CancelSurface.PIN


async def test_string_values_normalised(aconn):
    async with aconn.exec_profile(
        autocommit_overlay="force_on",
        pipeline_admit="reject_enter",
        prepare_bind="forbid_prepare",
        cancel_surface="raise_on_cancel",
    ) as prof:
        assert prof.autocommit_overlay is AutocommitOverlay.FORCE_ON
        assert prof.pipeline_admit is PipelineAdmit.REJECT_ENTER
        assert prof.prepare_bind is PrepareBind.FORBID_PREPARE
        assert prof.cancel_surface is CancelSurface.RAISE_ON_CANCEL
    assert aconn.profiles.depth == 0


async def test_context_pushes_and_pops(aconn):
    profile = ExecProfile(prepare_bind=PrepareBind.FORCE_PREPARE)
    async with aconn.exec_profile(profile) as frame:
        assert frame is profile
        assert aconn.profiles.depth == 1
        assert aconn.profiles.top is profile
    assert aconn.profiles.depth == 0


async def test_exception_path_pops_clean(aconn):
    with pytest.raises(ZeroDivisionError):
        async with aconn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_OFF):
            raise ZeroDivisionError
    assert aconn.profiles.depth == 0


async def test_nested_frames_restore_outer(aconn):
    async with aconn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_OFF):
        async with aconn.exec_profile(prepare_bind=PrepareBind.FORBID_PREPARE):
            assert aconn.profiles.depth == 2
            assert aconn.profiles.autocommit_overlay is AutocommitOverlay.FORCE_OFF
            assert aconn.profiles.prepare_bind is PrepareBind.FORBID_PREPARE
        assert aconn.profiles.depth == 1
        assert aconn.profiles.prepare_bind is PrepareBind.PIN
    assert aconn.profiles.depth == 0


async def test_explicit_push_pop_and_out_of_order(aconn):
    outer = aconn.push_exec_profile(
        ExecProfile(autocommit_overlay=AutocommitOverlay.FORCE_OFF)
    )
    inner = aconn.push_exec_profile(ExecProfile(prepare_bind=PrepareBind.FORCE_PREPARE))
    with pytest.raises(e.ProgrammingError):
        aconn.pop_exec_profile(outer)
    aconn.pop_exec_profile(inner)
    aconn.pop_exec_profile(outer)
    assert aconn.profiles.depth == 0
    with pytest.raises(e.ProgrammingError):
        aconn.pop_exec_profile(outer)


# --- autocommit_overlay -----------------------------------------------------


async def test_force_on_does_not_touch_autocommit_attr(aconn, table):
    assert aconn.autocommit is False
    async with aconn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_ON):
        await aconn.execute(f"insert into {TABLE} values ('a', 1)")
        assert aconn.autocommit is False
        assert aconn.info.transaction_status == IDLE
    assert aconn.autocommit is False


async def test_force_on_autocommits_visible_from_other_connection(aconn, table, dsn):
    async with aconn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_ON):
        await aconn.execute(f"insert into {TABLE} values ('a', 1)")
        async with await observer(dsn) as obs:
            assert await rows(obs) == 1


async def test_force_on_both_connection_and_cursor_execute(aconn, table, dsn):
    async with aconn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_ON):
        await aconn.execute(f"insert into {TABLE} values ('a', 1)")
        cur = aconn.cursor()
        await cur.execute(f"insert into {TABLE} values ('b', 2)")
    async with await observer(dsn) as obs:
        assert await rows(obs) == 2


async def test_force_on_push_refused_with_implicit_transaction_open(aconn):
    await aconn.execute("select 1")  # opens the implicit BEGIN
    assert aconn.info.transaction_status == INTRANS
    with pytest.raises(e.ProgrammingError):
        aconn.push_exec_profile(
            ExecProfile(autocommit_overlay=AutocommitOverlay.FORCE_ON)
        )
    assert aconn.profiles.depth == 0
    await aconn.rollback()


async def test_force_on_push_refused_inside_transaction_and_savepoint(aconn):
    async with aconn.transaction():
        with pytest.raises(e.ProgrammingError):
            aconn.push_exec_profile(
                ExecProfile(autocommit_overlay=AutocommitOverlay.FORCE_ON)
            )
        async with aconn.transaction(savepoint_name="sp"):
            with pytest.raises(e.ProgrammingError):
                aconn.push_exec_profile(
                    ExecProfile(autocommit_overlay=AutocommitOverlay.FORCE_ON)
                )
    assert aconn.profiles.depth == 0


async def test_transaction_enter_refused_under_force_on(aconn):
    async with aconn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_ON):
        with pytest.raises(e.ProgrammingError):
            async with aconn.transaction():
                pass
        with pytest.raises(e.ProgrammingError):
            async with aconn.transaction(savepoint_name="sp"):
                pass


async def test_force_off_pending_until_explicit_commit(dsn, table):
    conn = await autocommit_conn(dsn)
    try:
        async with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_OFF):
            await conn.execute(f"insert into {TABLE} values ('a', 1)")
            assert conn.info.transaction_status == INTRANS
            async with await observer(dsn) as obs:
                assert await rows(obs) == 0
        async with await observer(dsn) as obs:
            assert await rows(obs) == 0
        await conn.commit()
        async with await observer(dsn) as obs:
            assert await rows(obs) == 1
    finally:
        await conn.close()


async def test_force_off_rollback_discards(dsn, table):
    conn = await autocommit_conn(dsn)
    try:
        async with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_OFF):
            await conn.execute(f"insert into {TABLE} values ('a', 1)")
        await conn.rollback()
        async with await observer(dsn) as obs:
            assert await rows(obs) == 0
    finally:
        await conn.close()


async def test_force_off_explicit_transaction_commits(dsn, table):
    conn = await autocommit_conn(dsn)
    try:
        async with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_OFF):
            async with conn.transaction():
                await conn.execute(f"insert into {TABLE} values ('a', 1)")
        async with await observer(dsn) as obs:
            assert await rows(obs) == 1
    finally:
        await conn.close()


async def test_pin_keeps_flat_autocommit(dsn, table):
    conn = await autocommit_conn(dsn)
    try:
        async with conn.exec_profile():
            await conn.execute(f"insert into {TABLE} values ('a', 1)")
        async with await observer(dsn) as obs:
            assert await rows(obs) == 1
    finally:
        await conn.close()


# --- pipeline_admit ---------------------------------------------------------


async def test_reject_enter_pipeline_does_not_enqueue(aconn):
    async with aconn.exec_profile(pipeline_admit=PipelineAdmit.REJECT_ENTER):
        with pytest.raises(e.ProgrammingError):
            async with aconn.pipeline():
                pass
        assert aconn._pipeline is None
        assert aconn.pgconn.pipeline_status == 0


async def test_reject_enter_direct_pipeline_object(aconn):
    p = AsyncPipeline(aconn)
    async with aconn.exec_profile(pipeline_admit=PipelineAdmit.REJECT_ENTER):
        with pytest.raises(e.ProgrammingError):
            async with p:
                pass
    assert aconn._pipeline is None
    assert aconn.pgconn.pipeline_status == 0


async def test_reject_enter_does_not_block_transaction(aconn):
    async with aconn.exec_profile(pipeline_admit=PipelineAdmit.REJECT_ENTER):
        async with aconn.transaction():
            await aconn.execute("select 1")


async def test_force_sync_on_exit_drains_and_marks(aconn):
    async with aconn.exec_profile(pipeline_admit=PipelineAdmit.FORCE_SYNC_ON_EXIT):
        async with aconn.pipeline() as p:
            assert p._profile_force_sync is True
            await aconn.execute("select 1")
        assert p._profile_force_synced is True
        assert not p.command_queue
        assert not p.result_queue


async def test_pin_pipeline_behaviour_unchanged(aconn):
    async with aconn.pipeline() as p:
        assert p._profile_force_sync is False
        await aconn.execute("select 1")
    assert p._profile_force_synced is False


# --- prepare_bind -----------------------------------------------------------


async def test_force_prepare_uses_named_statement(aconn, dsn):
    aconn.prepare_threshold = None
    async with aconn.exec_profile(prepare_bind=PrepareBind.FORCE_PREPARE):
        await aconn.execute("select %s::int", (1,))
        await aconn.execute("select %s::int", (2,))
        cur = aconn.cursor()
        await cur.execute("select %s::int", (3,))
    key = (b"select $1::int", (21,))
    assert aconn._prepared._names.get(key) == b"_pg3_0"
    assert aconn._prepared._prepared_idx == 1


async def test_forbid_prepare_never_hits_or_creates_cache(aconn):
    aconn.prepare_threshold = 0
    async with aconn.exec_profile(prepare_bind=PrepareBind.FORBID_PREPARE):
        for i in range(4):
            await aconn.execute("select %s::int", (i,))
            await aconn.execute("select %s::int", (i,), prepare=True)
    assert not aconn._prepared._names
    assert not aconn._prepared._counts
    assert aconn._prepared._prepared_idx == 0


async def test_client_cursor_prepare_axes(dsn, anyio_backend):
    conn = await psycopg.AsyncConnection.connect(dsn, cursor_factory=AsyncClientCursor)
    conn.prepare_threshold = None
    try:
        async with conn.exec_profile(prepare_bind=PrepareBind.FORCE_PREPARE):
            await conn.execute("select 41 + 1")
            await conn.execute("select 41 + 1")
        assert b"_pg3_0" in conn._prepared._names.values()

        conn2 = await psycopg.AsyncConnection.connect(
            dsn, cursor_factory=AsyncClientCursor
        )
        conn2.prepare_threshold = 0
        try:
            async with conn2.exec_profile(prepare_bind=PrepareBind.FORBID_PREPARE):
                await conn2.execute("select 41 + 1")
                await conn2.execute("select 41 + 1")
            assert not conn2._prepared._names
            assert not conn2._prepared._counts
        finally:
            await conn2.close()
    finally:
        await conn.close()


async def test_server_cursor_force_on_conflicts(aconn):
    cur = aconn.cursor("a_srv_cur")
    try:
        async with aconn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_ON):
            with pytest.raises(psycopg.errors.Error):
                await cur.execute("select 1")
            assert not aconn.closed
            await aconn.execute("select 1")
    finally:
        await cur.close()


async def test_server_cursor_force_off_behaves_transactional(dsn, anyio_backend):
    conn = await autocommit_conn(dsn)
    try:
        async with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_OFF):
            async with conn.cursor("a_srv_cur2") as cur:
                await cur.execute("select 1 as x")
                assert (await cur.fetchone())[0] == 1
            assert conn.info.transaction_status == INTRANS
        await conn.commit()
        assert conn.info.transaction_status == IDLE
    finally:
        await conn.close()


# --- cancel_surface ---------------------------------------------------------


async def test_cancel_pin_raises_as_before(aconn):
    async def canceller():
        await asyncio.sleep(0.3)
        aconn.cancel()

    errors: list[Exception] = []

    async def run():
        try:
            await aconn.execute("select pg_sleep(2)")
        except Exception as exc:
            errors.append(exc)

    await asyncio.gather(run(), canceller())
    assert len(errors) == 1
    assert isinstance(errors[0], e.QueryCanceled)
    await aconn.rollback()


async def test_cancel_raise_on_cancel_real_cancel(aconn):
    async def canceller():
        await asyncio.sleep(0.3)
        aconn.cancel()

    errors: list[Exception] = []

    async def run():
        async with aconn.exec_profile(cancel_surface=CancelSurface.RAISE_ON_CANCEL):
            try:
                await aconn.execute("select pg_sleep(2)")
            except Exception as exc:
                errors.append(exc)

    await asyncio.gather(run(), canceller())
    assert len(errors) == 1
    assert isinstance(errors[0], e.QueryCanceled)
    await aconn.rollback()


async def test_cancel_raise_on_cancel_success_result(aconn, monkeypatch):
    # Deterministic cancel/result race seam: the server result is benign but
    # a cancel was dispatched after send; the frame raises QueryCanceled.
    monkeypatch.setattr(
        psycopg.AsyncCursor,
        "_cancel_dispatched_during_execute",
        lambda self: True,
    )
    async with aconn.exec_profile(cancel_surface=CancelSurface.RAISE_ON_CANCEL):
        with pytest.raises(e.QueryCanceled):
            await aconn.execute("select 1")
    await aconn.rollback()


async def test_executemany_raise_on_cancel_success_result(aconn, monkeypatch):
    monkeypatch.setattr(
        psycopg.AsyncCursor,
        "_cancel_dispatched_during_execute",
        lambda self: True,
    )
    async with aconn.exec_profile(cancel_surface=CancelSurface.RAISE_ON_CANCEL):
        with pytest.raises(e.QueryCanceled):
            await aconn.cursor().executemany("select 1", [(), (), (), ()])
    await aconn.rollback()


async def test_cancel_ignore_result_completes_without_closing(dsn, anyio_backend):
    conn = await autocommit_conn(dsn)

    async def canceller():
        await asyncio.sleep(0.3)
        conn.cancel()

    errors: list[Exception] = []
    try:

        async def run():
            async with conn.exec_profile(
                cancel_surface=CancelSurface.IGNORE_CANCEL_RESULT
            ):
                cur = await conn.execute("select pg_sleep(2)")
                assert not conn.closed
                return cur

        task = asyncio.ensure_future(canceller())
        cur = await run()
        await task
        cur1 = await conn.execute("select 1")
        assert (await cur1.fetchone())[0] == 1
        assert cur.pgresult is None
        assert cur.statusmessage is None
    except Exception as exc:  # pragma: no cover
        errors.append(exc)
    finally:
        await conn.close()
    assert not errors


async def test_ignore_cancel_does_not_swallow_unrelated_errors(aconn):
    async with aconn.exec_profile(cancel_surface=CancelSurface.IGNORE_CANCEL_RESULT):
        with pytest.raises(e.UndefinedTable):
            await aconn.execute("select * from no_such_profile_table_xyz")
    await aconn.rollback()


# --- late push boundaries ----------------------------------------------------


async def test_late_push_pipeline_admit_rejected(aconn):
    async with aconn.pipeline():
        with pytest.raises(e.ProgrammingError):
            aconn.push_exec_profile(
                ExecProfile(pipeline_admit=PipelineAdmit.REJECT_ENTER)
            )
        with pytest.raises(e.ProgrammingError):
            aconn.push_exec_profile(
                ExecProfile(pipeline_admit=PipelineAdmit.FORCE_SYNC_ON_EXIT)
            )
        with pytest.raises(e.ProgrammingError):
            aconn.push_exec_profile(
                ExecProfile(prepare_bind=PrepareBind.FORBID_PREPARE)
            )
    assert aconn.profiles.depth == 0


async def test_late_push_while_command_in_flight_rejected(aconn):
    seen: list[Exception] = []

    async def worker():
        try:
            await aconn.execute("select pg_sleep(1.5)")
        except Exception as exc:
            seen.append(exc)

    task = asyncio.ensure_future(worker())
    while aconn.pgconn.transaction_status != ACTIVE:
        await asyncio.sleep(0.005)
    try:
        with pytest.raises(e.ProgrammingError):
            aconn.push_exec_profile(
                ExecProfile(cancel_surface=CancelSurface.RAISE_ON_CANCEL)
            )
    finally:
        await task
    assert not seen
    await aconn.rollback()


# --- connection isolation ----------------------------------------------------


async def test_two_connections_do_not_share_stack_in_one_task(dsn, anyio_backend):
    c1 = await psycopg.AsyncConnection.connect(dsn)
    c2 = await psycopg.AsyncConnection.connect(dsn)
    try:
        async with c1.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_ON):
            async with c2.exec_profile(pipeline_admit=PipelineAdmit.REJECT_ENTER):
                # Both driven by the same asyncio task: stacks still differ.
                assert c1.profiles is not c2.profiles
                assert c1.profiles.pipeline_admit is PipelineAdmit.PIN
                assert c2.profiles.autocommit_overlay is AutocommitOverlay.PIN

                async with c1.pipeline():
                    await c1.execute("select 1")
                assert c1.info.transaction_status == IDLE

                with pytest.raises(e.ProgrammingError):
                    async with c2.pipeline():
                        pass
    finally:
        await c1.close()
        await c2.close()


# --- Copy is explicitly out of the stack ------------------------------------


async def test_copy_does_not_read_autocommit_overlay(dsn, table):
    conn = await autocommit_conn(dsn)
    try:
        async with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_OFF):
            async with conn.cursor().copy(f"copy {TABLE} from stdin") as cp:
                await cp.write_row(("via_copy", "9"))
            assert conn.info.transaction_status == IDLE
        async with await observer(dsn) as obs:
            assert (await rows(obs, "where name = 'via_copy'")) == 1
    finally:
        await conn.close()
