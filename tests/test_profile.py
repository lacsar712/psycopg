"""
Tests for the connection-scoped execution profile policy stack (sync end).

Covers, on `Connection`/`Cursor`/`Transaction`/`Pipeline`:

- push/pop, exception auto-pop, nesting, late-push boundary rejection;
- autocommit_overlay pin/force_on/force_off, incl. transaction/savepoint
  conflicts and observable cross-connection visibility;
- pipeline_admit pin/reject_enter/force_sync_on_exit;
- prepare_bind pin/force_prepare/forbid_prepare on Cursor, ClientCursor and
  server-side cursors;
- cancel_surface pin/raise_on_cancel/ignore_cancel_result after a real and a
  deterministically simulated cancel/result race;
- two connections driven from the same thread never sharing a stack top;
- Copy explicitly not reading the stack.
"""

# Copyright (C) 2026 The Psycopg Team

import threading

import pytest

import psycopg
from psycopg import ClientCursor, ExecProfile, Pipeline
from psycopg import errors as e
from psycopg import pq
from psycopg import (
    AutocommitOverlay,
    CancelSurface,
    PipelineAdmit,
    PrepareBind,
)

from ._test_profile import TABLE, make_canceller, wait_active

IDLE = pq.TransactionStatus.IDLE
INTRANS = pq.TransactionStatus.INTRANS


@pytest.fixture
def table(conn):
    conn.execute(f"drop table if exists {TABLE}")
    conn.execute(f"create table {TABLE} (name text, i int)")
    conn.commit()
    yield TABLE
    conn.execute(f"drop table if exists {TABLE}")
    conn.commit()


def observer(dsn):
    return psycopg.connect(dsn, autocommit=True)


def autocommit_conn(dsn):
    return psycopg.connect(dsn, autocommit=True)


# --- stack mechanics --------------------------------------------------------


def test_empty_stack_is_pin(conn):
    assert conn.profiles.depth == 0
    assert not conn.profiles
    assert conn.profiles.top is None
    assert conn.profiles.frames == ()
    assert conn.profiles.autocommit_overlay is AutocommitOverlay.PIN
    assert conn.profiles.pipeline_admit is PipelineAdmit.PIN
    assert conn.profiles.prepare_bind is PrepareBind.PIN
    assert conn.profiles.cancel_surface is CancelSurface.PIN


def test_string_values_normalised(conn):
    with conn.exec_profile(
        autocommit_overlay="force_on",
        pipeline_admit="reject_enter",
        prepare_bind="forbid_prepare",
        cancel_surface="raise_on_cancel",
    ) as prof:
        assert prof.autocommit_overlay is AutocommitOverlay.FORCE_ON
        assert prof.pipeline_admit is PipelineAdmit.REJECT_ENTER
        assert prof.prepare_bind is PrepareBind.FORBID_PREPARE
        assert prof.cancel_surface is CancelSurface.RAISE_ON_CANCEL
    assert conn.profiles.depth == 0


def test_context_pushes_and_pops(conn):
    profile = ExecProfile(prepare_bind=PrepareBind.FORCE_PREPARE)
    with conn.exec_profile(profile) as frame:
        assert frame is profile
        assert conn.profiles.depth == 1
        assert conn.profiles.top is profile

    assert conn.profiles.depth == 0


def test_exception_path_pops_clean(conn):
    with pytest.raises(ZeroDivisionError):
        with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_OFF):
            raise ZeroDivisionError
    assert conn.profiles.depth == 0


def test_nested_frames_restore_outer(conn):
    with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_OFF):
        assert conn.profiles.autocommit_overlay is AutocommitOverlay.FORCE_OFF
        with conn.exec_profile(prepare_bind=PrepareBind.FORBID_PREPARE):
            assert conn.profiles.depth == 2
            assert conn.profiles.autocommit_overlay is AutocommitOverlay.FORCE_OFF
            assert conn.profiles.prepare_bind is PrepareBind.FORBID_PREPARE
        assert conn.profiles.depth == 1
        assert conn.profiles.prepare_bind is PrepareBind.PIN
    assert conn.profiles.depth == 0


def test_explicit_push_pop_and_out_of_order(conn):
    outer = conn.push_exec_profile(
        ExecProfile(autocommit_overlay=AutocommitOverlay.FORCE_OFF)
    )
    inner = conn.push_exec_profile(ExecProfile(prepare_bind=PrepareBind.FORCE_PREPARE))
    with pytest.raises(e.ProgrammingError):
        conn.pop_exec_profile(outer)
    conn.pop_exec_profile(inner)
    conn.pop_exec_profile(outer)
    assert conn.profiles.depth == 0
    with pytest.raises(e.ProgrammingError):
        conn.pop_exec_profile(outer)


# --- autocommit_overlay -----------------------------------------------------


def test_force_on_does_not_touch_autocommit_attr(conn, table):
    assert conn.autocommit is False
    with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_ON):
        conn.execute(f"insert into {TABLE} values ('a', 1)")
        assert conn.autocommit is False
        assert conn.info.transaction_status == IDLE
    assert conn.autocommit is False


def test_force_on_autocommits_visible_from_other_connection(conn, table, dsn):
    with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_ON):
        conn.execute(f"insert into {TABLE} values ('a', 1)")
        with observer(dsn) as obs:
            assert obs.execute(f"select count(*) from {TABLE}").fetchone()[0] == 1


def test_force_on_both_connection_and_cursor_execute(conn, table, dsn):
    with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_ON):
        conn.execute(f"insert into {TABLE} values ('a', 1)")
        cur = conn.cursor()
        cur.execute(f"insert into {TABLE} values ('b', 2)")
    with observer(dsn) as obs:
        assert obs.execute(f"select count(*) from {TABLE}").fetchone()[0] == 2


def test_force_on_push_refused_with_implicit_transaction_open(conn):
    conn.execute("select 1")  # opens the implicit BEGIN on a default conn
    assert conn.info.transaction_status == INTRANS
    with pytest.raises(e.ProgrammingError):
        conn.push_exec_profile(
            ExecProfile(autocommit_overlay=AutocommitOverlay.FORCE_ON)
        )
    assert conn.profiles.depth == 0
    conn.rollback()


def test_force_on_push_refused_inside_transaction_and_savepoint(conn):
    with conn.transaction():
        with pytest.raises(e.ProgrammingError):
            conn.push_exec_profile(
                ExecProfile(autocommit_overlay=AutocommitOverlay.FORCE_ON)
            )
        with conn.transaction(savepoint_name="sp"):
            with pytest.raises(e.ProgrammingError):
                conn.push_exec_profile(
                    ExecProfile(autocommit_overlay=AutocommitOverlay.FORCE_ON)
                )
    assert conn.profiles.depth == 0


def test_transaction_enter_refused_under_force_on(conn):
    with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_ON):
        with pytest.raises(e.ProgrammingError):
            with conn.transaction():
                pass
        # nested savepoint must be refused too
        with pytest.raises(e.ProgrammingError):
            with conn.transaction(savepoint_name="sp"):
                pass


def test_force_off_pending_until_explicit_commit(dsn, table):
    conn = autocommit_conn(dsn)
    try:
        with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_OFF):
            conn.execute(f"insert into {TABLE} values ('a', 1)")
            assert conn.info.transaction_status == INTRANS
            with observer(dsn) as obs:
                assert obs.execute(f"select count(*) from {TABLE}").fetchone()[0] == 0
        # popping the frame does not commit
        with observer(dsn) as obs:
            assert obs.execute(f"select count(*) from {TABLE}").fetchone()[0] == 0
        conn.commit()
        with observer(dsn) as obs:
            assert obs.execute(f"select count(*) from {TABLE}").fetchone()[0] == 1
    finally:
        conn.close()


def test_force_off_rollback_discards(dsn, table):
    conn = autocommit_conn(dsn)
    try:
        with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_OFF):
            conn.execute(f"insert into {TABLE} values ('a', 1)")
        conn.rollback()
        with observer(dsn) as obs:
            assert obs.execute(f"select count(*) from {TABLE}").fetchone()[0] == 0
    finally:
        conn.close()


def test_force_off_explicit_transaction_commits(dsn, table):
    conn = autocommit_conn(dsn)
    try:
        with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_OFF):
            with conn.transaction():
                conn.execute(f"insert into {TABLE} values ('a', 1)")
        with observer(dsn) as obs:
            assert obs.execute(f"select count(*) from {TABLE}").fetchone()[0] == 1
    finally:
        conn.close()


def test_pin_keeps_flat_autocommit(dsn, table):
    conn = autocommit_conn(dsn)
    try:
        with conn.exec_profile():  # all pin
            conn.execute(f"insert into {TABLE} values ('a', 1)")
        with observer(dsn) as obs:
            assert obs.execute(f"select count(*) from {TABLE}").fetchone()[0] == 1
    finally:
        conn.close()


# --- pipeline_admit ---------------------------------------------------------


def test_reject_enter_pipeline_does_not_enqueue(conn):
    with conn.exec_profile(pipeline_admit=PipelineAdmit.REJECT_ENTER):
        with pytest.raises(e.ProgrammingError):
            with conn.pipeline():
                pass
        assert conn._pipeline is None
        assert conn.pgconn.pipeline_status == 0


def test_reject_enter_direct_pipeline_object(conn):
    p = Pipeline(conn)
    with conn.exec_profile(pipeline_admit=PipelineAdmit.REJECT_ENTER):
        with pytest.raises(e.ProgrammingError):
            with p:
                pass
    assert conn._pipeline is None
    assert conn.pgconn.pipeline_status == 0


def test_reject_enter_does_not_block_transaction(conn):
    with conn.exec_profile(pipeline_admit=PipelineAdmit.REJECT_ENTER):
        with conn.transaction():
            conn.execute("select 1")


def test_force_sync_on_exit_drains_and_marks(conn):
    with conn.exec_profile(pipeline_admit=PipelineAdmit.FORCE_SYNC_ON_EXIT):
        with conn.pipeline() as p:
            assert p._profile_force_sync is True
            conn.execute("select 1")
        assert p._profile_force_synced is True
        assert not p.command_queue
        assert not p.result_queue


def test_pin_pipeline_behaviour_unchanged(conn):
    with conn.pipeline() as p:
        assert p._profile_force_sync is False
        conn.execute("select 1")
    assert p._profile_force_synced is False


# --- prepare_bind -----------------------------------------------------------


def test_force_prepare_uses_named_statement(conn, dsn):
    conn.prepare_threshold = None  # automatic preparing disabled
    with conn.exec_profile(prepare_bind=PrepareBind.FORCE_PREPARE):
        conn.execute("select %s::int", (1,))
        conn.execute("select %s::int", (2,))
        cur = conn.cursor()
        cur.execute("select %s::int", (3,))
    key = (b"select $1::int", (21,))
    assert conn._prepared._names.get(key) == b"_pg3_0"
    # same shape reused, no new statement allocated
    assert conn._prepared._prepared_idx == 1


def test_forbid_prepare_never_hits_or_creates_cache(conn):
    conn.prepare_threshold = 0  # every execute would normally prepare
    with conn.exec_profile(prepare_bind=PrepareBind.FORBID_PREPARE):
        for i in range(4):
            conn.execute("select %s::int", (i,))
            conn.execute("select %s::int", (i,), prepare=True)
    assert not conn._prepared._names
    assert not conn._prepared._counts
    assert conn._prepared._prepared_idx == 0


def test_client_cursor_prepare_axes(dsn):
    conn = psycopg.connect(dsn, cursor_factory=ClientCursor)
    conn.prepare_threshold = None
    try:
        with conn.exec_profile(prepare_bind=PrepareBind.FORCE_PREPARE):
            conn.execute("select 41 + 1")
            conn.execute("select 41 + 1")
        assert b"_pg3_0" in conn._prepared._names.values()

        conn2 = psycopg.connect(dsn, cursor_factory=ClientCursor)
        conn2.prepare_threshold = 0
        try:
            with conn2.exec_profile(prepare_bind=PrepareBind.FORBID_PREPARE):
                conn2.execute("select 41 + 1")
                conn2.execute("select 41 + 1")
            assert not conn2._prepared._names
            assert not conn2._prepared._counts
        finally:
            conn2.close()
    finally:
        conn.close()


def test_server_cursor_force_on_conflicts(conn):
    cur = conn.cursor("srv_cur")
    try:
        with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_ON):
            with pytest.raises(psycopg.errors.Error):
                cur.execute("select 1")
            # connection not closed and recoverable
            assert not conn.closed
            conn.execute("select 1")  # still autocommit-like in the frame
    finally:
        cur.close()


def test_server_cursor_force_off_behaves_transactional(dsn):
    conn = autocommit_conn(dsn)
    try:
        with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_OFF):
            with conn.cursor("srv_cur2") as cur:
                cur.execute("select 1 as x")
                assert cur.fetchone()[0] == 1
            assert conn.info.transaction_status == INTRANS
        conn.commit()
        assert conn.info.transaction_status == IDLE
    finally:
        conn.close()


# --- cancel_surface ---------------------------------------------------------


def test_cancel_pin_raises_as_before(conn):
    errors: list[Exception] = []
    t = make_canceller(conn, errors)
    t.start()
    with pytest.raises(e.QueryCanceled):
        conn.execute("select pg_sleep(2)")
    t.join()
    conn.rollback()
    assert not errors


def test_cancel_raise_on_cancel_real_cancel(conn):
    errors: list[Exception] = []
    t = make_canceller(conn, errors)
    t.start()
    with conn.exec_profile(cancel_surface=CancelSurface.RAISE_ON_CANCEL):
        with pytest.raises(e.QueryCanceled):
            conn.execute("select pg_sleep(2)")
    t.join()
    conn.rollback()
    assert not errors


def test_cancel_raise_on_cancel_success_result(conn, monkeypatch):
    # Deterministic cancel/result race seam: a cancel was dispatched after
    # send, yet the server reports a benign successful result. The frame must
    # still surface QueryCanceled instead of returning success.
    monkeypatch.setattr(
        psycopg.Cursor, "_cancel_dispatched_during_execute", lambda self: True
    )
    with conn.exec_profile(cancel_surface=CancelSurface.RAISE_ON_CANCEL):
        with pytest.raises(e.QueryCanceled):
            conn.execute("select 1")
    conn.rollback()


def test_executemany_raise_on_cancel_success_result(conn, monkeypatch):
    monkeypatch.setattr(
        psycopg.Cursor, "_cancel_dispatched_during_execute", lambda self: True
    )
    with conn.exec_profile(cancel_surface=CancelSurface.RAISE_ON_CANCEL):
        with pytest.raises(e.QueryCanceled):
            conn.cursor().executemany("select 1", [(), (), (), ()])
    conn.rollback()


def test_cancel_ignore_result_completes_without_closing(dsn):
    conn = autocommit_conn(dsn)
    errors: list[Exception] = []
    try:
        t = make_canceller(conn, errors)
        t.start()
        with conn.exec_profile(cancel_surface=CancelSurface.IGNORE_CANCEL_RESULT):
            cur = conn.execute("select pg_sleep(2)")
            # execute returned instead of raising; connection still usable
            assert not conn.closed
        t.join()
        assert conn.execute("select 1").fetchone()[0] == 1
        # the suppressed execute left the cursor in an empty (not errored) state
        assert cur.pgresult is None
        assert cur.statusmessage is None
    finally:
        conn.close()
    assert not errors


def test_ignore_cancel_does_not_swallow_unrelated_errors(conn):
    with conn.exec_profile(cancel_surface=CancelSurface.IGNORE_CANCEL_RESULT):
        with pytest.raises(e.UndefinedTable):
            conn.execute("select * from no_such_profile_table_xyz")
    conn.rollback()


# --- late push boundaries ----------------------------------------------------


def test_late_push_pipeline_admit_rejected(conn):
    with conn.pipeline():
        with pytest.raises(e.ProgrammingError):
            conn.push_exec_profile(
                ExecProfile(pipeline_admit=PipelineAdmit.REJECT_ENTER)
            )
        with pytest.raises(e.ProgrammingError):
            conn.push_exec_profile(
                ExecProfile(pipeline_admit=PipelineAdmit.FORCE_SYNC_ON_EXIT)
            )
        with pytest.raises(e.ProgrammingError):
            conn.push_exec_profile(ExecProfile(prepare_bind=PrepareBind.FORBID_PREPARE))
    assert conn.profiles.depth == 0


def test_late_push_while_command_in_flight_rejected(conn):
    seen: list[Exception] = []

    def worker():
        try:
            conn.execute("select pg_sleep(1.5)")
        except Exception as exc:
            seen.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    wait_active(conn.pgconn)
    try:
        with pytest.raises(e.ProgrammingError):
            conn.push_exec_profile(
                ExecProfile(cancel_surface=CancelSurface.RAISE_ON_CANCEL)
            )
    finally:
        t.join()
    assert not seen  # sleeping query finished normally
    conn.rollback()


# --- connection isolation ----------------------------------------------------


def test_two_connections_do_not_share_stack(dsn):
    c1 = psycopg.connect(dsn)
    c2 = psycopg.connect(dsn)
    try:
        with c1.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_ON):
            with c2.exec_profile(pipeline_admit=PipelineAdmit.REJECT_ENTER):
                assert c1.profiles.depth == 1
                assert c2.profiles.depth == 1
                assert c1.profiles.pipeline_admit is PipelineAdmit.PIN
                assert c2.profiles.autocommit_overlay is AutocommitOverlay.PIN

                # c1 may still use pipeline and autocommit semantics
                with c1.pipeline():
                    c1.execute("select 1")
                assert c1.info.transaction_status == IDLE

                # c2 may not enter a pipeline
                with pytest.raises(e.ProgrammingError):
                    with c2.pipeline():
                        pass

                # the two stacks are distinct objects
                assert c1.profiles is not c2.profiles
    finally:
        c1.close()
        c2.close()


# --- Copy is explicitly out of the stack ------------------------------------


def test_copy_does_not_read_autocommit_overlay(dsn, table):
    conn = autocommit_conn(dsn)
    try:
        with conn.exec_profile(autocommit_overlay=AutocommitOverlay.FORCE_OFF):
            with conn.cursor().copy(f"copy {TABLE} from stdin") as cp:
                cp.write_row(("via_copy", "9"))
            # pin path: no BEGIN, immediately visible from another connection
            assert conn.info.transaction_status == IDLE
        with observer(dsn) as obs:
            assert (
                obs.execute(
                    f"select count(*) from {TABLE} where name = 'via_copy'"
                ).fetchone()[0]
                == 1
            )
    finally:
        conn.close()
