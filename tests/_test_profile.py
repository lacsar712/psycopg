"""
Shared helpers for the execution profile policy stack tests (sync + async).
"""

# Copyright (C) 2026 The Psycopg Team

from __future__ import annotations

import threading
import time

import psycopg

TABLE = "exec_profile_t"


def wait_active(pgconn, timeout=5.0):
    """Block until the server reports a command in flight (ACTIVE)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pgconn.transaction_status == psycopg.pq.TransactionStatus.ACTIVE:
            return
        time.sleep(0.005)
    raise AssertionError("the connection never reached ACTIVE status")


def make_canceller(conn, errors, delay=0.3):
    """Thread dispatching conn.cancel() after *delay* seconds."""

    def cancel():
        try:
            time.sleep(delay)
            conn.cancel()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    return threading.Thread(target=cancel)
