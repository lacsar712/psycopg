"""
Connection-scoped execution profile policy stack.

An :class:`ExecProfile` is a frame describing how a *nested region* of code
should behave for four independent execution axes:

- autocommit (:class:`AutocommitOverlay`),
- pipeline admission (:class:`PipelineAdmit`),
- prepared statements (:class:`PrepareBind`),
- what happens when :meth:`~psycopg.Connection.cancel` reaches the server
  (:class:`CancelSurface`).

The stack is owned by a connection instance: one :class:`ExecProfileStack` is
created in :meth:`BaseConnection.__init__` and every read goes through
``conn._profiles``. There is intentionally no process-level state and no
:class:`~contextvars.ContextVar`: two connections (even when driven from the
same thread or the same asyncio task) always observe their own top frame.

The empty stack is equivalent to the historical behaviour: every axis reports
its ``pin`` value and all the connection flat attributes (``autocommit``,
``prepare_threshold``, ...) keep driving the execution.
"""

# Copyright (C) 2026 The Psycopg Team

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import TYPE_CHECKING, Any

from . import errors as e
from . import pq

if TYPE_CHECKING:
    from ._connection_base import BaseConnection

IDLE = pq.TransactionStatus.IDLE
ACTIVE = pq.TransactionStatus.ACTIVE


class AutocommitOverlay(str, Enum):
    """How a frame overlays the connection ``autocommit`` flag.

    - ``PIN``: keep the connection :attr:`~psycopg.Connection.autocommit`
      value (and the regular ``transaction()``/savepoint behaviour).
    - ``FORCE_ON``: every ``execute()`` in the frame must behave as if
      autocommit were enabled and the statement durably committed. Entering a
      transaction while the frame is active is refused, and executing with an
      outer transaction already open raises instead of enrolling into it.
    - ``FORCE_OFF``: the frame must not rely on implicit autocommit. Writes
      stay pending until an explicit ``commit()`` or a successful
      ``transaction()`` exit.
    """

    PIN = "pin"
    FORCE_ON = "force_on"
    FORCE_OFF = "force_off"


class PipelineAdmit(str, Enum):
    """Whether a frame allows entering pipeline mode.

    - ``PIN``: keep the regular ``pipeline()`` entry/sync behaviour.
    - ``REJECT_ENTER``: entering ``pipeline()`` raises before the connection
      reaches the enqueue state.
    - ``FORCE_SYNC_ON_EXIT``: the pipeline must perform an observable
      sync/flush draining every queued command before it exits.
    """

    PIN = "pin"
    REJECT_ENTER = "reject_enter"
    FORCE_SYNC_ON_EXIT = "force_sync_on_exit"


class PrepareBind(str, Enum):
    """How a frame binds ``execute()`` to the prepared statement machinery.

    - ``PIN``: keep the :class:`~psycopg._preparing.PrepareManager` policy.
    - ``FORCE_PREPARE``: same-shape repeated executes must use a named
      prepared statement, regardless of ``prepare_threshold``.
    - ``FORBID_PREPARE``: neither hit nor populate the prepare cache; always
      send through the non-prepared query path.
    """

    PIN = "pin"
    FORCE_PREPARE = "force_prepare"
    FORBID_PREPARE = "forbid_prepare"


class CancelSurface(str, Enum):
    """How a frame surfaces a :meth:`~psycopg.Connection.cancel` that reached
    the server while a query was running.

    - ``PIN``: keep the historical cancel/executing-query race.
    - ``RAISE_ON_CANCEL``: once a cancel has been dispatched during the
      current execute, the execute must surface :class:`~psycopg.errors.QueryCanceled`,
      even if the server happened to return a successful result.
    - ``IGNORE_CANCEL_RESULT``: a ``57014`` result caused by a cancel
      dispatched during the execute is swallowed and the execute completes
      without raising (the connection is *not* closed to emulate success).
    """

    PIN = "pin"
    RAISE_ON_CANCEL = "raise_on_cancel"
    IGNORE_CANCEL_RESULT = "ignore_cancel_result"


AutocommitOverlay.__module__ = "psycopg"
PipelineAdmit.__module__ = "psycopg"
PrepareBind.__module__ = "psycopg"
CancelSurface.__module__ = "psycopg"


@dataclass(frozen=True)
class ExecProfile:
    """A value of the execution profile policy stack.

    Every axis defaults to ``pin``, so a profile only describes the axes that
    actually need to diverge from the connection behaviour. Plain ``str``
    values (e.g. ``"force_on"``) are accepted and normalised.
    """

    autocommit_overlay: AutocommitOverlay = AutocommitOverlay.PIN
    pipeline_admit: PipelineAdmit = PipelineAdmit.PIN
    prepare_bind: PrepareBind = PrepareBind.PIN
    cancel_surface: CancelSurface = CancelSurface.PIN

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "autocommit_overlay",
            AutocommitOverlay(self.autocommit_overlay),  # type: ignore[arg-type]
        )
        object.__setattr__(
            self,
            "pipeline_admit",
            PipelineAdmit(self.pipeline_admit),  # type: ignore[arg-type]
        )
        object.__setattr__(
            self,
            "prepare_bind",
            PrepareBind(self.prepare_bind),  # type: ignore[arg-type]
        )
        object.__setattr__(
            self,
            "cancel_surface",
            CancelSurface(self.cancel_surface),  # type: ignore[arg-type]
        )


ExecProfile.__module__ = "psycopg"


@dataclass
class _ProfileFrame:
    """A pushed :class:`ExecProfile` with its per-connection bookkeeping."""

    profile: ExecProfile


class ExecProfileStack:
    """
    Per-connection stack of :class:`ExecProfile` frames.

    The stack lives on the connection (``conn._profiles``); it never uses
    process globals or :class:`~contextvars.ContextVar`. Two connections can
    be active concurrently on the same thread/asyncio task without ever
    reading each other's top frame, because each read is keyed on the
    connection instance.

    Pushing a frame after the connection has crossed a boundary the frame
    would govern (a command already sent, a Pipeline already entered, a
    Transaction already entered for ``force_on``) raises instead of silently
    applying to a region that started under a different frame.
    """

    def __init__(self, conn: BaseConnection[Any]):
        self._conn = conn
        self._frames: list[_ProfileFrame] = []

    # -- stack mechanics ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._frames)

    def __bool__(self) -> bool:
        return bool(self._frames)

    @property
    def depth(self) -> int:
        """Number of frames currently on the stack."""
        return len(self._frames)

    @property
    def frames(self) -> tuple[ExecProfile, ...]:
        """Snapshots of the profiles on the stack, bottom to top."""
        return tuple(frame.profile for frame in self._frames)

    @property
    def top(self) -> ExecProfile | None:
        """The active profile, or `!None` if the stack is empty."""
        return self._frames[-1].profile if self._frames else None

    def push(self, profile: ExecProfile) -> _ProfileFrame:
        """Push a profile frame, enforcing late-push boundary rules.

        Raise `~psycopg.errors.ProgrammingError` (without mutating the stack)
        when pushing would govern a boundary already crossed:

        - a command is currently in flight (server transaction status
          ``ACTIVE``): no frame can be pushed after ``execute()`` sent data;
        - a Pipeline is active: ``pipeline_admit``/``prepare_bind`` other than
          ``pin`` cannot be pushed;
        - a transaction is open: ``autocommit_overlay=force_on`` cannot be
          pushed.
        """
        conn = self._conn
        txstatus = conn.pgconn.transaction_status

        if txstatus == ACTIVE:
            raise e.ProgrammingError(
                "cannot push an execution profile while a command is in flight"
                " (server transaction status ACTIVE): push before sending execute()"
            )

        if conn._pipeline is not None and (
            profile.pipeline_admit is not PipelineAdmit.PIN
            or profile.prepare_bind is not PrepareBind.PIN
        ):
            raise e.ProgrammingError(
                "cannot push an execution profile with pipeline_admit or"
                " prepare_bind set while a Pipeline is already active:"
                " push the profile before entering pipeline()"
            )

        if profile.autocommit_overlay is AutocommitOverlay.FORCE_ON and (
            conn._num_transactions or txstatus != IDLE
        ):
            raise e.ProgrammingError(
                "cannot push autocommit_overlay=force_on while a transaction"
                " is open: push the profile before entering transaction()"
            )

        frame = _ProfileFrame(profile)
        self._frames.append(frame)
        return frame

    def pop(self, frame: _ProfileFrame) -> None:
        """Pop exactly `frame`, which must be the current top."""
        if not self._frames:
            raise e.ProgrammingError(
                "cannot pop an execution profile: the stack is empty"
            )
        if self._frames[-1] is not frame:
            raise e.ProgrammingError(
                "cannot pop an execution profile out of order: another frame"
                " was pushed on top of it"
            )
        self._frames.pop()

    # -- axis reads (independent exit points) -------------------------------
    #
    # Resolution walks the stack from the top down: the nearest frame setting
    # an axis to a non-pin value wins; a frame pinning an axis keeps whatever
    # was already in effect below it (an outer frame, or the connection flat
    # behaviour when no frame sets the axis). This makes frames compose:
    # force_off outside + forbid_prepare inside yields both effects at once.

    def _resolve(self, field: str, pin: Any) -> Any:
        for frame in reversed(self._frames):
            value = getattr(frame.profile, field)
            if value is not pin:
                return value
        return pin

    @property
    def autocommit_overlay(self) -> AutocommitOverlay:
        """Effective ``autocommit_overlay`` (outer frames show through pins)."""
        return self._resolve("autocommit_overlay", AutocommitOverlay.PIN)

    @property
    def pipeline_admit(self) -> PipelineAdmit:
        """Effective ``pipeline_admit`` (outer frames show through pins)."""
        return self._resolve("pipeline_admit", PipelineAdmit.PIN)

    @property
    def prepare_bind(self) -> PrepareBind:
        """Effective ``prepare_bind`` (outer frames show through pins)."""
        return self._resolve("prepare_bind", PrepareBind.PIN)

    @property
    def cancel_surface(self) -> CancelSurface:
        """Effective ``cancel_surface`` (outer frames show through pins)."""
        return self._resolve("cancel_surface", CancelSurface.PIN)

    # -- effective views, merged with connection state ----------------------

    def effective_autocommit(self) -> bool:
        """Effective autocommit for an ``execute()`` boundary.

        Merge the connection flat ``autocommit`` flag with the top overlay.
        ``force_on`` additionally refuses (with a testable error) to deliver
        an autocommit-like execute while an outer transaction is open: it
        never toggles ``autocommit`` to fake it.
        """
        overlay = self.autocommit_overlay
        conn = self._conn
        if overlay is AutocommitOverlay.FORCE_ON:
            if conn._num_transactions or conn.pgconn.transaction_status != IDLE:
                raise e.ProgrammingError(
                    "autocommit_overlay=force_on cannot run execute() inside"
                    " an open transaction: commit/rollback the outer"
                    " transaction or pop the profile frame"
                )
            return True
        if overlay is AutocommitOverlay.FORCE_OFF:
            return False
        return bool(conn._autocommit)


class _ExecProfileContext:
    """Dual sync/async context manager pushing/popping an :class:`ExecProfile`.

    Push/pop perform no IO, so the same object supports ``with`` on
    `Connection` and ``async with`` on `AsyncConnection`. The frame is always
    popped on the exception path, so frames never leak.
    """

    __slots__ = ("_stack", "_profile", "_frame")

    def __init__(self, stack: ExecProfileStack, profile: ExecProfile):
        self._stack = stack
        self._profile = profile
        self._frame: _ProfileFrame | None = None

    def __enter__(self) -> ExecProfile:
        self._frame = self._stack.push(self._profile)
        return self._profile

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        frame, self._frame = self._frame, None
        if frame is not None:
            self._stack.pop(frame)
        return False

    async def __aenter__(self) -> ExecProfile:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        return self.__exit__(exc_type, exc_val, exc_tb)
