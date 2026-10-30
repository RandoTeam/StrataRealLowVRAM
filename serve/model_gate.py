"""serve/model_gate.py - #38: which model the requests run on, and the switch from one model to another.

A request holds a lease on the model it runs on for as long as it runs (Service.use_model), from before its prompt is
read with that model's tokenizer to the end of its answer.  A request for another model waits until no lease is held,
and the model is switched while nothing runs.  While it waits, new requests for the loaded model wait behind it, so a
busy model cannot keep another one out; requests go in the order they came, and when a switch is done every request
waiting for the new model goes together.  A switch that fails fails every request waiting for that model, so a model
that does not start is not tried again by each of them.

Nothing here starts or stops an engine: the `switch` a request brings does that.  Threads only; no server imports.
"""
from __future__ import annotations

import contextlib
import copy
import threading
import time


class Cancelled(Exception):
    """The request gave up its place while it waited (its client left)."""


class ModelGate:
    POLL_S = 0.25                                   # how often a waiting request looks at its cancel event
    ANY = object()                                  # a request's target when any loaded model will do: it never switches

    def __init__(self, current):
        """`current()` names the model that is loaded (any value that compares equal for the same model)."""
        self._current = current
        self.switching = None                       # the model a switch is loading, while it runs
        self.users = 0                              # the leases held
        self._queue: list[dict] = []                # the waiting requests, oldest first
        self._cond = threading.Condition()

    @property
    def current(self):
        return self._current()

    @contextlib.contextmanager
    def use(self, target, switch, cancel: threading.Event | None = None, timeout: float | None = None):
        """Hold a lease on `target` (ANY: the model that is loaded when the request's turn comes) for the `with` block,
        after `switch()` made it the loaded model if it was not.
        Raises whatever `switch()` raised (a copy for the requests that waited for it), Cancelled when `cancel` is
        set while waiting, TimeoutError after `timeout` seconds of waiting (None: no limit)."""
        self._enter(target, switch, cancel, timeout)
        try:
            yield
        finally:
            with self._cond:
                self.users -= 1
                self._schedule()
                self._cond.notify_all()

    def _enter(self, target, switch, cancel, timeout):
        deadline = None if timeout is None else time.monotonic() + timeout
        ticket = {"target": target, "go": False, "switch": False, "error": None}
        with self._cond:
            self._queue.append(ticket)
            self._schedule()
            while not (ticket["go"] or ticket["switch"] or ticket["error"]):
                if (cancel is not None and cancel.is_set()) or (deadline is not None and time.monotonic() >= deadline):
                    self._queue.remove(ticket)
                    self._schedule()
                    self._cond.notify_all()
                    raise Cancelled() if cancel is not None and cancel.is_set() else TimeoutError()
                self._cond.wait(self.POLL_S if deadline is None else min(self.POLL_S, deadline - time.monotonic()))
            if ticket["go"]:
                return
            if ticket["error"] is not None:
                raise copy.copy(ticket["error"])
            if cancel is not None and cancel.is_set():  # its turn came after its client left: the next one's turn
                self.switching = None
                self._queue.remove(ticket)
                self._schedule()
                self._cond.notify_all()
                raise Cancelled()
        try:                                        # this request switches: nothing runs, nothing else starts
            switch()
        except BaseException as e:
            with self._cond:
                self.switching = None
                self._queue.remove(ticket)
                for t in self._queue:                # the others waiting for this model fail with it
                    if t["target"] == target and isinstance(e, Exception):
                        t["error"] = e
                self._queue = [t for t in self._queue if t["error"] is None]
                self._schedule()
                self._cond.notify_all()
            raise
        with self._cond:
            self.switching = None
            self._queue.remove(ticket)
            self.users += 1
            for t in self._queue:                    # every request waiting for the model now loaded goes with it
                if t["target"] == target:
                    t["go"] = True
                    self.users += 1
            self._queue = [t for t in self._queue if not t["go"]]
            self._schedule()
            self._cond.notify_all()

    def _schedule(self):
        """Under the lock: let in the requests for the loaded model that are not behind one for another model; when
        nothing runs, give the oldest waiting request for another model its switch."""
        if self.switching is not None:
            return
        current = self.current
        while self._queue and self._queue[0]["target"] in (current, self.ANY):
            self._queue.pop(0)["go"] = True
            self.users += 1
        if self._queue and self.users == 0:
            head = self._queue[0]
            head["switch"] = True
            self.switching = head["target"]
