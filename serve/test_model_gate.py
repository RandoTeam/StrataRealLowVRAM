"""#38: serve/model_gate.py - the leases requests hold on the loaded model, and the switch to another one, with threads
and no engine.  python -m unittest serve.test_model_gate -v"""
import threading
import time
import unittest

from serve.model_gate import Cancelled, ModelGate


def until(cond, t=5.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class Gate(unittest.TestCase):
    def setUp(self):
        self.loaded = "a"
        self.gate = ModelGate(lambda: self.loaded)
        self.log = []                              # what happened, in order
        self.fail = {}                             # a model whose switch raises this
        self.hold_switch = None                    # an event a switch waits for

    def switch_to(self, target):
        def switch():
            self.log.append(f"switch {target}")
            if self.hold_switch is not None:
                self.hold_switch.wait(5)
            if target in self.fail:
                raise self.fail[target]
            self.loaded = target
        return switch

    def request(self, name, target, release: threading.Event, cancel=None, timeout=None):
        """A request in a thread: enters (logs "<name> in"), holds its lease until `release`, then logs "<name> out"
        - or logs "<name> <error>" when it does not get in."""
        def run():
            try:
                with self.gate.use(target, self.switch_to(target), cancel, timeout):
                    self.log.append(f"{name} in {self.loaded}")
                    release.wait(5)
                self.log.append(f"{name} out")
            except Exception as e:                 # noqa: BLE001 - the test reads it from the log
                self.log.append(f"{name} {type(e).__name__}")
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    def test_requests_for_the_loaded_model_run_together(self):
        go = threading.Event()
        threads = [self.request(f"a{i}", "a", go) for i in range(3)]
        self.assertTrue(until(lambda: self.gate.users == 3))
        go.set()
        for t in threads:
            t.join(5)
        self.assertEqual((self.gate.users, self.loaded), (0, "a"))
        self.assertNotIn("switch", " ".join(self.log))

    def test_a_switch_waits_until_no_request_runs(self):
        a, b = threading.Event(), threading.Event()
        self.request("a1", "a", a)
        self.assertTrue(until(lambda: self.gate.users == 1))
        tb = self.request("b1", "b", b)
        time.sleep(0.4)
        self.assertEqual(self.log, ["a1 in a"])        # b1 waits: a1 still runs on a
        a.set()
        self.assertTrue(until(lambda: "b1 in b" in self.log))
        self.assertEqual(self.log, ["a1 in a", "a1 out", "switch b", "b1 in b"])
        b.set()
        tb.join(5)

    def test_a_waiting_switch_holds_back_new_requests_for_the_loaded_model(self):
        """No starvation: a1 runs, b1 waits for it, a2 comes after b1 - a2 runs only after b1, on a switched back."""
        a1, b1, a2 = threading.Event(), threading.Event(), threading.Event()
        self.request("a1", "a", a1)
        self.assertTrue(until(lambda: self.gate.users == 1))
        self.request("b1", "b", b1)
        self.assertTrue(until(lambda: len(self.gate._queue) == 1))
        self.request("a2", "a", a2)
        time.sleep(0.4)
        self.assertEqual(self.log, ["a1 in a"])        # a2 did not slip in on the loaded model
        a1.set()
        self.assertTrue(until(lambda: "b1 in b" in self.log))
        b1.set()
        self.assertTrue(until(lambda: "a2 in a" in self.log))
        a2.set()
        self.assertTrue(until(lambda: "a2 out" in self.log))
        self.assertEqual(self.log, ["a1 in a", "a1 out", "switch b", "b1 in b", "b1 out", "switch a", "a2 in a", "a2 out"])

    def test_requests_for_the_new_model_go_together(self):
        """b1, c1, b2 wait in that order: b2 goes with b1 (one switch, not three), c1 after both."""
        a1, b, c1 = threading.Event(), threading.Event(), threading.Event()
        self.request("a1", "a", a1)
        self.assertTrue(until(lambda: self.gate.users == 1))
        for i, (name, target, ev) in enumerate((("b1", "b", b), ("c1", "c", c1), ("b2", "b", b))):
            self.request(name, target, ev)
            self.assertTrue(until(lambda: len(self.gate._queue) == i + 1))   # queued in this order
        a1.set()
        self.assertTrue(until(lambda: "b1 in b" in self.log and "b2 in b" in self.log))
        self.assertNotIn("c1 in c", self.log)
        b.set()
        self.assertTrue(until(lambda: "c1 in c" in self.log))
        c1.set()
        self.assertEqual([x for x in self.log if x.startswith("switch")], ["switch b", "switch c"])

    def test_a_failed_switch_fails_every_request_waiting_for_that_model(self):
        self.fail["b"] = RuntimeError("no VRAM")
        a1, ev = threading.Event(), threading.Event()
        self.request("a1", "a", a1)
        self.assertTrue(until(lambda: self.gate.users == 1))
        for i, (name, target) in enumerate((("b1", "b"), ("b2", "b"), ("c1", "c"))):
            self.request(name, target, ev)
            self.assertTrue(until(lambda: len(self.gate._queue) == i + 1))
        a1.set()
        self.assertTrue(until(lambda: "c1 in c" in self.log))
        ev.set()
        self.assertTrue(until(lambda: "c1 out" in self.log))
        self.assertIn("b1 RuntimeError", self.log)
        self.assertIn("b2 RuntimeError", self.log)
        self.assertEqual([x for x in self.log if x.startswith("switch")], ["switch b", "switch c"])   # b tried once
        self.assertEqual((self.loaded, self.gate.users, self.gate.switching), ("c", 0, None))

    def test_a_request_whose_client_left_gives_its_place_up(self):
        a1, cancel = threading.Event(), threading.Event()
        self.request("a1", "a", a1)
        self.assertTrue(until(lambda: self.gate.users == 1))
        self.request("b1", "b", threading.Event(), cancel=cancel)
        self.assertTrue(until(lambda: len(self.gate._queue) == 1))
        cancel.set()
        self.assertTrue(until(lambda: "b1 Cancelled" in self.log))
        a1.set()
        self.assertTrue(until(lambda: "a1 out" in self.log))
        self.assertEqual((self.loaded, self.gate._queue, self.gate.users), ("a", [], 0))
        self.assertNotIn("switch b", self.log)

    def test_a_wait_with_a_timeout_ends_without_a_switch(self):
        a1 = threading.Event()
        self.request("a1", "a", a1)
        self.assertTrue(until(lambda: self.gate.users == 1))
        t0 = time.time()
        with self.assertRaises(TimeoutError):
            with self.gate.use("b", self.switch_to("b"), timeout=0.3):
                pass
        self.assertLess(time.time() - t0, 2)
        a1.set()
        self.assertTrue(until(lambda: "a1 out" in self.log))
        self.assertEqual((self.loaded, self.gate._queue), ("a", []))

    def test_a_request_for_any_model_runs_on_the_one_loaded_at_its_turn(self):
        """A request naming no installed model waits behind a switch like the others, then runs on the new model:
        it never switches back to the model that was loaded when it came."""
        a1, b1, x = threading.Event(), threading.Event(), threading.Event()
        self.request("a1", "a", a1)
        self.assertTrue(until(lambda: self.gate.users == 1))
        self.request("b1", "b", b1)
        self.assertTrue(until(lambda: len(self.gate._queue) == 1))
        self.request("x", ModelGate.ANY, x)
        self.assertTrue(until(lambda: len(self.gate._queue) == 2))
        a1.set()
        self.assertTrue(until(lambda: "x in b" in self.log))
        b1.set()
        x.set()
        self.assertTrue(until(lambda: "x out" in self.log and "b1 out" in self.log))
        self.assertEqual([e for e in self.log if e.startswith("switch")], ["switch b"])

    def test_nothing_runs_while_a_model_switches(self):
        self.hold_switch = threading.Event()
        b1, a2 = threading.Event(), threading.Event()
        self.request("b1", "b", b1)
        self.assertTrue(until(lambda: self.gate.switching == "b"))
        self.request("a2", "a", a2)                    # the model still loaded is a, but it is being switched away
        self.request("b2", "b", b1)
        time.sleep(0.4)
        self.assertEqual((self.gate.users, self.log), (0, ["switch b"]))
        self.hold_switch.set()
        self.assertTrue(until(lambda: "b1 in b" in self.log and "b2 in b" in self.log))
        self.assertNotIn("a2 in a", self.log)
        b1.set()
        self.assertTrue(until(lambda: "a2 in a" in self.log))
        a2.set()
        self.assertTrue(until(lambda: "a2 out" in self.log))


if __name__ == "__main__":
    unittest.main()
