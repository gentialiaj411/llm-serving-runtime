from __future__ import annotations

import os
import unittest

from bench.chaos.run_chaos import is_pid_alive
from bench.harness.run import pctl


class BenchHelperTests(unittest.TestCase):
    def test_percentile_interpolates_small_samples(self) -> None:
        self.assertEqual(pctl([], 0.95), 0.0)
        self.assertEqual(pctl([7.0], 0.95), 7.0)
        self.assertEqual(pctl([0.0, 10.0], 0.50), 5.0)
        self.assertEqual(pctl([0.0, 10.0], 0.95), 9.5)

    def test_pid_liveness_detects_current_process(self) -> None:
        self.assertTrue(is_pid_alive(os.getpid()))
        self.assertFalse(is_pid_alive(99999999))


if __name__ == "__main__":
    unittest.main()
