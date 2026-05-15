from __future__ import annotations

import os
import unittest

from bench.chaos.run_chaos import is_pid_alive
from bench.harness.run import pctl, row_cost_per_million


class BenchHelperTests(unittest.TestCase):
    def test_percentile_interpolates_small_samples(self) -> None:
        self.assertEqual(pctl([], 0.95), 0.0)
        self.assertEqual(pctl([7.0], 0.95), 7.0)
        self.assertEqual(pctl([0.0, 10.0], 0.50), 5.0)
        self.assertEqual(pctl([0.0, 10.0], 0.95), 9.5)

    def test_pid_liveness_detects_current_process(self) -> None:
        self.assertTrue(is_pid_alive(os.getpid()))
        self.assertFalse(is_pid_alive(99999999))

    def test_row_cost_is_per_scenario(self) -> None:
        short = {"duration_s": 10.0, "total_output_tokens": 1000}
        long = {"duration_s": 20.0, "total_output_tokens": 1000}
        self.assertAlmostEqual(row_cost_per_million(short, gpu_hour_usd=3.6, gpu_count=1), 10.0)
        self.assertAlmostEqual(row_cost_per_million(long, gpu_hour_usd=3.6, gpu_count=1), 20.0)


if __name__ == "__main__":
    unittest.main()
