import unittest
from gpu_memory import ProcessGpuMemorySampler


class FakeCounters:
    def __init__(self, rows):
        self.rows, self.closed = iter(rows), False
    def read(self):
        value = next(self.rows)
        if isinstance(value, Exception):
            raise value
        return value
    def close(self):
        self.closed = True


class GpuMemoryTests(unittest.TestCase):
    def test_sums_adapters_and_preserves_sampled_peak(self):
        sampler = ProcessGpuMemorySampler(42, api=FakeCounters([[(0, 100), (1, 200)], [(0, 50)]]))
        self.assertEqual(sampler.sample(), 300)
        self.assertEqual(sampler.sample(), 50)
        self.assertEqual(sampler.snapshot()["peak_bytes"], 300)
        self.assertEqual(sampler.snapshot()["samples"], 2)

    def test_missing_or_invalid_counter_is_unknown_not_zero(self):
        sampler = ProcessGpuMemorySampler(42, api=FakeCounters([[], [(0xC0000BC6, 0)], [(0, -1)]]))
        self.assertIsNone(sampler.sample())
        self.assertIsNone(sampler.sample())
        self.assertIsNone(sampler.sample())
        self.assertIsNone(sampler.snapshot()["peak_bytes"])

    def test_valid_zero_is_distinct_from_unavailable(self):
        sampler = ProcessGpuMemorySampler(42, api=FakeCounters([[(0, 0)]]))
        self.assertEqual(sampler.sample(), 0)
        self.assertEqual(sampler.snapshot()["samples"], 1)

    def test_failures_leave_reason_and_close_backend(self):
        api = FakeCounters([OSError("counter unavailable")])
        sampler = ProcessGpuMemorySampler(42, api=api)
        self.assertIsNone(sampler.sample())
        self.assertIn("counter unavailable", sampler.snapshot()["unavailable_reason"])
        sampler.close()
        self.assertTrue(api.closed)


if __name__ == "__main__":
    unittest.main()
