"""Small semantic and conservation checks; not stability tests."""
import unittest

import hyperstochastic_fcfm as sim


class FCFMTests(unittest.TestCase):
    def test_extension_keeps_creation_order(self):
        groups = [frozenset({4}), frozenset({2})]
        self.assertIsNone(sim.fcfm_step(groups, 1))
        self.assertEqual(groups, [frozenset({1, 4}), frozenset({2})])
        self.assertIsNone(sim.fcfm_step(groups, 3))
        self.assertEqual(groups, [frozenset({1, 3, 4}), frozenset({2})])
        self.assertEqual(sim.fcfm_step(groups, 2), frozenset({1, 2, 3, 4}))
        self.assertEqual(groups, [frozenset({2})])

    def test_skip_incompatible_and_complete_immediately(self):
        groups = [frozenset({1}), frozenset({2}), frozenset({4})]
        self.assertEqual(sim.fcfm_step(groups, 1), frozenset({1, 2}))
        self.assertEqual(groups, [frozenset({1}), frozenset({4})])
        self.assertIsNone(sim.fcfm_step([], 4))

    def test_manuscript_priority_differs_from_lexicographic(self):
        state = sim.model.empty_wml_state()
        state[frozenset({3})] = 2
        state[frozenset({2, 4})] = 1
        self.assertEqual(sim.manuscript_wml_step(state, 1), frozenset({1, 3}))
        self.assertEqual(state[frozenset({2, 4})], 1)

    def test_conservation_and_reproducibility(self):
        for seed in range(3):
            first = sim.run_comparison(500, seed, validate=True)
            second = sim.run_comparison(500, seed, validate=True)
            self.assertEqual(first[2], second[2])
            sim.np.testing.assert_array_equal(first[0], second[0])


if __name__ == "__main__":
    unittest.main()
