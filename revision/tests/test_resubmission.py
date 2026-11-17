"""Tests of train-only calibration, exclusion and controlled resampling."""
import unittest
import numpy as np
import pandas as pd
from ieee_access_resubmit.experiments import (
    fit_height_anchor, matched_sample, gateway_split, disjoint_spatial_splits,
)
from hybrid_pathloss import packet_groups
from revision_experiments import minimum_distances


class ResubmissionProtocolTests(unittest.TestCase):
    def test_anchor_fit_ignores_every_held_out_target(self):
        rng = np.random.default_rng(9)
        n = 400
        df = pd.DataFrame({'gw': np.tile(['a','b','c','d'], n//4),
            'antenna_height': np.tile([12.,16.,30.,55.], n//4),
            'gw_distance_m': rng.uniform(100, 10000, n), 'path_loss': rng.normal(130, 10, n)})
        df['gw_distance_rounded_km'] = (df.gw_distance_m/1000).round(2)
        train = np.arange(200)
        before, meta = fit_height_anchor(df, train)
        df.loc[200:, 'path_loss'] += 1000
        after, changed_meta = fit_height_anchor(df, train)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(meta, changed_meta)
        self.assertTrue(meta['full_rank'])

    def test_stratified_subsamples_preserve_exact_counts_and_candidate_membership(self):
        labels = np.repeat(np.arange(5), 20)
        original = np.arange(100)
        reference = np.array([1,2,21,40,41,42,61,81,82])
        selected = matched_sample(original, reference, labels, np.random.default_rng(4))
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(set(selected) <= set(original))
        np.testing.assert_array_equal(np.bincount(labels[selected]), np.bincount(labels[reference]))
        with self.assertRaises(ValueError):
            matched_sample(np.arange(20), reference, labels, np.random.default_rng(4))

    def test_gateway_excludes_shared_packets_and_nearby_positions_from_both_sets(self):
        df = pd.DataFrame({'latitude': [50.,50.]+list(np.linspace(50.0001,50.2,100)),
            'longitude': [7.]*102, 'gw': ['held','other']+['other']*100,
            'pkt_number': ['same','same']+[str(i) for i in range(100)],
            'transceived_at': ['2021-01-01']*102})
        parts, _ = gateway_split(df, 'held', 500)
        train, val, test = parts
        groups = packet_groups(df)
        for dev in (train, val):
            self.assertGreater(len(dev), 0)
            self.assertFalse(set(groups.iloc[dev]) & set(groups.iloc[test]))
            self.assertGreaterEqual(minimum_distances(np.deg2rad(df[['latitude','longitude']].to_numpy()), dev, test).min(), 500)
        self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[val]))

    def test_spatial_test_folds_are_disjoint_and_cover_every_row(self):
        lat, lon = np.meshgrid(np.linspace(50, 50.2, 25), np.linspace(7, 7.3, 25))
        df = pd.DataFrame({'latitude': lat.ravel(), 'longitude': lon.ravel()})
        tests = []
        for _, parts in disjoint_spatial_splits(df):
            tests.extend(parts[2])
            for a,b in ((0,1),(0,2),(1,2)):
                self.assertFalse(set(parts[a]) & set(parts[b]))
                if len(parts[a]) and len(parts[b]):
                    self.assertGreaterEqual(minimum_distances(np.deg2rad(df[['latitude','longitude']].to_numpy()), parts[a], parts[b]).min(), 500)
        np.testing.assert_array_equal(np.sort(tests), np.arange(len(df)))


if __name__ == '__main__': unittest.main()
