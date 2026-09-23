"""Adversarial tests: python -m unittest discover -s tests -v."""
import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from analytics_core import (
    BOUNDARY_LIMITATION, DataValidationError, Thresholds, _communities, _hits,
    _temporal_features, analyze, classify_role, explain_node,
    find_common_recipients, load_dataset, validate_outputs, write_outputs, analyze_sensitivity,
)

LARGE_ID = 100000003115284100


def fixture(folder):
    nodes = pd.DataFrame({"gid": [1, 2, LARGE_ID, 4, 5, 6, 9],
                          "depth": [0, 0, 1, 2, 3, 4, 0],
                          "is_seed": [True, True, False, False, False, False, True]})
    tx = pd.DataFrame({"src": [1, 1, 2, LARGE_ID, 4, 5],
                       "dst": [LARGE_ID, LARGE_ID, LARGE_ID, 4, 5, 6],
                       "date": pd.to_datetime(["2026-07-01", "2026-07-01", "2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"]),
                       "sum_kzt": [5000., 5000., 10000., 20000., 20000., 20000.]})
    edges = tx.groupby(["src", "dst"], as_index=False).agg(sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size"))
    depth = dict(zip(nodes.gid, nodes.depth))
    edges["depth"] = edges.src.map(lambda src: min(depth[src] + 1, 4))
    for name, frame in (("nodes", nodes), ("edges", edges), ("transactions", tx)):
        frame.to_parquet(folder / f"{name}.parquet", index=False)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        fixture(self.folder)

    def tearDown(self):
        self.temp.cleanup()

    def test_csv_roundtrip_and_normalized_metrics(self):
        result = analyze(self.folder)
        paths = write_outputs(result, self.folder / "output")
        self.assertEqual(len(paths), 3)
        nodes, clusters, top = [pd.read_csv(path) for path in paths]
        validate_outputs(nodes, clusters, top, set(result.graph))
        self.assertIn(LARGE_ID, set(nodes.gid))
        self.assertEqual(len(nodes), 7)
        self.assertTrue(nodes.sensitivity_scenarios.eq(36).all())
        self.assertAlmostEqual(nodes.page_rank.sum(), 1)
        self.assertAlmostEqual(nodes.hub_score.sum(), 1)
        self.assertAlmostEqual(nodes.authority_score.sum(), 1)

    def test_boundary_and_isolates(self):
        result = analyze(self.folder)
        nodes = result.nodes_roles.set_index("gid")
        self.assertIn(BOUNDARY_LIMITATION, nodes.loc[6, "limitation"])
        self.assertNotIn(nodes.loc[6, "role"], {"transit", "terminal"})
        self.assertIn(9, result.graph)
        self.assertEqual(nodes.loc[9, "priority_score"], 0)
        self.assertEqual(nodes.loc[9, "role"], "peripheral")
        self.assertNotEqual(nodes.loc[9, "cluster_id"], nodes.loc[1, "cluster_id"])

    def test_common_recipients_direction_and_horizon(self):
        result = analyze(self.folder)
        common = find_common_recipients(result, ["1", "2"])
        self.assertEqual({item["gid"] for item in common}, {str(LARGE_ID), "4", "5", "6"})
        self.assertTrue(all(item["source_count"] == 2 for item in common))
        for item in common:
            for path in item["paths"]:
                self.assertLessEqual(len(path) - 1, 4)
                self.assertTrue(all(result.graph.has_edge(int(a), int(b)) for a, b in zip(path, path[1:])))
        self.assertEqual(find_common_recipients(result, [6, 9]), [])
        with self.assertRaises(ValueError):
            find_common_recipients(result, [1, 1])
        with self.assertRaises(ValueError):
            find_common_recipients(result, [1, 999])

    def test_repeated_transactions_preserved(self):
        _, _, tx = load_dataset(self.folder)
        self.assertEqual(len(tx), 6)
        result = analyze(self.folder)
        self.assertEqual(result.graph[1][LARGE_ID]["n_tx"], 2)
        self.assertEqual(result.graph[1][LARGE_ID]["amount_cents"], 1_000_000)

    def test_mismatched_aggregate_fails(self):
        edges = pd.read_parquet(self.folder / "edges.parquet")
        edges.loc[0, "sum_kzt"] += 5000
        edges.to_parquet(self.folder / "edges.parquet", index=False)
        with self.assertRaisesRegex(DataValidationError, "суммам"):
            analyze(self.folder)

    def test_unknown_endpoint_fails(self):
        tx = pd.read_parquet(self.folder / "transactions.parquet")
        tx.loc[0, "dst"] = 999
        tx.to_parquet(self.folder / "transactions.parquet", index=False)
        with self.assertRaisesRegex(DataValidationError, "отсутствующие"):
            load_dataset(self.folder)

    def test_float_ids_rejected(self):
        nodes = pd.read_parquet(self.folder / "nodes.parquet")
        nodes["gid"] = nodes.gid.astype(float)
        nodes.to_parquet(self.folder / "nodes.parquet", index=False)
        with self.assertRaisesRegex(DataValidationError, "целочисленный"):
            load_dataset(self.folder)

    def test_determinism(self):
        first, second = analyze(self.folder), analyze(self.folder)
        pd.testing.assert_frame_equal(first.nodes_roles, second.nodes_roles)
        pd.testing.assert_frame_equal(first.clusters, second.clusters)

    def test_explanations_include_rows_safe_ids(self):
        result = analyze(self.folder)
        detail = explain_node(result, str(LARGE_ID))
        self.assertEqual(detail["gid"], str(LARGE_ID))
        self.assertEqual(len(detail["transactions"]), 4)
        self.assertEqual(len(detail["blocks"]), 4)
        self.assertEqual(detail["sensitivity"]["scenario_count"], 36)
        json.dumps(detail, ensure_ascii=False, allow_nan=False)
        self.assertTrue(all(isinstance(row["src"], str) for row in detail["transactions"]))

    def test_transit_requires_time_seed_balance_not_used(self):
        result = analyze(self.folder)
        row = result.features.loc[LARGE_ID].to_dict()
        self.assertEqual(classify_role(row, Thresholds(), result.coordinator_cutoff), "transit")
        row["temporal_in_share"] = row["temporal_out_share"] = 0
        self.assertNotEqual(classify_role(row, Thresholds(), result.coordinator_cutoff), "transit")
        row["is_seed"] = True
        row["temporal_in_share"] = row["temporal_out_share"] = 1
        self.assertNotEqual(classify_role(row, Thresholds(), result.coordinator_cutoff), "transit")

    def test_sensitivity_detects_threshold_border(self):
        result = analyze(self.folder)
        row = result.features.loc[LARGE_ID].to_dict()
        row.update(in_partners=3, out_partners=1, pass_through=0.4,
                   temporal_in_share=0.0, temporal_out_share=0.0)
        sensitivity = analyze_sensitivity(pd.DataFrame([row]), Thresholds(), result.coordinator_cutoff)[LARGE_ID]
        self.assertEqual(sensitivity["baseline_role"], "consolidator")
        self.assertLess(sensitivity["stability_score"], 1)
        self.assertIn("peripheral", sensitivity["alternative_roles"])

    def test_validation_detects_nan_and_missing_label(self):
        result = analyze(self.folder)
        nodes = result.nodes_roles.copy()
        nodes.loc[0, "score"] = np.nan
        with self.assertRaises(DataValidationError):
            validate_outputs(nodes, result.clusters, result.top_nodes, set(result.graph))
        nodes = result.nodes_roles.copy()
        nodes.loc[nodes.depth.eq(4), "limitation"] = "incorrect"
        with self.assertRaisesRegex(DataValidationError, "граничные"):
            validate_outputs(nodes, result.clusters, result.top_nodes, set(result.graph))


class AlgorithmTests(unittest.TestCase):
    def test_communities_reciprocal_edges_and_isolate(self):
        graph = nx.DiGraph()
        graph.add_edge(1, 2, amount_cents=100)
        graph.add_edge(2, 1, amount_cents=200)
        graph.add_node(3)
        communities = _communities(graph, 42)
        self.assertEqual(communities[1], communities[2])
        self.assertNotEqual(communities[1], communities[3])

    def test_temporal_matching_never_reuses_money_or_orders_same_day(self):
        tx = pd.DataFrame({"src": [1, 2, 2], "dst": [2, 3, 4],
                           "date": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
                           "amount_cents": [1000000, 1000000, 1000000]})
        features = _temporal_features(tx)[2]
        self.assertEqual(features["temporal_matched_amount"], 10000)
        self.assertEqual(features["temporal_out_share"], 0.5)
        tx["date"] = pd.to_datetime("2026-07-01")
        self.assertEqual(_temporal_features(tx)[2]["temporal_matched_amount"], 0)

    def test_hits_and_communities_edgeless_graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([1, 2])
        hubs, authorities = _hits(graph)
        self.assertEqual(hubs, {1: 0, 2: 0})
        self.assertEqual(authorities, hubs)
        self.assertEqual(len(set(_communities(graph, 42).values())), 2)


if __name__ == "__main__":
    unittest.main()
