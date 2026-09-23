import json
import unittest

from moneygraph.assistant_tools import GraphTools, priority_band
from moneygraph.data import aggregate, from_records
from moneygraph.engine import analyze, cluster_rows


A = 100000000331309100
B = A + 1
C = A + 2
D = A + 3
ISOLATED = A + 4


def fixture():
    records = [dict(gid=g, depth=d, is_seed=s) for g, d, s in
               [(A, 0, True), (B, 1, False), (C, 4, False), (D, 1, False), (ISOLATED, 0, True)]]
    transactions = [
        dict(src=A, dst=B, date="2026-07-02", sum_kzt=10000.11),
        dict(src=A, dst=B, date="2026-07-02", sum_kzt=10000.11),
        dict(src=B, dst=C, date="2026-07-03", sum_kzt=15000.12),
        dict(src=A, dst=D, date="2026-07-01", sum_kzt=5000.13),
        dict(src=D, dst=C, date="2026-07-04", sum_kzt=6000.14),
    ]
    data = from_records(records, aggregate(transactions), transactions, {"synthetic": True})
    nodes, _ = analyze(data)
    for index, node in enumerate(nodes):
        node["priority_score"] = [0.7, 0.69, 0.4, 0.39, 0][index]
    report = dict(nodes=nodes, clusters=cluster_rows(data, nodes))
    return data, report


class AssistantToolsTests(unittest.TestCase):
    def setUp(self):
        self.data, self.report = fixture()
        self.tools = GraphTools(self.data, self.report)

    def call(self, name, **arguments):
        result = self.tools.call(name, arguments)
        self.assertTrue(result["ok"], result)
        json.dumps(result, allow_nan=False)
        return result

    def test_strict_definitions_and_copy(self):
        definitions = self.tools.definitions
        self.assertEqual(len(definitions), 7)
        for definition in definitions:
            self.assertTrue(definition["strict"])
            schema = definition["parameters"]
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))
        definitions[0]["name"] = "changed"
        self.assertEqual(self.tools.definitions[0]["name"], "overview")

    def test_overview_amount_not_double_counted_with_edges(self):
        result = self.call("overview")
        self.assertEqual(result["n_nodes"], 5)
        self.assertEqual(result["n_transactions"], 5)
        self.assertEqual(result["n_edges"], 4)
        self.assertEqual(result["total_kzt"], 46000.61)
        self.assertEqual(result["date_from"], "2026-07-01")
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["priority_counts"], {"high": 1, "medium": 2, "low": 2})

    def test_exact_identifiers_and_repeated_rows_kept(self):
        result = self.call("transactions", gid=str(A), direction="out", limit=10)
        self.assertEqual(result["matched_count"], 3)
        self.assertEqual(result["matched_total_kzt"], 25000.35)
        self.assertEqual([row["source_id"] for row in result["rows"]], ["tx:4", "tx:1", "tx:2"])
        self.assertTrue(all(row["src"] == str(A) for row in result["rows"]))
        self.assertEqual([row["dst"] for row in result["rows"]][1:], [str(B), str(B)])
        self.assertNotEqual(str(A), str(B))

    def test_transaction_filters_and_paging_keep_input_row_ids(self):
        result = self.call("transactions", date_from="2026-07-02", date_to="2026-07-03",
                           min_amount=10000, limit=1, offset=1)
        self.assertEqual(result["matched_count"], 3)
        self.assertEqual(result["matched_total_kzt"], 35000.34)
        self.assertEqual(result["rows"][0]["source_id"], "tx:2")
        self.assertEqual(result["next_offset"], 2)
        result = self.call("transactions", gid=str(C), direction="in", min_amount=10000)
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["rows"][0]["src"], str(B))

    def test_all_input_rows_available_beyond_report_preview(self):
        tx = [dict(src=A, dst=B, date="2026-07-01", sum_kzt=5000.01) for _ in range(130)]
        data = from_records(list(self.data.nodes.values()), aggregate(tx), tx)
        nodes, _ = analyze(data)
        self.assertEqual(len(next(n for n in nodes if n["gid"] == str(A))["transactions"]), 100)
        tools = GraphTools(data, {"nodes": nodes, "clusters": cluster_rows(data, nodes)})
        result = tools.call("transactions", {"gid": str(A), "offset": 120, "limit": 20})
        self.assertEqual(result["matched_count"], 130)
        self.assertEqual(result["matched_total_kzt"], 650001.3)
        self.assertEqual(len(result["rows"]), 10)
        self.assertEqual(result["rows"][0]["source_id"], "tx:121")
        self.assertIsNone(result["next_offset"])
        details = tools.call("node_details", {"gid": str(A)})
        self.assertEqual(details["observed"]["n_transactions"], 130)
        self.assertTrue(details["transactions_truncated"])

    def test_rank_bands_and_role_filters(self):
        self.assertEqual(priority_band(.7), "high")
        self.assertEqual(priority_band(.4), "medium")
        result = self.call("rank_nodes", priority="medium", limit=1)
        self.assertEqual(result["matched_count"], 2)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["nodes"][0]["gid"], str(B))
        role = self.report["nodes"][0]["role"]
        result = self.call("rank_nodes", role=role)
        self.assertTrue(all(node["role_hypothesis"] == role for node in result["nodes"]))

    def test_node_boundary_and_isolated_node_are_hypotheses(self):
        result = self.call("node_details", gid=str(C))
        self.assertNotEqual(result["role_hypothesis"], "terminal")
        self.assertTrue(result["metrics"]["boundary"])
        self.assertEqual(result["observed"]["incoming_kzt"], 21000.26)
        self.assertIn("границ", " ".join(result["limitations"]))
        isolated = self.call("node_details", gid=str(ISOLATED))
        self.assertEqual(isolated["observed"]["n_transactions"], 0)
        self.assertEqual(isolated["priority"], "low")

    def test_connections_shortest_direction_and_shared_neighbors(self):
        result = self.call("connections", gid=str(A), other_gid=str(C))
        self.assertTrue(result["path_found"])
        self.assertEqual(result["shortest_hops"], 2)
        self.assertEqual(result["path"], [str(A), str(B), str(C)])
        self.assertEqual(result["path_edges"][0]["transaction_source_ids"], ["tx:1", "tx:2"])
        reverse = self.call("connections", gid=str(C), other_gid=str(A))
        self.assertFalse(reverse["path_found"])
        self.assertIsNone(reverse["path"])
        common = self.call("connections", gid=str(B), other_gid=str(D))
        self.assertEqual(common["common_senders"], [str(A)])
        self.assertEqual(common["common_recipients"], [str(C)])
        identity = self.call("connections", gid=str(B), other_gid=str(B))
        self.assertEqual(identity["shortest_hops"], 0)

    def test_frequency_threshold_and_singleton_pairs(self):
        result = self.call("recurring_patterns", min_transactions=2)
        self.assertEqual(result["frequent_pairs_count"], 1)
        self.assertEqual(result["rare_pairs_count"], 3)
        self.assertEqual(result["frequent"][0]["n_transactions"], 2)
        self.assertEqual(result["frequent"][0]["sum_kzt"], 20000.22)
        result = self.call("recurring_patterns", gid=str(A), min_transactions=3)
        self.assertEqual(result["total_pairs"], 2)
        self.assertEqual(result["frequent_pairs_count"], 0)
        self.assertEqual(result["rare_pairs_count"], 1)

    def test_cluster_sums_each_transaction_once(self):
        for node in self.report["nodes"]:
            node["cluster_id"] = 0 if node["gid"] in (str(A), str(B)) else 1
        self.report["clusters"] = [{"cluster_id": 0, "hypothesis": "test"}, {"cluster_id": 1}]
        tools = GraphTools(self.data, self.report)
        result = tools.call("cluster_details", {"cluster_id": 0})
        self.assertEqual(result["internal_n_transactions"], 2)
        self.assertEqual(result["internal_kzt"], 20000.22)
        self.assertEqual(result["external_incoming_kzt"], 0)
        self.assertEqual(result["external_outgoing_kzt"], 20000.25)
        self.assertEqual(result["internal_transaction_sources"], ["tx:1", "tx:2"])

    def test_unsafe_or_invalid_arguments_return_structured_errors(self):
        cases = [("node_details", {"gid": A}),
                 ("node_details", {}),
                 ("node_details", {"gid": str(2**63)}),
                 ("node_details", {"gid": "01"}),
                 ("transactions", {"limit": 51}),
                 ("transactions", {"limit": True}),
                 ("transactions", {"min_amount": float("nan")}),
                 ("transactions", {"offset": -1}),
                 ("transactions", {"date_from": "2026-02-30"}),
                 ("transactions", {"date_from": "2026-07-03", "date_to": "2026-07-01"}),
                 ("transactions", {"direction": "in"}),
                 ("transactions", {"sql": "drop table nodes"}),
                 ("rank_nodes", {"role": "criminal"}),
                 ("recurring_patterns", {"min_transactions": 1}),
                 ("cluster_details", {"cluster_id": False})]
        for name, args in cases:
            with self.subTest(name=name, args=args):
                result = self.tools.call(name, args)
                self.assertFalse(result["ok"])
                self.assertEqual(result["error"]["code"], "invalid_arguments")
        self.assertEqual(self.tools.call("node_details", {"gid": "999"})["error"]["code"], "not_found")
        self.assertEqual(self.tools.call("cluster_details", {"cluster_id": 10**1000})["error"]["code"], "not_found")
        self.assertEqual(self.tools.call("execute_code", {})["error"]["code"], "unknown_tool")
        self.assertFalse(self.tools.call([], {})["ok"])
        self.assertFalse(self.tools.call("overview", [])["ok"])

    def test_empty_dataset_transaction_lists(self):
        data = from_records([dict(gid=A, depth=0, is_seed=True)], [], [])
        nodes, _ = analyze(data)
        tools = GraphTools(data, {"nodes": nodes, "clusters": cluster_rows(data, nodes)})
        result = tools.call("overview", {})
        self.assertEqual(result["total_kzt"], 0)
        self.assertIsNone(result["date_from"])
        self.assertEqual(tools.call("recurring_patterns", {})["rare"], [])
        self.assertEqual(tools.call("connections", {"gid": str(A)})["incoming"], [])


if __name__ == "__main__":
    unittest.main()
