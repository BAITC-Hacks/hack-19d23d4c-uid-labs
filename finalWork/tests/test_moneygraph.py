import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from moneygraph.data import from_records, aggregate, DataError, load, write_csv
from moneygraph.engine import analyze, compatible_amount, witness_certificates
from moneygraph.experiments import disclosed, benchmark
from moneygraph.pipeline import run, NODE_COLUMNS, CLUSTER_COLUMNS, TOP_COLUMNS
from moneygraph.synthetic import generate


def data(nodes, transactions):
    ns=[dict(gid=g,depth=d,is_seed=s) for g,d,s in nodes]
    ts=[dict(src=a,dst=b,date=day,sum_kzt=amount) for a,b,day,amount in transactions]
    return from_records(ns,aggregate(ts),ts,dict(synthetic=True,source_format='test'))


def basic():
    return data([(1,0,True),(2,1,False),(3,2,False),(4,0,True)],
                [(1,2,'2026-07-01',10000),(2,3,'2026-07-02',10000)])


class RulesTests(unittest.TestCase):
    def test_invalid_configuration_fails_before_calculation(self):
        for cfg in ({'centrality_sources':1.5},{'priority_weights':[float('nan')]*5},
                    {'transit_min':2,'transit_max':1},{'unknown':1}):
            with self.assertRaises(ValueError):
                analyze(basic(),cfg)

    def test_isolated_seed_survives(self):
        nodes,_=analyze(basic()); n=next(n for n in nodes if n['gid']=='4')
        self.assertEqual(n['role'],'peripheral')
        self.assertEqual(n['priority_score'],0)
        self.assertTrue(n['evidence'])
        self.assertEqual(sum(x['cluster_id']==n['cluster_id'] for x in nodes),1)

    def test_boundary_is_never_terminal(self):
        d=data([(1,0,True),(2,4,False)],[(1,2,'2026-07-01',10000)])
        n=analyze(d)[0][1]
        self.assertNotEqual(n['role'],'terminal')
        self.assertTrue(n['metrics']['boundary'])

    def test_boundary_can_still_have_consolidation_evidence(self):
        d=data([(1,0,True),(2,0,True),(3,0,True),(4,4,False)],
               [(i,4,'2026-07-01',10000) for i in (1,2,3)])
        n=analyze(d)[0][3]
        self.assertEqual(n['role'],'consolidator')

    def test_path_search_limit_cannot_change_observation_boundary(self):
        d=data([(1,0,True),(2,4,False)],[(1,2,'2026-07-01',10000)])
        for path_limit in (2,4,5,10):
            n=analyze(d,{'max_depth':path_limit})[0][1]
            self.assertTrue(n['metrics']['boundary'])
            self.assertNotEqual(n['role'],'terminal')

    def test_seed_ratio_cannot_assign_transit_or_terminal(self):
        d=data([(1,0,True),(2,0,True),(3,1,False)],
               [(1,2,'2026-07-01',10000),(2,3,'2026-07-02',10000)])
        n=next(n for n in analyze(d)[0] if n['gid']=='2')
        self.assertNotIn(n['role'],('transit','terminal'))

    def test_next_day_supports_transit(self):
        n=next(n for n in analyze(basic())[0] if n['gid']=='2')
        self.assertEqual(n['role'],'transit')
        self.assertEqual(n['metrics']['temporal_matched_kzt'],10000)

    def test_same_day_does_not_prove_order(self):
        d=data([(1,0,True),(2,1,False),(3,2,False)],
               [(1,2,'2026-07-01',10000),(2,3,'2026-07-01',10000)])
        n=analyze(d)[0][1]
        self.assertEqual(n['metrics']['temporal_matched_kzt'],0)
        self.assertEqual(n['metrics']['same_day_extra_kzt'],10000)
        self.assertNotEqual(n['role'],'transit')

    def test_out_before_in_is_not_transit(self):
        d=data([(1,0,True),(2,1,False),(3,2,False)],
               [(1,2,'2026-07-03',10000),(2,3,'2026-07-01',10000)])
        self.assertEqual(analyze(d)[0][1]['metrics']['temporal_matched_kzt'],0)

    def test_amount_capacity_is_not_reused(self):
        self.assertEqual(compatible_amount([(1,10000)],[(2,10000),(3,10000)]),10000)

    def test_cycle_does_not_multiply_seed_support(self):
        d=data([(1,0,True),(2,1,False),(3,2,False)],
               [(1,2,'2026-07-01',10000),(2,3,'2026-07-02',10000),(3,2,'2026-07-03',10000)])
        self.assertEqual(analyze(d)[0][1]['metrics']['seed_reach'],1)

    def test_boundary_certificate_is_executable(self):
        d=data([(1,0,True),(2,1,False),(3,2,False),(4,3,False),(5,4,False)],
               [(i,i+1,f'2026-07-0{i}',10000) for i in range(1,5)])
        cert=witness_certificates(d)
        self.assertEqual(len(cert),1)
        self.assertTrue(cert[0]['observed_same'])
        self.assertTrue(cert[0]['property_changes'])

    def test_artificial_boundary_does_not_leak_reverse_edges(self):
        d=data([(1,0,True),(2,1,False),(3,2,False)],
               [(1,2,'2026-07-01',10000),(2,3,'2026-07-02',10000),(3,2,'2026-07-03',10000)])
        visible=disclosed(d,{1,2})
        self.assertNotIn((3,2),{(e['src'],e['dst']) for e in visible.edges})
        self.assertTrue(visible.nodes[3]['is_boundary'])
        self.assertNotEqual(analyze(visible)[0][2]['role'],'terminal')


class DataTests(unittest.TestCase):
    def test_boolean_false_string(self):
        d=from_records([dict(gid='1',depth='0',is_seed='false')],[],[])
        self.assertFalse(d.nodes[1]['is_seed'])

    def test_duplicate_transactions_are_retained(self):
        d=data([(1,0,True),(2,1,False)],[(1,2,'2026-07-01',5000)]*2)
        self.assertEqual(len(d.transactions),2)
        self.assertEqual(d.edges[0]['n_tx'],2)
        self.assertEqual(d.edges[0]['sum_kzt'],10000)

    def test_edges_transactions_are_reconciled(self):
        d=basic(); bad=[dict(e) for e in d.edges]; bad[0]['sum_kzt']+=5000
        with self.assertRaises(DataError):
            from_records(d.nodes.values(),bad,d.transactions)

    def test_unknown_endpoint_is_error(self):
        with self.assertRaises(DataError):
            data([(1,0,True)],[(1,2,'2026-07-01',5000)])

    def test_duplicate_node_is_error(self):
        with self.assertRaises(DataError):
            data([(1,0,True),(1,0,True)],[])

    def test_invalid_money_is_error(self):
        for invalid in (0,-1,float('inf'),float('nan')):
            with self.assertRaises(DataError):
                data([(1,0,True),(2,1,False)],[(1,2,'2026-07-01',invalid)])

    def test_float_gid_is_rejected(self):
        with self.assertRaises(DataError):
            data([(1.0,0,True)],[])

    def test_int64_gid_is_exact(self):
        big=9007199254740993
        d=data([(big,0,True),(big+1,1,False)],[(big,big+1,'2026-07-01',5000)])
        self.assertEqual([n['gid'] for n in analyze(d)[0]],[str(big),str(big+1)])

    def test_change_in_input_changes_metrics(self):
        a=basic(); b=basic(); b.transactions[0]['sum_kzt']=20000
        b=from_records(b.nodes.values(),aggregate(b.transactions),b.transactions)
        self.assertNotEqual(analyze(a)[0][1]['metrics']['in_kzt'],analyze(b)[0][1]['metrics']['in_kzt'])

    @unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'pyarrow not installed; Parquet roundtrip unverified here')
    def test_parquet_roundtrip(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as folder:
            d=basic()
            for name,rows in [('nodes',list(d.nodes.values())),('edges',d.edges),('transactions',d.transactions)]:
                pq.write_table(pa.Table.from_pylist(rows),Path(folder)/f'{name}.parquet')
            result=load(folder,'parquet')
            self.assertEqual(result.nodes,d.nodes)
            self.assertEqual(len(result.transactions),len(d.transactions))


class EndToEndTests(unittest.TestCase):
    def test_exports_and_repeatability(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); generate(base/'input',80)
            d=load(base/'input','csv')
            report=run(d,base/'a',stability=True)
            run(d,base/'b',stability=True)
            for filename,columns in [('nodes_roles.csv',NODE_COLUMNS),('clusters.csv',CLUSTER_COLUMNS),('top_nodes.csv',TOP_COLUMNS)]:
                with (base/'a'/filename).open(encoding='utf-8-sig',newline='') as h:
                    reader=csv.DictReader(h); rows=list(reader)
                self.assertEqual(reader.fieldnames,columns)
                self.assertEqual((base/'a'/filename).read_bytes(),(base/'b'/filename).read_bytes())
                self.assertGreater(len(rows),0)
            self.assertEqual(len(report['nodes']),80)
            self.assertEqual(len(report['top_nodes']),20)
            self.assertTrue(report['meta']['synthetic'])
            scores=[n['priority_score'] for n in report['top_nodes']]
            self.assertEqual(scores,sorted(scores,reverse=True))
            self.assertTrue(all(0<len(n['evidence'])<=200 for n in report['nodes']))
            self.assertTrue(all(0<=n['role_score']<=1 for n in report['nodes']))
            html=(base/'a'/'dashboard.html').read_text(encoding='utf-8')
            self.assertNotIn('__REPORT_JSON__',html)
            self.assertIn('application/json',html)

    def test_cluster_internal_volume_counts_each_edge_once(self):
        with tempfile.TemporaryDirectory() as folder:
            d=basic(); report=run(d,folder)
            mapping={int(n['gid']):n['cluster_id'] for n in report['nodes']}
            expected=sum(e['sum_kzt'] for e in d.edges if mapping[e['src']]==mapping[e['dst']])
            self.assertAlmostEqual(sum(c['sum_kzt_internal'] for c in report['clusters']),expected)
            self.assertEqual(report['meta']['total_kzt'],20000)

    def test_embedded_json_cannot_close_script(self):
        with tempfile.TemporaryDirectory() as folder:
            d=basic(); d.warnings.append('</script><script>alert(1)</script>')
            run(d,folder)
            html=(Path(folder)/'dashboard.html').read_text(encoding='utf-8')
            self.assertNotIn('</script><script>alert(1)</script>',html)
            self.assertIn('\\u003c/script>',html)

    def test_benchmark_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); generate(root/'input',50)
            rows=benchmark(load(root/'input','csv'),root/'benchmark',budget=2,random_runs=1)
            self.assertEqual(len(rows),8)
            self.assertTrue(all(0<=r['final_recall']<=1 for r in rows))


if __name__=='__main__':
    unittest.main()
