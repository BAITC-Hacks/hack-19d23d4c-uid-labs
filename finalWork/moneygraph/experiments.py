"""Controlled graph disclosure; the full FILE is a reference, not criminal truth."""
from collections import deque
from pathlib import Path
import json
import random
from .data import aggregate, from_records, write_csv
from .engine import analyze, adjacency
from .pipeline import json_write


def distances(seeds, outgoing):
    found={g:0 for g in seeds}; queue=deque(seeds)
    while queue:
        g=queue.popleft()
        for other in outgoing.get(g,{}):
            if other not in found:
                found[other]=found[g]+1; queue.append(other)
    return found


def disclosed(full, expanded):
    seeds={g for g,n in full.nodes.items() if n['is_seed']}
    tx=[t for t in full.transactions if t['src'] in expanded]
    visible=seeds | {t[key] for t in tx for key in ('src','dst')}
    edges=aggregate(tx)
    adj={g:{} for g in visible}
    for e in edges:
        adj[e['src']][e['dst']]=e['sum_kzt']
    depth=distances(seeds,adj)
    nodes=[dict(gid=g,depth=depth.get(g,4),is_seed=g in seeds) for g in sorted(visible)]
    data=from_records(nodes,edges,tx,dict(synthetic=full.meta.get('synthetic',False),source_format='controlled-disclosure'))
    for g,n in data.nodes.items():
        n['is_boundary']=g not in expanded
    return data


def benchmark(full,out,budget=10,random_runs=3):
    if budget<1 or budget>100 or random_runs<1:
        raise ValueError('budget must be 1..100 and random_runs >= 1')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    # Same settings for reference and every partial view; no fitting on hidden answers.
    cfg=dict(centrality_sources=16,community_iterations=8)
    reference,_=analyze(full,cfg)
    ref_top=sorted(reference,key=lambda n:(-n['priority_score'],int(n['gid'])))[:20]
    target={n['gid'] for n in ref_top}
    ref_metrics={n['gid']:n['metrics'] for n in ref_top}
    seeds={g for g,n in full.nodes.items() if n['is_seed']}
    full_out,_=adjacency(full)
    full_depth=distances(seeds,full_out)
    all_rows=[]
    strategies=[('priority',0),('amount',0),('degree',0)]+[('random',r) for r in range(random_runs)]
    for cutoff in (2,3):
        initial={g for g,d in full_depth.items() if d<cutoff}
        for strategy,run in strategies:
            rng=random.Random(1000+run)
            expanded=set(initial)
            covered_seeds=set()
            for step in range(budget+1):
                current=disclosed(full,expanded)
                nodes,_=analyze(current,cfg)
                ranked=sorted(nodes,key=lambda n:(-n['priority_score'],int(n['gid'])))
                recovered={n['gid'] for n in ranked[:20]}
                lookup={n['gid']:n for n in nodes}
                error=sum(abs(lookup.get(g,{}).get('metrics',{}).get('in_kzt',0)-ref_metrics[g]['in_kzt'])
                          for g in target)/max(1,sum(ref_metrics[g]['in_kzt'] for g in target))
                row=dict(cutoff=cutoff,strategy=strategy,run=run,queries=step,
                         recall_at_20=len(target & recovered)/max(1,len(target)),
                         reference_inflow_relative_error=error,n_observed_nodes=len(nodes),
                         n_observed_transactions=len(current.transactions),query_gid='')
                all_rows.append(row)
                if step==budget:
                    break
                # No peeking at hidden degree, amount, labels or original node depth.
                candidates=[n for n in nodes if int(n['gid']) not in expanded and n['depth']<4]
                if not candidates:
                    break
                if strategy=='random':
                    chosen=rng.choice(sorted(candidates,key=lambda n:int(n['gid'])))
                else:
                    def score(n):
                        if strategy=='amount': return n['metrics']['in_kzt']
                        if strategy=='degree': return n['metrics']['in_degree']
                        new=len(set(n['seed_ids'])-covered_seeds)/max(1,len(n['seed_ids']))
                        return n['priority_score']*(.8+.2*new)
                    chosen=min(candidates,key=lambda n:(-score(n),int(n['gid'])))
                row['query_gid']=chosen['gid']
                covered_seeds.update(chosen['seed_ids'])
                expanded.add(int(chosen['gid']))
    columns=['cutoff','strategy','run','queries','recall_at_20','reference_inflow_relative_error',
             'n_observed_nodes','n_observed_transactions','query_gid']
    write_csv(out/'disclosure_benchmark.csv',all_rows,columns)
    summaries=[]
    for cutoff in (2,3):
        for strategy,run in strategies:
            rows=[r for r in all_rows if r['cutoff']==cutoff and r['strategy']==strategy and r['run']==run]
            hit=next((r['queries'] for r in rows if r['recall_at_20']>=.9),None)
            curve=[r['recall_at_20'] for r in rows]
            curve += [curve[-1]]*(budget+1-len(curve))
            summaries.append(dict(cutoff=cutoff,strategy=strategy,run=run,
                                  queries_to_90pct=hit,final_recall=rows[-1]['recall_at_20'],
                                  normalized_auc=sum((a+b)/2 for a,b in zip(curve,curve[1:]))/budget))
    json_write(out/'benchmark_summary.json',dict(results=summaries,config=cfg,budget=budget,
               reference='Full supplied file, not actual criminal roles or external bank transfers',
               limitations=['Two cutoffs of one dataset are not independent populations.',
                            'Random repetitions describe baseline variability only.',
                            'Unknown fourth-hop outgoing transfers are never treated as known empty answers.',
                            'Priority is a fixed heuristic, not calibrated expected information gain.']))
    return summaries
