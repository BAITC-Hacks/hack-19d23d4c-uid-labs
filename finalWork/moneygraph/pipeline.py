from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import time
from . import __version__
from .data import write_csv
from .engine import analyze, cluster_rows, next_queries, witness_certificates, roles

NODE_COLUMNS=['gid','role','role_score','cluster_id','priority_score','evidence']
CLUSTER_COLUMNS=['cluster_id','n_nodes','n_seed','sum_kzt_internal','top_gids','hypothesis']
TOP_COLUMNS=['rank','gid','role','priority_score','why']


def json_write(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sensitivity(nodes,cfg):
    rows=[]
    for node in nodes:
        assignments=[]
        for factor in (.8,1.,1.2):
            changed=dict(cfg)
            changed['consolidator_in']=max(2,round(cfg['consolidator_in']*factor))
            changed['distributor_out']=max(2,round(cfg['distributor_out']*factor))
            changed['transit_temporal']=min(.95,cfg['transit_temporal']*factor)
            changed['coordinator_seeds']=max(2,round(cfg['coordinator_seeds']*factor))
            assignments.append(roles(node['metrics'],node,changed)[0][0])
        rows.append(dict(gid=node['gid'],base_role=node['role'],
                         role_retention=sum(r==node['role'] for r in assignments)/len(assignments),
                         scenarios=';'.join(assignments),
                         interpretation='Чувствительность к трём наборам порогов; не вероятность роли.'))
    return rows


def run(data,out,top=20,stability=False,config=None,started=None):
    start=started or time.perf_counter()
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    nodes,cfg=analyze(data,config)
    clusters=cluster_rows(data,nodes)
    ranked=sorted(nodes,key=lambda n:(-n['priority_score'],int(n['gid'])))
    top_rows=[dict(rank=i,gid=n['gid'],role=n['role'],priority_score=n['priority_score'],
                   why=n['evidence']+'; приоритет: охват seed, оборот, центральность, признаки роли и плательщики.')
              for i,n in enumerate(ranked[:max(20,top)],1)]
    queries=next_queries(nodes,max(20,top))
    witnesses=witness_certificates(data)
    warnings=list(data.warnings)
    warnings += ['Роли и приоритеты — гипотезы для проверки; оценки не являются вероятностями виновности.',
                 'Оборот графа включает несколько звеньев: это не размер преступного дохода.',
                 'Пути от seed структурные; временная связь конкретных денег по ним не доказана.',
                 'Сила роли, приоритет проверки и чувствительность порогов — разные показатели.']
    if data.meta.get('synthetic'):
        warnings.insert(0,'СИНТЕТИЧЕСКИЕ ДАННЫЕ. Это инженерная демонстрация, не результаты реального кейса.')
    boundary=sum(n['metrics']['boundary'] for n in nodes)
    isolated=sum(not n['metrics']['n_tx'] for n in nodes)
    neighbors={gid:set() for gid in data.nodes}
    for edge in data.edges:
        neighbors[edge['src']].add(edge['dst']); neighbors[edge['dst']].add(edge['src'])
    unseen=set(neighbors); components=0
    while unseen:
        stack=[unseen.pop()]; components+=1
        while stack:
            new=neighbors[stack.pop()] & unseen
            unseen.difference_update(new); stack.extend(new)
    if boundary:
        warnings.append(f'Узлов на границе наблюдения: {boundary}; отсутствие исходящих не доказывает конечное получение.')
    dates=[t['date'] for t in data.transactions]
    meta=dict(data.meta,n_nodes=len(nodes),n_edges=len(data.edges),n_transactions=len(data.transactions),
              total_kzt=round(sum(t['sum_kzt'] for t in data.transactions),2),
              n_seed=sum(n['is_seed'] for n in nodes),n_clusters=len(clusters),n_isolates=isolated,
              n_weak_components=components,n_weak_components_with_edges=components-isolated,
              date_from=min(dates) if dates else None,date_to=max(dates) if dates else None,
              runtime_seconds=round(time.perf_counter()-start,4),version=__version__,
              transaction_preview_limit=100,
              centrality_method=f"Directed unweighted Brandes, at most {cfg['centrality_sources']} sampled active sources")
    report=dict(meta=meta,warnings=warnings,thresholds=cfg,nodes=nodes,
                edges=[dict(e,src=str(e['src']),dst=str(e['dst'])) for e in data.edges],
                clusters=clusters,top_nodes=top_rows,inquiries=queries,boundary_witnesses=witnesses)
    validation=dict(valid=True,n_nodes=len(nodes),n_edges=len(data.edges),n_transactions=len(data.transactions),
                    all_nodes_preserved=len(nodes)==len(data.nodes),
                    edges_reconcile_transactions=True,
                    all_evidence_valid=all(0<len(n['evidence'])<=200 for n in nodes),
                    boundary_terminal_count=sum(n['role']=='terminal' and n['metrics']['boundary'] for n in nodes),
                    scores_finite_and_in_range=all(0<=n['role_score']<=1 and 0<=n['priority_score']<=1 for n in nodes),
                    role_counts=dict(Counter(n['role'] for n in nodes)),
                    warnings=warnings,real_case_accuracy='not available: no ground-truth roles')
    write_csv(out/'nodes_roles.csv',nodes,NODE_COLUMNS)
    write_csv(out/'clusters.csv',clusters,CLUSTER_COLUMNS)
    write_csv(out/'top_nodes.csv',top_rows,TOP_COLUMNS)
    features=[dict(gid=n['gid'],depth=n['depth'],is_seed=n['is_seed'],**n['metrics']) for n in nodes]
    write_csv(out/'features.csv',features,list(features[0]))
    write_csv(out/'next_queries.csv',queries,['gid','reason','request'])
    if stability:
        write_csv(out/'stability.csv',sensitivity(nodes,cfg),['gid','base_role','role_retention','scenarios','interpretation'])
    elif (out/'stability.csv').exists():
        (out/'stability.csv').unlink()
    template=Path(__file__).with_name('dashboard.html').read_text(encoding='utf-8')
    # Escaping '<' prevents JSON values from closing the data script tag.
    def html_write():
        embedded=json.dumps(report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
        (out/'dashboard.html').write_text(template.replace('__REPORT_JSON__',embedded),encoding='utf-8')
    json_write(out/'report.json',report)
    html_write()
    json_write(out/'validation.json',validation)
    meta['runtime_seconds']=round(time.perf_counter()-start,4)
    json_write(out/'report.json',report)
    html_write()
    generated=['nodes_roles.csv','clusters.csv','top_nodes.csv','features.csv','next_queries.csv',
               'dashboard.html','report.json','validation.json']+(['stability.csv'] if stability else [])
    hashes={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in generated}
    manifest=dict(version=__version__,python=platform.python_version(),platform=platform.platform(),
                  generated_at=datetime.now(timezone.utc).isoformat(),synthetic=meta.get('synthetic',False),
                  input_sha256=meta.get('input_sha256',{}),output_sha256=hashes,config=cfg,
                  runtime_seconds=round(time.perf_counter()-start,4),
                  runtime_scope='Reading, analysis, exports, local HTML and hashes; excludes final manifest write')
    json_write(out/'run_manifest.json',manifest)
    return report
