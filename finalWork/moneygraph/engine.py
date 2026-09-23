"""Deterministic, explainable graph heuristics; scores are not probabilities."""
from bisect import bisect_right
from collections import defaultdict, deque
from datetime import date
import math
import random

# Fixed by the case's extraction protocol, independently of path-search limits.
OBSERVATION_DEPTH = 4

DEFAULTS = dict(consolidator_in=3, distributor_out=5, coordinator_seeds=3,
                coordinator_external_clusters=2, coordinator_centrality=0.90,
                transit_min=0.8, transit_max=1.2, transit_temporal=0.5,
                terminal_ratio=0.1, max_depth=4, temporal_days=2,
                community_resolution=1.0, community_iterations=15,
                centrality_sources=64, random_seed=42,
                priority_weights=[0.35, 0.20, 0.20, 0.15, 0.10])

ROLE_LABELS = dict(consolidator='Признаки консолидации', transit='Признаки транзита',
                   distributor='Признаки распределения', terminal='Кандидат в конечные получатели',
                   coordinator='Кандидат на координирующую роль', peripheral='Недостаточно признаков роли')


def clamp(value):
    return max(0.0, min(1.0, value))


def percentile(values):
    positive = sorted(v for v in values.values() if v > 0)
    return {k: bisect_right(positive, v) / len(positive) if v > 0 and positive else 0.0
            for k, v in values.items()}


def adjacency(data):
    outgoing = {gid: {} for gid in data.nodes}
    incoming = {gid: {} for gid in data.nodes}
    for edge in data.edges:
        outgoing[edge['src']][edge['dst']] = edge['sum_kzt']
        incoming[edge['dst']][edge['src']] = edge['sum_kzt']
    return outgoing, incoming


def communities(data, cfg):
    """One-level modularity local optimization on a log-weighted undirected graph.

    This is not a claim of full hierarchical Louvain or ground-truth group recovery.
    """
    neighbors = {gid: defaultdict(float) for gid in data.nodes}
    for edge in data.edges:
        a, b = edge['src'], edge['dst']
        if a != b:
            weight = math.log1p(edge['sum_kzt'])
            neighbors[a][b] += weight
            neighbors[b][a] += weight
    degree = {gid: sum(n.values()) for gid, n in neighbors.items()}
    total = sum(degree.values())
    labels = {gid: gid for gid in data.nodes}
    totals = dict(degree)
    rng = random.Random(cfg['random_seed'])
    order = sorted(data.nodes)
    if total:
        for _ in range(cfg['community_iterations']):
            rng.shuffle(order)
            changed = 0
            for gid in order:
                if not degree[gid]:
                    continue
                old = labels[gid]
                totals[old] -= degree[gid]
                weights = defaultdict(float)
                for other, weight in neighbors[gid].items():
                    weights[labels[other]] += weight
                candidates = set(weights) | {old}
                scores = {c: weights[c] - cfg['community_resolution'] * degree[gid] * totals[c] / total
                          for c in candidates}
                best = min(candidates, key=lambda c: (-scores[c], c))
                if scores[best] <= scores[old] + 1e-12:
                    best = old
                labels[gid] = best
                totals[best] += degree[gid]
                changed += best != old
            if not changed:
                break
    # Split any disconnected remainder of a moved community, preserve isolates.
    groups, seen = [], set()
    for start in sorted(data.nodes):
        if start in seen:
            continue
        group, queue = [], [start]
        seen.add(start)
        while queue:
            gid = queue.pop()
            group.append(gid)
            for other in neighbors[gid]:
                if other not in seen and labels[other] == labels[start]:
                    seen.add(other)
                    queue.append(other)
        groups.append(sorted(group))
    groups.sort(key=lambda group: group[0])
    return {gid: cid for cid, group in enumerate(groups) for gid in group}


def centrality(outgoing, cfg):
    """Directed unweighted Brandes over a reproducible sample of sources."""
    nodes = sorted(outgoing)
    active = [gid for gid in nodes if outgoing[gid]]
    sources = random.Random(cfg['random_seed']).sample(active, min(len(active), cfg['centrality_sources']))
    result = dict.fromkeys(nodes, 0.0)
    for source in sources:
        pred = defaultdict(list)
        sigma, distance = {source: 1.0}, {source: 0}
        stack, queue = [], deque([source])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in sorted(outgoing[v]):
                if w not in distance:
                    distance[w] = distance[v] + 1
                    queue.append(w)
                if distance[w] == distance[v] + 1:
                    sigma[w] = sigma.get(w, 0.0) + sigma[v]
                    pred[w].append(v)
        delta = defaultdict(float)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != source:
                result[w] += delta[w]
    return result


def seed_paths(data, outgoing, max_depth):
    support = {gid: set() for gid in data.nodes}
    samples = {gid: [] for gid in data.nodes}
    for seed in sorted(gid for gid, n in data.nodes.items() if n['is_seed']):
        paths = {seed: [seed]}
        queue = deque([seed])
        while queue:
            gid = queue.popleft()
            if len(paths[gid]) - 1 >= max_depth:
                continue
            for other in sorted(outgoing[gid]):
                if other in paths:
                    continue
                path = paths[gid] + [other]
                paths[other] = path
                queue.append(other)
                support[other].add(seed)
                if len(samples[other]) < 3:
                    samples[other].append(path)
    return support, samples


def compatible_amount(incoming, outgoing, max_days=2, include_same_day=False):
    """FIFO capacity matching; one input amount is never reused within this calculation."""
    pools = [[d, amount] for d, amount in sorted(incoming)]
    matched = 0.0
    minimum = 0 if include_same_day else 1
    for out_day, amount in sorted(outgoing):
        remaining = amount
        for pool in pools:
            difference = out_day - pool[0]
            if minimum <= difference <= max_days and pool[1] > 0:
                used = min(remaining, pool[1])
                matched += used
                pool[1] -= used
                remaining -= used
                if remaining <= 1e-9:
                    break
    return matched


def roles(metric, node, cfg):
    m = metric
    scores = {}
    if m['in_degree'] >= cfg['consolidator_in']:
        scores['consolidator'] = .45 + .25 * min(m['in_degree'] / 8, 1) + .15 * m['seed_norm'] + .15 * m['in_percentile']
    if m['out_degree'] >= cfg['distributor_out']:
        scores['distributor'] = .45 + .30 * min(m['out_degree'] / 20, 1) + .15 * m['out_entropy'] + .10 * m['out_percentile']
    ratio = m['pass_ratio']
    if not node['is_seed'] and not m['boundary'] and ratio is not None:
        if cfg['transit_min'] <= ratio <= cfg['transit_max'] and m['temporal_ratio'] >= cfg['transit_temporal']:
            balance = 1 - min(abs(ratio - 1) / max(1 - cfg['transit_min'], cfg['transit_max'] - 1, .01), 1)
            scores['transit'] = .45 + .30 * m['temporal_ratio'] + .15 * balance + .10 * min(m['n_tx'] / 5, 1)
        if ratio <= cfg['terminal_ratio']:
            scores['terminal'] = .35 + .25 * min(m['in_degree'] / 5, 1) + .20 * (not m['right_censored']) + .10 * min(m['n_tx'] / 4, 1)
    if (m['seed_reach'] >= cfg['coordinator_seeds'] and
            m['external_clusters'] >= cfg['coordinator_external_clusters'] and
            m['centrality'] > 0 and m['centrality_percentile'] >= cfg['coordinator_centrality']):
        scores['coordinator'] = .45 + .25 * m['seed_norm'] + .20 * m['centrality_percentile'] + .10 * min(m['external_clusters'] / 3, 1)
    if not scores:
        scores['peripheral'] = .1 if m['n_tx'] == 0 else .2
    return sorted(((role, round(clamp(score), 6)) for role, score in scores.items()),
                  key=lambda item: (-item[1], item[0]))


def explain(role, m, seed, temporal_days=2):
    facts = {
        'consolidator': f"{m['in_degree']} плательщиков; охват {m['seed_reach']} seed; вход {m['in_kzt']:.0f} KZT",
        'distributor': f"{m['out_degree']} получателей; выход {m['out_kzt']:.0f} KZT; энтропия {m['out_entropy']:.2f}",
        'transit': f"Выход/вход {m['pass_ratio']:.2f}; временно совместимо за 1–{temporal_days} дн. {m['temporal_ratio']:.0%}" if m['pass_ratio'] is not None else '',
        'terminal': f"В наблюдаемом срезе выход/вход {m['pass_ratio']:.2f}; требуется проверка полноты" if m['pass_ratio'] is not None else '',
        'coordinator': f"Охват {m['seed_reach']} seed; внешних кластеров {m['external_clusters']}; высокая структурная центральность",
        'peripheral': 'Нет наблюдаемых переводов' if not m['n_tx'] else 'Недостаточно признаков формальных правил ролей',
    }
    note = '; граница наблюдения' if m['boundary'] else '; вход seed неполон' if seed else ''
    return (ROLE_LABELS[role] + ': ' + facts[role] + note)[:200]


def analyze(data, config=None):
    cfg = dict(DEFAULTS)
    if set(config or {}) - set(DEFAULTS):
        raise ValueError('Неизвестные параметры конфигурации')
    cfg.update(config or {})
    integer_keys=('consolidator_in','distributor_out','coordinator_seeds',
                  'coordinator_external_clusters','max_depth','temporal_days',
                  'community_iterations','centrality_sources','random_seed')
    for key in integer_keys:
        if type(cfg[key]) is not int or (cfg[key]<1 and key!='random_seed'):
            raise ValueError(f'{key}: ожидается положительное целое (random_seed может быть любым целым)')
    for key in ('coordinator_centrality','transit_min','transit_max','transit_temporal',
                'terminal_ratio','community_resolution'):
        if not isinstance(cfg[key],(int,float)) or not math.isfinite(cfg[key]) or cfg[key]<0:
            raise ValueError(f'{key}: ожидается конечное неотрицательное число')
    if cfg['transit_min']>cfg['transit_max']:
        raise ValueError('transit_min должен быть <= transit_max')
    for key in ('coordinator_centrality','transit_temporal','terminal_ratio'):
        if cfg[key]>1:
            raise ValueError(f'{key}: ожидается значение 0..1')
    weights=cfg['priority_weights']
    if not isinstance(weights,list) or len(weights)!=5 or any(not isinstance(w,(int,float)) or not math.isfinite(w) or w<0 for w in weights) or sum(weights)<=0:
        raise ValueError('priority_weights: пять конечных неотрицательных весов, сумма > 0')
    outgoing, incoming = adjacency(data)
    cluster = communities(data, cfg)
    central = centrality(outgoing, cfg)
    central_rank = percentile(central)
    support, paths = seed_paths(data, outgoing, cfg['max_depth'])
    in_tx, out_tx, transactions = defaultdict(list), defaultdict(list), defaultdict(list)
    last_day = max((date.fromisoformat(t['date']).toordinal() for t in data.transactions), default=0)
    for tx in data.transactions:
        ordinal = date.fromisoformat(tx['date']).toordinal()
        # Self-transfers are recorded but do not support pass-through matching.
        if tx['src'] != tx['dst']:
            in_tx[tx['dst']].append((ordinal, tx['sum_kzt']))
            out_tx[tx['src']].append((ordinal, tx['sum_kzt']))
        transactions[tx['src']].append(tx)
        if tx['src'] != tx['dst']:
            transactions[tx['dst']].append(tx)
    in_totals = {g: sum(incoming[g].values()) for g in data.nodes}
    out_totals = {g: sum(outgoing[g].values()) for g in data.nodes}
    in_rank, out_rank = percentile(in_totals), percentile(out_totals)
    volume_rank = percentile({g: in_totals[g] + out_totals[g] for g in data.nodes})
    degree_rank = percentile({g: len(set(incoming[g]) - {g}) for g in data.nodes})
    max_support = max((len(s) for s in support.values()), default=1) or 1
    nodes = []
    for gid, node in sorted(data.nodes.items()):
        inc, out = in_totals[gid], out_totals[gid]
        distinct_in, distinct_out = set(incoming[gid]) - {gid}, set(outgoing[gid]) - {gid}
        strict = compatible_amount(in_tx[gid], out_tx[gid], cfg['temporal_days'])
        possible = compatible_amount(in_tx[gid], out_tx[gid], cfg['temporal_days'], True)
        entropy = 0.0
        shares = [outgoing[gid][g] for g in distinct_out]
        if len(shares) > 1:
            total = sum(shares)
            entropy = -sum((w / total) * math.log(w / total) for w in shares) / math.log(len(shares))
        neighbors = distinct_in | distinct_out
        m = dict(in_degree=len(distinct_in), out_degree=len(distinct_out),
                 in_kzt=inc, out_kzt=out, observed_net_kzt=inc-out,
                 pass_ratio=out/inc if inc else None, seed_reach=len(support[gid]),
                 seed_norm=len(support[gid])/max_support,
                 temporal_matched_kzt=strict, temporal_ratio=strict/inc if inc else 0,
                 temporal_possible_kzt=possible, same_day_extra_kzt=max(0, possible-strict),
                 centrality=central[gid], centrality_percentile=central_rank[gid],
                 predecessor_groups=len({cluster[g] for g in distinct_in}),
                 external_clusters=len({cluster[g] for g in neighbors} - {cluster[gid]}),
                 boundary=node.get('is_boundary', node['depth'] >= OBSERVATION_DEPTH), n_tx=len(transactions[gid]),
                 right_censored=any(d >= last_day-cfg['temporal_days'] for d, _ in in_tx[gid]),
                 out_entropy=entropy, in_percentile=in_rank[gid], out_percentile=out_rank[gid],
                 volume_percentile=volume_rank[gid], in_degree_percentile=degree_rank[gid])
        alternatives = roles(m, node, cfg)
        role, score = alternatives[0]
        parts = [m['seed_norm'], volume_rank[gid], central_rank[gid], score, degree_rank[gid]]
        weights = cfg['priority_weights']
        priority = sum(v*w for v, w in zip(parts, weights))/sum(weights) if m['n_tx'] else 0
        limitations = ['Граф ограничен банком, периодом и порогом; роль является гипотезой.']
        if m['boundary']:
            limitations.append('Исходящие на границе обхода не наблюдаются; терминальность не установлена.')
        if node['is_seed']:
            limitations.append('Входящие seed неполны; отношение выход/вход не используется для transit/terminal.')
        if inc < out:
            limitations.append('Наблюдаемый выход превышает вход; это не баланс и не признак дохода.')
        if m['same_day_extra_kzt']:
            limitations.append('Есть переводы одной даты: их порядок и связь средств не установлены.')
        if m['right_censored']:
            limitations.append('Есть поступления у конца окна; последующие переводы могут быть вне периода.')
        if role == 'coordinator':
            limitations.append('Центральность и межкластерные связи не доказывают управление группой.')
        if m['boundary']:
            request = f'Исходящие переводы gid={gid} за тот же период без ограничения четырьмя коленами.'
        elif node['is_seed'] or inc < out:
            request = f'Полная входящая история gid={gid} за тот же период для проверки неполноты источников.'
        elif m['right_censored']:
            request = f'Переводы gid={gid} после конца окна для проверки гипотезы удержания/транзита.'
        else:
            request = f'Уточнить время и полноту входящих/исходящих gid={gid}; проверить наблюдаемые маршруты.'
        nodes.append(dict(gid=str(gid), depth=node['depth'], is_seed=node['is_seed'], role=role,
                          role_score=score, priority_score=round(clamp(priority), 6), cluster_id=cluster[gid],
                          evidence=explain(role, m, node['is_seed'], cfg['temporal_days']), metrics=m,
                          alternatives=[dict(role=r, score=s) for r, s in alternatives[1:]],
                          priority_components=dict(zip(['seed','volume','centrality','role','predecessors'], parts)),
                          seed_ids=[str(s) for s in sorted(support[gid])],
                          paths=[[str(g) for g in p] for p in paths[gid]],
                          transactions=[dict(t, src=str(t['src']), dst=str(t['dst'])) for t in sorted(transactions[gid], key=lambda t: (t['date'], t['tx_id']))[:100]],
                          limitations=limitations, next_request=request))
    return nodes, cfg


def cluster_rows(data, nodes):
    groups = defaultdict(list)
    mapping = {int(n['gid']): n['cluster_id'] for n in nodes}
    sums = defaultdict(float)
    for n in nodes:
        groups[n['cluster_id']].append(n)
    for e in data.edges:
        if mapping[e['src']] == mapping[e['dst']]:
            sums[mapping[e['src']]] += e['sum_kzt']
    result = []
    for cid, group in sorted(groups.items()):
        ranked = sorted(group, key=lambda n: (-n['priority_score'], int(n['gid'])))
        counts = defaultdict(int)
        for n in group:
            counts[n['role']] += 1
        dominant = max(counts, key=lambda role: (counts[role], role))
        hypothesis = ('Изолированный клиент, финансовая функция не установлена.' if len(group)==1 and not group[0]['metrics']['n_tx']
                      else f"Структурное сообщество; чаще встречается {dominant}. Связь с преступной группой требует проверки.")
        result.append(dict(cluster_id=cid, n_nodes=len(group), n_seed=sum(n['is_seed'] for n in group),
                           sum_kzt_internal=round(sums[cid], 2), top_gids=';'.join(n['gid'] for n in ranked[:5]),
                           hypothesis=hypothesis))
    return result


def next_queries(nodes, limit=20):
    remaining = [n for n in nodes if n['metrics']['n_tx']]
    covered, clusters, result = set(), set(), []
    while remaining and len(result) < limit:
        def utility(n):
            support = set(n['seed_ids'])
            novelty = len(support-covered)/max(1,len(support))
            unknown = n['metrics']['boundary'] or n['is_seed'] or n['metrics']['right_censored']
            return n['priority_score'] * (0.6+0.2*novelty+0.2*(n['cluster_id'] not in clusters)) * (1+.15*unknown)
        chosen = min(remaining, key=lambda n: (-utility(n), int(n['gid'])))
        result.append(dict(gid=chosen['gid'], reason=f"Приоритет {chosen['priority_score']:.3f}; новых seed {len(set(chosen['seed_ids'])-covered)}; кластер {chosen['cluster_id']}", request=chosen['next_request']))
        covered.update(chosen['seed_ids'])
        clusters.add(chosen['cluster_id'])
        remaining.remove(chosen)
    return result


def witness_certificates(data, limit=30, max_depth=4):
    """Construct two topological completions with the same depth-limited extraction."""
    seeds = [gid for gid,n in data.nodes.items() if n['is_seed']]
    base = {(e['src'],e['dst']) for e in data.edges}
    def observe(edges):
        adj = defaultdict(list)
        for a,b in edges:
            adj[a].append(b)
        seen = {g:0 for g in seeds}
        queue = deque(seeds)
        recorded = set()
        while queue:
            g=queue.popleft()
            if seen[g]>=max_depth:
                continue
            for other in adj[g]:
                recorded.add((g,other))
                if other not in seen:
                    seen[other]=seen[g]+1
                    queue.append(other)
        return recorded,seen
    observed, depths=observe(base)
    if observed != base:
        return []
    outgoing={a for a,b in base}
    result=[]
    for gid in sorted(data.nodes):
        if depths.get(gid)==max_depth and gid not in outgoing:
            augmented=base|{(gid,'hypothetical-recipient')}
            same=observe(augmented)[0]==observed
            result.append(dict(gid=str(gid),observed_same=same,property_changes=True,
                               scenario_a='В сценарии A у узла нет исходящих рёбер.',
                               scenario_b=f'В сценарии B добавлен исходящий перевод за границу колена {max_depth}.',
                               scope='Проверка структуры обхода. Добавленное ребро гипотетическое, не факт из датасета.'))
            if len(result)>=limit:
                break
    return result
