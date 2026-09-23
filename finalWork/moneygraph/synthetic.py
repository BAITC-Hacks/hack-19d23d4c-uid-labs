"""Synthetic engineering fixtures; never presented as results of the actual case."""
from collections import deque
from pathlib import Path
import json
import random
from .data import aggregate, write_csv


def generate(directory, size=2248, seed=42):
    if size < 30:
        raise ValueError('Synthetic demo needs at least 30 nodes')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    n_seed = min(81, max(4, size//20))
    isolated = max(1, round(n_seed*19/81))
    weights = [472, 462, 789, 444]
    counts = [max(2, int((size-n_seed)*w/sum(weights))) for w in weights]
    counts[-1] += size-n_seed-sum(counts)
    layers = [list(range(1, n_seed-isolated+1))]
    current = n_seed+1
    for count in counts:
        layers.append(list(range(current,current+count)))
        current += count
    pairs = set()
    for depth in range(1,5):
        for index,gid in enumerate(layers[depth]):
            pairs.add((layers[depth-1][index % len(layers[depth-1])],gid))
    cycle_edges = 2 * len(list(zip(layers[2][:8:2], layers[2][1:8:2])))
    target = max(len(pairs), round(size*3119/2248)-cycle_edges)
    attempts = 0
    while len(pairs)<target and attempts<target*40:
        attempts += 1
        depth = rng.randrange(4)
        pairs.add((rng.choice(layers[depth]),rng.choice(layers[depth+1])))
    # A few structural cycles inside the observed interior.
    for a,b in zip(layers[2][:8:2], layers[2][1:8:2]):
        pairs.add((a,b)); pairs.add((b,a))
    # Keep a stable target count where possible; spanning layer edges stay intact.
    depths={g:d for d,layer in enumerate(layers) for g in layer}
    outgoing={g:[] for g in range(1,size+1)}
    for a,b in pairs:
        outgoing[a].append(b)
    minimum={g:0 for g in range(1,n_seed+1)}
    queue=deque(minimum)
    while queue:
        a=queue.popleft()
        for b in outgoing[a]:
            if b not in minimum:
                minimum[b]=minimum[a]+1
                queue.append(b)
    tx=[]
    target_tx=max(len(pairs),round(size*4840/2248))
    ordered=sorted(pairs)
    multiplicity={pair:1 for pair in ordered}
    for _ in range(target_tx-len(pairs)):
        multiplicity[rng.choice(ordered)]+=1
    for a,b in ordered:
        for j in range(multiplicity[a,b]):
            day=min(29,3+2*minimum[a]+(j%2))
            tx.append(dict(src=a,dst=b,date=f'2026-07-{day:02d}',sum_kzt=5000*rng.randint(1,40)))
    nodes=[dict(gid=g,depth=minimum.get(g,0),is_seed=g<=n_seed) for g in range(1,size+1)]
    edges=aggregate(tx)
    for e in edges:
        e['depth']=minimum[e['dst']]
    write_csv(directory/'nodes.csv',nodes,['gid','depth','is_seed'])
    write_csv(directory/'edges.csv',edges,['src','dst','sum_kzt','n_tx','depth'])
    write_csv(directory/'transactions.csv',tx,['src','dst','date','sum_kzt'])
    metadata=dict(synthetic=True,generator='moneygraph synthetic engineering fixture',random_seed=seed,
                  n_nodes=size,n_seed=n_seed,isolated_seed=isolated,
                  note='Не данные организаторов. Истинные преступные роли отсутствуют. Проверка функциональности и производительности.')
    (directory/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return directory
