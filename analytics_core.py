"""Explainable, offline graph analytics for the Sled investigation navigator.

Roles and scores describe observed transaction patterns, not guilt or calibrated
probabilities. Monetary arithmetic uses integer tiyn; identifiers remain int64.
See README.md for formulas, decision precedence and observation limitations.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_array

LOG = logging.getLogger(__name__)
BOUNDARY_LIMITATION = "Дальнейшие переводы не наблюдаются (граница горизонта 4 колен)"
GENERAL_LIMITATION = (
    "Видна только предоставленная выборка внутрибанковских переводов от 5 000 ₸; "
    "net_flow не является остатком счёта. Роль — гипотеза, не вывод о виновности."
)
ROLE_LABELS = {
    "consolidator": "Агрегатор / Сбор",
    "distributor": "Веерный распределитель",
    "transit": "Транзитный узел",
    "terminal": "Накопитель / Касса",
    "source": "Источник / Дроп",
    "coordinator": "Координирующий узел (гипотеза)",
    "peripheral": "Недостаточно признаков роли",
}
NEXT_STEPS = {
    "consolidator": "Проверить происхождение входящих потоков и последовательность переводов получателям.",
    "distributor": "Проверить назначения веерных переводов и повторяемость набора получателей по дням.",
    "transit": "Запросить точное время операций и проверить связь поступлений с последующими переводами.",
    "terminal": "Запросить остатки, межбанковские операции и переводы за пределами периода наблюдения.",
    "source": "Запросить предшествующие входящие операции; проверить основание включения в исходную выборку.",
    "coordinator": "Проверить пути от исходных участников и связи между сообществами; роль руководителя не установлена.",
    "peripheral": "Запросить расширенную историю операций; текущих признаков недостаточно для специальной роли.",
}
PRIORITY_WEIGHTS = {
    "seed_reach": 0.30, "visible_volume": 0.20, "branching": 0.15,
    "betweenness_centrality": 0.15, "page_rank": 0.10, "authority_score": 0.10,
}


class DataValidationError(ValueError):
    """An input or generated output violates the documented data contract."""


@dataclass(frozen=True)
class Thresholds:
    collector_in_degree: float = 3.0
    collector_max_pass: float = 0.5
    distributor_out_degree: float = 10.0
    distributor_fanout_ratio: float = 2.0
    transit_balance_tolerance: float = 0.2
    transit_time_share: float = 0.5
    terminal_max_pass: float = 0.1
    coordinator_seed_reach: float = 3.0


@dataclass(frozen=True)
class Config:
    horizon: int = 4
    random_seed: int = 42
    betweenness_samples: int = 128
    top_n: int = 20
    thresholds: Thresholds = Thresholds()

    def __post_init__(self) -> None:
        if self.horizon != 4:
            raise ValueError("Для этого кейса горизонт должен быть равен 4.")
        if self.betweenness_samples < 1 or self.top_n < 20:
            raise ValueError("betweenness_samples >= 1; top_n >= 20.")
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self.thresholds).values()):
            raise ValueError("Все пороги должны быть положительными конечными числами.")


@dataclass
class AnalysisResult:
    graph: nx.DiGraph
    transactions: pd.DataFrame
    features: pd.DataFrame
    nodes_roles: pd.DataFrame
    clusters: pd.DataFrame
    top_nodes: pd.DataFrame
    seed_paths: dict[int, dict[int, list[int]]]
    sensitivity: dict[int, dict[str, Any]]
    config: Config
    coordinator_cutoff: float


def _require_columns(frame: pd.DataFrame, names: set[str], label: str) -> None:
    missing = names - set(frame.columns)
    if missing:
        raise DataValidationError(f"{label}: отсутствуют колонки {sorted(missing)}")
    if frame[list(names)].isna().any().any():
        raise DataValidationError(f"{label}: обязательные колонки содержат NaN/NULL.")


def _validate_ids(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    for column in columns:
        values = frame[column]
        if not pd.api.types.is_integer_dtype(values.dtype):
            raise DataValidationError(f"{label}.{column}: нужен целочисленный ID, не float/string.")
        if len(values) and (values.min() < 0 or values.max() > np.iinfo(np.int64).max):
            raise DataValidationError(f"{label}.{column}: ID вне диапазона int64.")
        frame[column] = values.astype("int64")


def _to_cents(values: pd.Series, label: str) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype):
        raise DataValidationError(f"{label}: суммы должны быть числами.")
    amounts = values.to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts <= 0).any():
        raise DataValidationError(f"{label}: суммы должны быть положительными и конечными.")
    scaled = amounts * 100
    if (scaled > 2**53 - 1).any():
        raise DataValidationError(f"{label}: сумма слишком велика для точного преобразования float в тиыны.")
    rounded = np.rint(scaled)
    if not np.allclose(scaled, rounded, rtol=0, atol=1e-4):
        raise DataValidationError(f"{label}: более двух значащих знаков после запятой.")
    cents = rounded.astype(np.int64)
    if sum(map(int, cents)) > np.iinfo(np.int64).max:
        raise DataValidationError(f"{label}: общий объём превышает int64.")
    return pd.Series(cents, index=values.index, dtype="int64")


def load_dataset(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and reconcile all three Parquet tables. Repeated transfers are kept."""
    data_dir = Path(data_dir)
    tables = {}
    for name in ("nodes", "edges", "transactions"):
        path = data_dir / f"{name}.parquet"
        if not path.is_file():
            raise DataValidationError(f"Не найден {path}. Поместите три Parquet-файла в data/.")
        try:
            tables[name] = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            raise DataValidationError(f"Не удалось прочитать {path}: {exc}") from exc
    nodes, edges, tx = (tables[k] for k in ("nodes", "edges", "transactions"))
    _require_columns(nodes, {"gid", "depth", "is_seed"}, "nodes")
    _require_columns(edges, {"src", "dst", "sum_kzt", "n_tx", "depth"}, "edges")
    _require_columns(tx, {"src", "dst", "date", "sum_kzt"}, "transactions")
    if nodes.empty or nodes.gid.duplicated().any():
        raise DataValidationError("nodes: таблица пуста или gid повторяются.")
    for frame, cols, label in ((nodes, ["gid"], "nodes"), (edges, ["src", "dst"], "edges"),
                                (tx, ["src", "dst"], "transactions")):
        _validate_ids(frame, cols, label)
    if not pd.api.types.is_bool_dtype(nodes.is_seed.dtype):
        raise DataValidationError("nodes.is_seed: требуется bool.")
    for frame, label, low in ((nodes, "nodes", 0), (edges, "edges", 1)):
        if not pd.api.types.is_integer_dtype(frame.depth.dtype) or not frame.depth.between(low, 4).all():
            raise DataValidationError(f"{label}.depth: требуется целое число от {low} до 4.")
    if not nodes.is_seed.eq(nodes.depth.eq(0)).all() or not nodes.is_seed.any():
        raise DataValidationError("nodes: depth=0 должен соответствовать is_seed; нужны исходные узлы.")
    ids = set(nodes.gid)
    for frame, label in ((edges, "edges"), (tx, "transactions")):
        if (set(frame.src) | set(frame.dst)) - ids:
            raise DataValidationError(f"{label}: есть ID, отсутствующие в nodes.")
        frame["amount_cents"] = _to_cents(frame.sum_kzt, f"{label}.sum_kzt")
    if edges.duplicated(["src", "dst"]).any():
        raise DataValidationError("edges: пара src/dst должна встречаться один раз.")
    if not pd.api.types.is_integer_dtype(edges.n_tx.dtype) or (edges.n_tx <= 0).any():
        raise DataValidationError("edges.n_tx: требуется положительное целое число.")
    try:
        tx["date"] = pd.to_datetime(tx.date, errors="raise")
        if tx.date.dt.tz is not None or not tx.date.eq(tx.date.dt.normalize()).all():
            raise DataValidationError("transactions.date: ожидается календарная дата без времени/часового пояса.")
    except (ValueError, TypeError, AttributeError) as exc:
        raise DataValidationError(f"Некорректные даты транзакций: {exc}") from exc
    if tx.date.isna().any():
        raise DataValidationError("transactions.date: пустая дата.")
    if not tx.empty and tx.amount_cents.min() < 500_000:
        raise DataValidationError("transactions: операция ниже заявленного порога 5 000 KZT.")
    agg = tx.groupby(["src", "dst"], as_index=False).agg(
        cents=("amount_cents", "sum"), count=("amount_cents", "size"))
    merged = edges.merge(agg, on=["src", "dst"], how="outer", indicator=True)
    if not merged._merge.eq("both").all():
        raise DataValidationError("edges и transactions не совпадают по парам src/dst.")
    if not merged.amount_cents.eq(merged.cents).all() or not merged.n_tx.eq(merged["count"]).all():
        raise DataValidationError("edges и transactions не совпадают по суммам или количеству операций.")
    # These IDs reference original Parquet row positions; they are not bank IDs.
    tx["transaction_ref"] = [f"row:{i + 1}" for i in range(len(tx))]
    return nodes.sort_values("gid").reset_index(drop=True), edges, tx


def build_graph(nodes: pd.DataFrame, transactions: pd.DataFrame) -> nx.DiGraph:
    """Construct a directed graph from transactions, explicitly including isolates."""
    graph = nx.DiGraph()
    for row in nodes.itertuples(index=False):
        graph.add_node(int(row.gid), depth=int(row.depth), is_seed=bool(row.is_seed))
    grouped = transactions.groupby(["src", "dst"], as_index=False).agg(
        amount_cents=("amount_cents", "sum"), n_tx=("amount_cents", "size"))
    for row in grouped.itertuples(index=False):
        graph.add_edge(int(row.src), int(row.dst), amount_cents=int(row.amount_cents),
                       amount_kzt=int(row.amount_cents) / 100, n_tx=int(row.n_tx))
    return graph


def _communities(graph: nx.DiGraph, seed: int) -> dict[int, int]:
    undirected = nx.Graph()
    undirected.add_nodes_from(graph)
    for src, dst, attrs in graph.edges(data=True):
        previous = undirected.get_edge_data(src, dst, {}).get("weight", 0)
        undirected.add_edge(src, dst, weight=previous + attrs["amount_cents"])
    if undirected.number_of_edges():
        groups = nx.community.louvain_communities(undirected, weight="weight", seed=seed)
    else:
        groups = [{gid} for gid in undirected]
    groups = sorted(groups, key=lambda group: (-len(group), min(group)))
    return {gid: index for index, group in enumerate(groups) for gid in group}


def _hits(graph: nx.DiGraph) -> tuple[dict[int, float], dict[int, float]]:
    """Weighted HITS by deterministic sparse power iteration.

    Uniform initialization selects a repeatable solution even when the largest
    singular value is repeated. ARPACK's arbitrary basis in that subspace would
    make downstream rankings unstable on disconnected/equal-weight graphs.
    """
    gids = list(graph)
    zeros = dict.fromkeys(gids, 0.0)
    if not graph.number_of_edges():
        return zeros.copy(), zeros.copy()
    if len(gids) == 1:
        return {gids[0]: 1.0}, {gids[0]: 1.0}
    matrix = csr_array(nx.to_scipy_sparse_array(graph, nodelist=gids, weight="amount_kzt", dtype=float, format="csr"))
    matrix = matrix / matrix.data.max()
    transpose = matrix.T.tocsr()
    authorities = np.full(len(gids), 1 / np.sqrt(len(gids)))
    for _ in range(10000):
        hubs = matrix @ authorities
        hubs /= np.linalg.norm(hubs)
        updated = transpose @ hubs
        updated /= np.linalg.norm(updated)
        difference = np.linalg.norm(updated - authorities, ord=1)
        authorities = updated
        if difference < 1e-10:
            break
    else:
        raise RuntimeError("HITS не сошёлся за 10000 итераций; результаты не экспортированы.")
    hubs = matrix @ authorities
    hubs /= hubs.sum()
    authorities /= authorities.sum()
    return dict(zip(gids, hubs)), dict(zip(gids, authorities))


def _temporal_features(transactions: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Greedy temporal compatibility, not attribution of money provenance.

    Match each observed outgoing tiyn at most once to earlier incoming tiyn
    aged 1-2 calendar days. Same-day operations are not ordered or matched.
    """
    incoming: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    outgoing: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in transactions.itertuples(index=False):
        if row.src == row.dst:  # A self-transfer provides no evidence of pass-through.
            continue
        day = row.date.date().toordinal()
        incoming[int(row.dst)][day] += int(row.amount_cents)
        outgoing[int(row.src)][day] += int(row.amount_cents)
    result = {}
    for gid in set(incoming) | set(outgoing):
        supply: deque[list[int]] = deque()
        matched = 0
        for day in sorted(set(incoming[gid]) | set(outgoing[gid])):
            while supply and day - supply[0][0] > 2:
                supply.popleft()
            demand = outgoing[gid].get(day, 0)
            while demand and supply:
                used = min(demand, supply[0][1])
                demand -= used
                supply[0][1] -= used
                matched += used
                if not supply[0][1]:
                    supply.popleft()
            if incoming[gid].get(day, 0):
                supply.append([day, incoming[gid][day]])
        ins, outs = sum(incoming[gid].values()), sum(outgoing[gid].values())
        result[gid] = {
            "temporal_matched_amount": matched / 100,
            "temporal_in_share": matched / ins if ins else 0.0,
            "temporal_out_share": matched / outs if outs else 0.0,
        }
    return result


def compute_metrics(graph: nx.DiGraph, transactions: pd.DataFrame, config: Config
                    ) -> tuple[pd.DataFrame, dict[int, dict[int, list[int]]]]:
    ids = list(graph)
    communities = _communities(graph, config.random_seed)
    page_rank = nx.pagerank(graph, alpha=0.85, weight="amount_kzt", max_iter=1000, tol=1e-10)
    samples = config.betweenness_samples if len(graph) > config.betweenness_samples else None
    between = nx.betweenness_centrality(graph, k=samples, normalized=True, weight=None, seed=config.random_seed)
    hubs, authorities = _hits(graph)
    temporal = _temporal_features(transactions)
    paths = {gid: dict(nx.single_source_shortest_path(graph, gid, cutoff=config.horizon))
             for gid in ids if graph.nodes[gid]["is_seed"]}
    origins: dict[int, list[int]] = defaultdict(list)
    for origin, destinations in paths.items():
        for destination in destinations:
            if destination != origin:
                origins[destination].append(origin)
    rows = []
    for gid in ids:
        attrs = graph.nodes[gid]
        ins = int(graph.in_degree(gid, weight="amount_cents"))
        outs = int(graph.out_degree(gid, weight="amount_cents"))
        in_degree, out_degree = graph.in_degree(gid), graph.out_degree(gid)
        self_loop = graph.has_edge(gid, gid)
        in_partners = in_degree - int(self_loop)
        out_partners = out_degree - int(self_loop)
        self_amount = graph[gid][gid]["amount_cents"] if self_loop else 0
        partner_in, partner_out = ins - self_amount, outs - self_amount
        temporal_row = temporal.get(gid, {"temporal_matched_amount": 0.0, "temporal_in_share": 0.0, "temporal_out_share": 0.0})
        rows.append({
            "gid": gid, "depth": attrs["depth"], "is_seed": attrs["is_seed"],
            "in_degree": in_degree, "out_degree": out_degree,
            "in_partners": in_partners, "out_partners": out_partners,
            "in_amount": ins / 100, "out_amount": outs / 100,
            "in_volume": ins / 100, "out_volume": outs / 100,
            "net_flow": (ins - outs) / 100,
            "partner_in_amount": partner_in / 100, "partner_out_amount": partner_out / 100,
            "pass_through": partner_out / partner_in if partner_in else 0.0,
            "pass_through_defined": partner_in > 0,
            "fanout_ratio": out_partners / max(in_partners, 1),
            "page_rank": page_rank[gid], "betweenness_centrality": between[gid],
            "hub_score": hubs[gid], "authority_score": authorities[gid],
            "cluster_id": communities[gid], "seed_reach": len(origins[gid]),
            "cross_cluster_count": len({communities[v] for v in set(graph.predecessors(gid)) | set(graph.successors(gid))
                                        if communities[v] != communities[gid]}),
            "boundary": attrs["depth"] == config.horizon,
            "seed_paths": json.dumps([[str(v) for v in paths[s][gid]] for s in sorted(origins[gid])[:3]], ensure_ascii=False),
            **temporal_row,
        })
    features = pd.DataFrame(rows).set_index("gid", drop=False)
    return features, paths


def _role_checks(row: dict[str, Any], thresholds: Thresholds, coordinator_cutoff: float) -> dict[str, bool]:
    t = thresholds
    # Balance-based hypotheses are disabled for seeds and the observation boundary.
    balance_usable = bool(row["pass_through_defined"] and not row["is_seed"] and not row["boundary"])
    return {
        "coordinator": bool(row["seed_reach"] >= t.coordinator_seed_reach and row["cross_cluster_count"] >= 2
                            and row["betweenness_centrality"] > 0 and row["betweenness_centrality"] >= coordinator_cutoff
                            and row["in_partners"] >= 2 and row["out_partners"] >= 2 and not row["boundary"]),
        "distributor": bool(row["out_partners"] >= t.distributor_out_degree and row["fanout_ratio"] >= t.distributor_fanout_ratio),
        "transit": bool(balance_usable and row["out_partners"] > 0
                        and abs(row["pass_through"] - 1) <= t.transit_balance_tolerance
                        and min(row["temporal_in_share"], row["temporal_out_share"]) >= t.transit_time_share),
        "consolidator": bool(row["in_partners"] >= t.collector_in_degree
                             and (not balance_usable or row["pass_through"] <= t.collector_max_pass)),
        "terminal": bool(balance_usable and row["pass_through"] <= t.terminal_max_pass),
        "source": bool(row["is_seed"] and row["out_partners"] > 0),
        "peripheral": True,
    }


def classify_role(row: dict[str, Any], thresholds: Thresholds, coordinator_cutoff: float) -> str:
    """First matching rule wins. Structural role takes precedence over seed status."""
    return next(role for role, matches in _role_checks(row, thresholds, coordinator_cutoff).items() if matches)


def analyze_sensitivity(features: pd.DataFrame, thresholds: Thresholds, coordinator_cutoff: float
                        ) -> dict[int, dict[str, Any]]:
    """Vary every role threshold individually and jointly by +/-10% and +/-20%."""
    scenarios = []
    base = asdict(thresholds)
    for factor in (0.8, 0.9, 1.1, 1.2):
        suffix = f"{round((factor - 1) * 100):+d}%"
        scenarios.append((f"all:{suffix}", Thresholds(**{key: value * factor for key, value in base.items()})))
        for key, value in base.items():
            scenarios.append((f"{key}:{suffix}", replace(thresholds, **{key: value * factor})))
    result = {}
    for row in features.to_dict("records"):
        baseline = classify_role(row, thresholds, coordinator_cutoff)
        changes = [{"scenario": label, "role": role} for label, scenario in scenarios
                   if (role := classify_role(row, scenario, coordinator_cutoff)) != baseline]
        result[int(row["gid"])] = {
            "baseline_role": baseline, "scenario_count": len(scenarios),
            "stability_score": 1 - len(changes) / len(scenarios),
            "changed_scenarios": changes,
            "alternative_roles": sorted({change["role"] for change in changes}),
        }
    return result


def _role_score(row: dict[str, Any], role: str) -> float:
    # This is an explicit heuristic strength, not a learned probability.
    strength = {
        "coordinator": 0.50 + 0.35 * min(row["seed_reach"] / 10, 1),
        "distributor": 0.50 + 0.35 * min(row["out_partners"] / 50, 1),
        "consolidator": 0.45 + 0.35 * min(row["in_partners"] / 10, 1),
        "transit": 0.50 + 0.35 * min(row["temporal_in_share"], row["temporal_out_share"]),
        "terminal": 0.55 + 0.20 * (1 - min(row["pass_through"], 1)),
        "source": 0.50,
        "peripheral": 0.15,
    }[role]
    if row["boundary"]:
        strength = min(strength, 0.40)
    if row["is_seed"] and role == "consolidator":
        strength = min(strength, 0.55)
    return round(strength, 6)


def _format_money(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ")


def _explanation(row: dict[str, Any], role: str, thresholds: Thresholds) -> str:
    facts = (f"{row['in_partners']} отправителей; вход {_format_money(row['in_amount'])} ₸; "
             f"{row['out_partners']} получателей; выход {_format_money(row['out_amount'])} ₸.")
    reason = {
        "coordinator": f"Пути от {row['seed_reach']} исходных узлов; связи с {row['cross_cluster_count']} другими кластерами; высокий betweenness.",
        "distributor": f"Получателей ≥ {thresholds.distributor_out_degree:g}; отношение числа получателей к max(отправителей,1)={row['fanout_ratio']:.2f}.",
        "consolidator": f"Отправителей ≥ {thresholds.collector_in_degree:g}; наблюдается схождение входящих связей.",
        "transit": f"Выход/вход={row['pass_through']:.3f}; совместимы по времени 1–2 дня {row['temporal_in_share']:.1%} входа и {row['temporal_out_share']:.1%} выхода.",
        "terminal": f"Выход/вход={row['pass_through']:.3f} ≤ {thresholds.terminal_max_pass:g}; узел находится внутри горизонта.",
        "source": "Узел отмечен is_seed; видны исходящие переводы. Название «Дроп» не устанавливает фактическую роль человека.",
        "peripheral": "Пороги специальных ролей не выполнены; это не свидетельство отсутствия риска.",
    }[role]
    return f"{facts} {reason}"


def _priority_components(features: pd.DataFrame) -> pd.DataFrame:
    signals = features[["seed_reach", "betweenness_centrality", "page_rank", "authority_score"]].copy()
    signals["visible_volume"] = features.in_amount + features.out_amount
    signals["branching"] = features[["in_partners", "out_partners"]].max(axis=1)
    for name in PRIORITY_WEIGHTS:
        values = np.log1p(signals[name]) if name in {"seed_reach", "visible_volume", "branching"} else signals[name]
        maximum = float(values.max())
        signals[name] = values / maximum if maximum > 0 else 0.0
    return signals


def _create_outputs(graph: nx.DiGraph, features: pd.DataFrame, sensitivity: dict[int, dict[str, Any]],
                    config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    components = _priority_components(features)
    rows = []
    for feature in features.to_dict("records"):
        gid = int(feature["gid"])
        sens = sensitivity[gid]
        role = sens["baseline_role"]
        role_score = _role_score(feature, role)
        limitation = GENERAL_LIMITATION
        if feature["boundary"]:
            limitation = BOUNDARY_LIMITATION + ". " + limitation
        if feature["is_seed"]:
            limitation += " Входящие на исходный узел неполны; коэффициент выход/вход не определяет роль."
        if feature["in_degree"] + feature["out_degree"] == 0:
            limitation += " Узел отсутствует в таблице переводов; наблюдаемая активность неизвестна."
        if role == "transit":
            limitation += " Сопоставление по дням не доказывает происхождение конкретных денег."
        if sens["alternative_roles"]:
            limitation += " Роль меняется при изменении порогов; требуется дополнительная проверка."
        next_step = ("Запросить исходящие переводы за пределами четвёртого колена. "
                     if feature["boundary"] else "") + NEXT_STEPS[role]
        explanation = _explanation(feature, role, config.thresholds)
        active = feature["in_partners"] + feature["out_partners"] > 0
        penalty = (0.7 if feature["boundary"] else 1.0) * (0.75 + 0.25 * sens["stability_score"])
        raw = sum(float(components.loc[gid, name]) * weight for name, weight in PRIORITY_WEIGHTS.items())
        priority = round(raw * penalty if active else 0.0, 8)
        evidence = (f"Вход: {feature['in_partners']} контраг., {feature['in_amount']:.2f} ₸; "
                    f"выход: {feature['out_partners']}, {feature['out_amount']:.2f} ₸; "
                    f"исходных узлов по путям ≤4: {feature['seed_reach']}.")[:200]
        rows.append({
            "gid": gid, "role": role, "role_label": ROLE_LABELS[role],
            "score": role_score, "role_score": role_score,
            "explanation": explanation, "limitation": limitation, "next_step": next_step,
            "hypothesis": "Гипотеза: " + ROLE_LABELS[role],
            "evidence": evidence, "priority_score": priority,
            "stability_score": round(sens["stability_score"], 8),
            "alternative_roles": ", ".join(sens["alternative_roles"]) or "Не выявлены в проверенных сценариях",
            "sensitivity_scenarios": sens["scenario_count"],
            **{key: value for key, value in feature.items() if key != "gid"},
        })
    nodes = pd.DataFrame(rows).sort_values("gid").reset_index(drop=True)
    mapping = dict(zip(nodes.gid, nodes.cluster_id))
    internal_cents: dict[int, int] = defaultdict(int)
    for src, dst, attributes in graph.edges(data=True):
        if mapping[src] == mapping[dst]:
            internal_cents[mapping[src]] += attributes["amount_cents"]
    cluster_rows = []
    for cluster_id, group in nodes.groupby("cluster_id", sort=True):
        dominant = group.role.value_counts().rename_axis("role").reset_index(name="count").sort_values(
            ["count", "role"], ascending=[False, True]).iloc[0]["role"]
        n_seed = int(group.is_seed.sum())
        volume = internal_cents[int(cluster_id)] / 100
        description = (f"Сообщество из {len(group)} узлов, исходных: {n_seed}. "
                       f"Преобладает: {ROLE_LABELS[dominant]}. Внутренние переводы: {_format_money(volume)} ₸. "
                       "Гипотеза о группе связанных потоков; общее управление не доказано.")
        leaders = group.sort_values(["priority_score", "gid"], ascending=[False, True]).head(5)
        cluster_rows.append({
            "cluster_id": int(cluster_id), "node_count": len(group), "total_volume": volume,
            "dominant_role": dominant, "description": description,
            "n_nodes": len(group), "n_seed": n_seed, "sum_kzt_internal": volume,
            "top_gids": ";".join(str(int(gid)) for gid in leaders.gid), "hypothesis": description,
        })
    clusters = pd.DataFrame(cluster_rows)
    top = nodes.sort_values(["priority_score", "gid"], ascending=[False, True]).head(config.top_n).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top["why"] = top.apply(lambda row: (
        f"Приоритет {row.priority_score:.4f}; исходных узлов с путями: {row.seed_reach}; "
        f"betweenness={row.betweenness_centrality:.6f}; устойчивость={row.stability_score:.1%}. "
        + row.explanation + " Ограничения: " + row.limitation), axis=1)
    return nodes, clusters, top.reset_index(drop=True)


def analyze(data_dir: str | Path = "data", config: Config | None = None) -> AnalysisResult:
    config = config or Config()
    nodes, _, transactions = load_dataset(data_dir)
    LOG.info("Загружено: %d узлов, %d транзакций.", len(nodes), len(transactions))
    graph = build_graph(nodes, transactions)
    features, seed_paths = compute_metrics(graph, transactions, config)
    positive_between = features.loc[features.betweenness_centrality > 0, "betweenness_centrality"]
    cutoff = float(positive_between.quantile(0.9)) if not positive_between.empty else 1.0
    sensitivity = analyze_sensitivity(features, config.thresholds, cutoff)
    roles, clusters, top = _create_outputs(graph, features, sensitivity, config)
    result = AnalysisResult(graph, transactions, features, roles, clusters, top, seed_paths, sensitivity, config, cutoff)
    validate_outputs(roles, clusters, top, set(map(int, nodes.gid)), config.top_n)
    return result


def find_common_recipients(result: AnalysisResult, sources: list[int | str], min_sources: int = 2
                           ) -> list[dict[str, Any]]:
    """Find directional convergence within four hops, with one shortest path/source.

    These paths are structural: chronological feasibility is not asserted.
    IDs in this JSON-ready response are strings to preserve browser precision.
    """
    sources = sorted({int(source) for source in sources})
    if len(sources) < 2 or not 2 <= min_sources <= len(sources):
        raise ValueError("Выберите минимум два разных источника; 2 <= min_sources <= число источников.")
    if any(source not in result.graph for source in sources):
        raise ValueError("Один из выбранных источников отсутствует в графе.")
    hits: dict[int, list[list[int]]] = defaultdict(list)
    for source in sources:
        paths = result.seed_paths.get(source)
        if paths is None:
            paths = nx.single_source_shortest_path(result.graph, source, cutoff=result.config.horizon)
        for destination, path in paths.items():
            if destination not in sources:
                hits[destination].append(path)
    priority = dict(zip(result.nodes_roles.gid, result.nodes_roles.priority_score))
    recipients = [{"gid": str(gid), "source_count": len(paths),
                   "paths": [[str(node) for node in path] for path in paths],
                   "priority_score": priority[gid],
                   "limitation": "Пути по агрегированному графу не доказывают хронологию и происхождение денег."}
                  for gid, paths in hits.items() if len(paths) >= min_sources]
    return sorted(recipients, key=lambda row: (-row["source_count"], -row["priority_score"], row["gid"]))


def explain_node(result: AnalysisResult, gid: int | str) -> dict[str, Any]:
    """Return formulas, rule checks, sensitivity and all incident transactions."""
    gid = int(gid)
    if gid not in result.graph:
        raise ValueError(f"Узел {gid} не найден.")
    row = result.features.loc[gid].to_dict()
    exported = result.nodes_roles.loc[result.nodes_roles.gid.eq(gid)].iloc[0]
    tx = result.transactions.loc[result.transactions.src.eq(gid) | result.transactions.dst.eq(gid)]
    components = _priority_components(result.features).loc[gid]
    return {
        "gid": str(gid),
        "blocks": {"Гипотеза": exported.hypothesis, "Основания": exported.explanation,
                   "Ограничения": exported.limitation, "Следующий шаг аналитика": exported.next_step},
        "thresholds": asdict(result.config.thresholds),
        "coordinator_betweenness_cutoff": result.coordinator_cutoff,
        "rule_checks_in_precedence_order": _role_checks(row, result.config.thresholds, result.coordinator_cutoff),
        "formulas": {
            "net_flow": "sum(incoming tiyn) / 100 - sum(outgoing tiyn) / 100; не остаток счёта",
            "pass_through": "partner_out_amount / partner_in_amount; при нулевом входе defined=false, значение-заполнитель 0",
            "page_rank": "Взвешенный PageRank: alpha=0.85, вес=amount_kzt, равномерная телепортация",
            "betweenness_centrality": f"Доля направленных кратчайших путей через узел; расстояние=число переходов; k=min({result.config.betweenness_samples},N), seed={result.config.random_seed}",
            "HITS": "hub = A @ authority; authority = A.T @ hub; A — взвешенная матрица; каждый вектор нормирован к сумме 1",
            "temporal_share": "FIFO-сопоставление объёмов по датам с лагом 1–2 дня без повторного использования; один день не упорядочивается",
            "priority": "Сумма вес*нормированный признак; затем ×0.7 для границы и ×(0.75+0.25*stability). Для отсутствия внешних связей: 0.",
            "role_score": {
                "coordinator": "0.50+0.35*min(seed_reach/10,1)",
                "distributor": "0.50+0.35*min(out_partners/50,1)",
                "consolidator": "0.45+0.35*min(in_partners/10,1)",
                "transit": "0.50+0.35*min(temporal_in_share,temporal_out_share)",
                "terminal": "0.55+0.20*(1-min(pass_through,1))",
                "source": "0.50", "peripheral": "0.15",
            }[exported.role] + "; максимум 0.40 на границе; максимум 0.55 для seed-сборщика. Эвристика, не вероятность.",
            "stability": "1 - число сценариев со сменой роли / 36; варьируются пороги правил, а не сами данные",
        },
        "metrics": {key: (value.item() if isinstance(value, np.generic) else value)
                    for key, value in row.items() if key not in {"gid", "seed_paths"}},
        "priority_terms": {name: {"weight": weight, "normalized_value": float(components[name]),
                                  "weighted_value": float(components[name]) * weight}
                           for name, weight in PRIORITY_WEIGHTS.items()},
        "sensitivity": result.sensitivity[gid],
        "paths_from_seeds": {str(source): [str(v) for v in paths[gid]] for source, paths in result.seed_paths.items()
                             if gid in paths and source != gid},
        "transactions": [{"ref": r.transaction_ref, "src": str(r.src), "dst": str(r.dst),
                          "date": r.date.strftime("%Y-%m-%d"), "sum_kzt": int(r.amount_cents) / 100}
                         for r in tx.sort_values(["date", "src", "dst", "transaction_ref"]).itertuples(index=False)],
    }


def validate_outputs(nodes: pd.DataFrame, clusters: pd.DataFrame, top: pd.DataFrame,
                     expected_ids: set[int], top_n: int = 20) -> None:
    required = {
        "nodes_roles": {"gid", "role", "score", "explanation", "limitation", "next_step", "role_score", "cluster_id", "priority_score", "evidence"},
        "clusters": {"cluster_id", "node_count", "total_volume", "dominant_role", "description", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"},
        "top_nodes": {"rank", "gid", "role", "priority_score", "why"},
    }
    for name, frame in (("nodes_roles", nodes), ("clusters", clusters), ("top_nodes", top)):
        _require_columns(frame, required[name], name)
        if frame.empty or frame.isna().any().any():
            raise DataValidationError(f"{name}: пустая таблица или есть NaN.")
        for column in frame.select_dtypes(include="number"):
            if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
                raise DataValidationError(f"{name}.{column}: нечисловое/бесконечное значение.")
        for column in frame.select_dtypes(include=["object", "string"]):
            if frame[column].astype(str).str.strip().eq("").any():
                raise DataValidationError(f"{name}.{column}: пустая строка.")
    if nodes.gid.duplicated().any() or set(map(int, nodes.gid)) != expected_ids:
        raise DataValidationError("nodes_roles: потерян или дублирован узел.")
    if not nodes.role.isin(ROLE_LABELS).all():
        raise DataValidationError("nodes_roles: неизвестная роль.")
    for column in ("score", "role_score", "priority_score", "stability_score"):
        if not nodes[column].between(0, 1).all():
            raise DataValidationError(f"nodes_roles.{column}: значение вне 0..1.")
    if not nodes.evidence.str.len().le(200).all():
        raise DataValidationError("evidence превышает 200 символов.")
    boundary = nodes.depth.eq(4)
    if not nodes.loc[boundary, "limitation"].str.contains(BOUNDARY_LIMITATION, regex=False).all():
        raise DataValidationError("Не все граничные узлы имеют обязательное ограничение.")
    if nodes.loc[boundary, "role"].isin(["terminal", "transit"]).any():
        raise DataValidationError("Граничный узел ошибочно получил балансовую роль.")
    if clusters.cluster_id.duplicated().any() or set(nodes.cluster_id) != set(clusters.cluster_id):
        raise DataValidationError("Кластеры и узлы не согласованы.")
    actual_counts = nodes.groupby("cluster_id").size().sort_index()
    expected_counts = clusters.set_index("cluster_id").node_count.sort_index()
    if not actual_counts.equals(expected_counts):
        raise DataValidationError("Размеры кластеров не согласованы.")
    expected_top = nodes.sort_values(["priority_score", "gid"], ascending=[False, True]).head(top_n)
    if list(top.gid) != list(expected_top.gid) or list(top["rank"]) != list(range(1, len(top) + 1)):
        raise DataValidationError("top_nodes: неверный состав/порядок топа.")


def write_outputs(result: AnalysisResult, out_dir: str | Path = ".") -> list[Path]:
    """Stage, read back and validate CSVs before replacing any destination file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {"nodes_roles.csv": result.nodes_roles, "clusters.csv": result.clusters, "top_nodes.csv": result.top_nodes}
    temporary: dict[str, Path] = {}
    try:
        for name, frame in frames.items():
            descriptor, temp_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=out_dir)
            os.close(descriptor)
            path = Path(temp_name)
            temporary[name] = path
            frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")
        reloaded = {name: pd.read_csv(path, dtype={"gid": "int64"} if name != "clusters.csv" else None)
                    for name, path in temporary.items()}
        validate_outputs(reloaded["nodes_roles.csv"], reloaded["clusters.csv"], reloaded["top_nodes.csv"],
                         set(result.graph), result.config.top_n)
        for name, path in temporary.items():
            os.replace(path, out_dir / name)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    return [out_dir / name for name in frames]
