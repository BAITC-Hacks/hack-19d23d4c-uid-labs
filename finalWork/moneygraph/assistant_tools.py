"""Bounded, read-only evidence tools for the language-model assistant.

Amounts are calculated here, never delegated to a language model. ``tx:N`` is
the one-based row in the validated input, not a bank transaction identifier.
"""
from collections import Counter, defaultdict, deque
from copy import deepcopy
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import math
import re

from .data import Dataset
from .engine import ROLE_LABELS


MAX_ROWS = 50
PRIORITY_THRESHOLDS = {"high": 0.7, "medium": 0.4}
LIMITATIONS = [
    "Роли — гипотезы, оценки — сила правил и приоритет проверки, не вероятность виновности.",
    "Граф ограничен банком, периодом, порогом 5 000 KZT и четырьмя коленами обхода.",
    "Отсутствие исходящих на глубине 4 не доказывает конечное получение денег.",
    "Входящие seed неполны; отношение выход/вход не доказывает их транзитную или конечную роль.",
    "Даты не содержат время суток. Структурный путь не доказывает перемещение одних и тех же денег.",
    "Оборот суммирует наблюдаемые переводы, включая звенья цепочек; это не размер преступного дохода.",
    "tx:N — номер строки входной таблицы, не банковский идентификатор. Одинаковые строки сохранены.",
]


class ToolInputError(ValueError):
    pass


def priority_band(score):
    return "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"


def _sum(rows):
    value = sum((Decimal(str(row["sum_kzt"])) for row in rows), Decimal(0))
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _nullable(schema):
    result = deepcopy(schema)
    result["type"] = [result["type"], "null"]
    if "enum" in result:
        result["enum"].append(None)
    return result


GID = {"type": "string", "pattern": r"^-?(0|[1-9][0-9]*)$", "maxLength": 20,
       "description": "Точный gid из данных строкой: не преобразовывать через float или Number."}
LIMIT = {"type": "integer", "minimum": 1, "maximum": MAX_ROWS}
DAY = {"type": "string", "pattern": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"}


def _tool(name, description, properties):
    return {"type": "function", "name": name, "description": description, "strict": True,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False}}


DEFINITIONS = [
    _tool("overview", "Размер и период всей выгрузки, оборот и ограничения наблюдения.", {}),
    _tool("rank_nodes", "Кого проверить первым: ранжировать все узлы по рассчитанному приоритету. Роли — гипотезы.", {
        "role": _nullable({"type": "string", "enum": list(ROLE_LABELS)}),
        "priority": _nullable({"type": "string", "enum": ["high", "medium", "low"]}),
        "limit": LIMIT}),
    _tool("node_details", "Наблюдаемые признаки, предполагаемая роль, ограничения и следующий запрос по точному gid.", {"gid": GID}),
    _tool("transactions", "Найти реальные строки всей transactions, включая строки за пределами предпросмотра. Даты включительно, offset для следующих страниц.", {
        "gid": _nullable(GID), "direction": {"type": "string", "enum": ["in", "out", "all"]},
        "date_from": _nullable(DAY), "date_to": _nullable(DAY),
        "min_amount": _nullable({"type": "number", "minimum": 0}),
        "limit": LIMIT, "offset": {"type": "integer", "minimum": 0, "maximum": 10000000}}),
    _tool("connections", "Наблюдаемые связи клиента; с other_gid — кратчайший направленный путь и общие контрагенты. Путь не доказывает движение тех же денег.", {
        "gid": GID, "other_gid": _nullable(GID)}),
    _tool("cluster_details", "Структурная группа: участники, внутренний и внешний оборот, приоритетные узлы. Не подтверждает преступную группу.", {
        "cluster_id": {"type": "integer", "minimum": 0}}),
    _tool("recurring_patterns", "Частые направленные пары с не менее min_transactions переводами и редкие пары с одним переводом. Частота сама по себе не признак преступления.", {
        "gid": _nullable(GID), "min_transactions": {"type": "integer", "minimum": 2, "maximum": 10000000},
        "limit": LIMIT}),
]


class GraphTools:
    def __init__(self, data: Dataset, report: dict):
        self.data = data
        self.report = report
        self.nodes = {str(row["gid"]): row for row in report.get("nodes", [])}
        self.clusters = {row["cluster_id"]: row for row in report.get("clusters", [])}
        self.by_node = defaultdict(list)
        self.by_pair = defaultdict(list)
        self.incoming = defaultdict(set)
        self.outgoing = defaultdict(set)
        self.rows = []
        for index, row in enumerate(data.transactions, 1):
            tx = {"source_id": f"tx:{index}", "input_row": index,
                  "src": str(row["src"]), "dst": str(row["dst"]),
                  "date": row["date"], "sum_kzt": float(row["sum_kzt"])}
            self.rows.append(tx)
            self.by_node[tx["src"]].append(tx)
            if tx["dst"] != tx["src"]:
                self.by_node[tx["dst"]].append(tx)
            self.by_pair[tx["src"], tx["dst"]].append(tx)
            self.outgoing[tx["src"]].add(tx["dst"])
            self.incoming[tx["dst"]].add(tx["src"])

    @property
    def definitions(self):
        return deepcopy(DEFINITIONS)

    def _gid(self, value):
        if not isinstance(value, str) or not re.fullmatch(GID["pattern"], value):
            raise ToolInputError("gid должен быть точной десятичной строкой без округления.")
        if not -(2**63) <= int(value) < 2**63:
            raise ToolInputError("gid находится вне диапазона int64.")
        if int(value) not in self.data.nodes:
            raise LookupError(f"gid={value} отсутствует в выгрузке.")
        return value

    def _validate(self, name, arguments):
        if not isinstance(arguments, dict):
            raise ToolInputError("Аргументы должны быть JSON-объектом.")
        spec = next(item for item in DEFINITIONS if item["name"] == name)["parameters"]["properties"]
        unknown = set(arguments) - set(spec)
        if unknown:
            raise ToolInputError("Неизвестные аргументы: " + ", ".join(sorted(unknown)))
        for key, value in arguments.items():
            rule = spec[key]
            types = rule["type"] if isinstance(rule["type"], list) else [rule["type"]]
            if value is None and "null" in types:
                continue
            valid = (("string" in types and isinstance(value, str)) or
                     ("integer" in types and isinstance(value, int) and not isinstance(value, bool)) or
                     ("number" in types and isinstance(value, (int, float)) and not isinstance(value, bool)))
            if not valid:
                raise ToolInputError(f"Неверный тип аргумента {key}.")
            if "enum" in rule and value not in rule["enum"]:
                raise ToolInputError(f"Недопустимое значение {key}.")
            if isinstance(value, (int, float)):
                if (isinstance(value, float) and not math.isfinite(value)) or value < rule.get("minimum", -math.inf) or value > rule.get("maximum", math.inf):
                    raise ToolInputError(f"Аргумент {key} вне допустимого диапазона.")
            if isinstance(value, str):
                if len(value) > rule.get("maxLength", 100) or ("pattern" in rule and not re.fullmatch(rule["pattern"], value)):
                    raise ToolInputError(f"Неверный формат аргумента {key}.")
        for key, rule in spec.items():
            if key not in arguments and key in ("gid", "cluster_id") and rule["type"] != ["string", "null"]:
                raise ToolInputError(f"Не указан {key}.")

    def call(self, name: str, arguments: dict) -> dict:
        if not isinstance(name, str) or name not in {item["name"] for item in DEFINITIONS}:
            return {"ok": False, "error": {"code": "unknown_tool", "message": "Неизвестный инструмент."}}
        try:
            self._validate(name, arguments)
            result = getattr(self, "_" + name)(**arguments)
            return {"ok": True, "tool": name, **result}
        except ToolInputError as error:
            return {"ok": False, "error": {"code": "invalid_arguments", "message": str(error)}}
        except LookupError as error:
            return {"ok": False, "error": {"code": "not_found", "message": str(error)}}

    def _summary(self, row):
        gid = str(row["gid"])
        return {"source_id": "node:" + gid, "gid": gid, "depth": row.get("depth"),
                "is_seed": row.get("is_seed"), "role_hypothesis": row.get("role"),
                "role_label": ROLE_LABELS.get(row.get("role"), "Роль не определена"),
                "role_score": row.get("role_score"), "priority_score": row.get("priority_score", 0),
                "priority": priority_band(row.get("priority_score", 0)), "cluster_id": row.get("cluster_id"),
                "evidence": row.get("evidence", "")[:400]}

    def _overview(self):
        dates = [row["date"] for row in self.rows]
        return {"source_id": "dataset:overview", "n_nodes": len(self.data.nodes),
                "n_edges": len(self.data.edges), "n_transactions": len(self.rows),
                "n_seed": sum(bool(node["is_seed"]) for node in self.data.nodes.values()),
                "n_clusters": len(self.clusters), "total_kzt": _sum(self.rows),
                "currency": "KZT", "date_from": min(dates) if dates else None,
                "date_to": max(dates) if dates else None,
                "synthetic": bool(self.data.meta.get("synthetic", False)),
                "priority_counts": dict(Counter(priority_band(n.get("priority_score", 0)) for n in self.nodes.values())),
                "role_counts": dict(Counter(n.get("role") for n in self.nodes.values())),
                "priority_thresholds": dict(PRIORITY_THRESHOLDS),
                "input_sha256": dict(self.data.meta.get("input_sha256", {})),
                "limitations": LIMITATIONS + self.data.warnings[:10]}

    def _rank_nodes(self, role=None, priority=None, limit=10):
        rows = [node for node in self.nodes.values()
                if (role is None or node.get("role") == role)
                and (priority is None or priority_band(node.get("priority_score", 0)) == priority)]
        rows.sort(key=lambda n: (-n.get("priority_score", 0), int(n["gid"])))
        return {"source_id": "analysis:priority", "matched_count": len(rows),
                "nodes": [dict(rank=index, **self._summary(row)) for index, row in enumerate(rows[:limit], 1)],
                "truncated": len(rows) > limit, "priority_thresholds": dict(PRIORITY_THRESHOLDS),
                "interpretation": "Очередь проверки по эвристикам, не список виновных. Пороговые диапазоны служат для сортировки."}

    def _node_details(self, gid):
        self._gid(gid)
        node = self.nodes.get(gid)
        if node is None:
            raise LookupError("Расчёт роли для узла отсутствует; пересчитайте отчёт.")
        rows = self.by_node[gid]
        incoming = [row for row in rows if row["dst"] == gid]
        outgoing = [row for row in rows if row["src"] == gid]
        metrics = {key: value for key, value in node.get("metrics", {}).items()
                   if value is None or isinstance(value, (bool, int, float))}
        return {**self._summary(node), "metrics": metrics,
                "observed": {"n_transactions": len(rows), "incoming_count": len(incoming),
                             "outgoing_count": len(outgoing), "incoming_kzt": _sum(incoming),
                             "outgoing_kzt": _sum(outgoing)},
                "priority_components": dict(node.get("priority_components", {})),
                "seed_ids": node.get("seed_ids", [])[:20], "seed_count": len(node.get("seed_ids", [])),
                "paths": [path[:5] for path in node.get("paths", [])[:3]],
                "path_note": "Наблюдаемые структурные пути от seed; связь одних и тех же денег не установлена.",
                "next_request": node.get("next_request", "")[:500],
                "limitations": node.get("limitations", [])[:10] + LIMITATIONS[:1],
                "transactions": sorted(rows, key=lambda r: (r["date"], r["input_row"]))[:10],
                "transactions_truncated": len(rows) > 10,
                "more_transactions": "Используйте transactions с gid и offset для всех строк."}

    def _transactions(self, gid=None, direction="all", date_from=None, date_to=None,
                      min_amount=None, limit=20, offset=0):
        if gid is not None:
            self._gid(gid)
        elif direction != "all":
            raise ToolInputError("Для direction=in/out укажите gid.")
        for value in (date_from, date_to):
            if value is not None:
                try:
                    date.fromisoformat(value)
                except ValueError as error:
                    raise ToolInputError("Не существует указанной календарной даты.") from error
        if date_from is not None and date_to is not None and date_from > date_to:
            raise ToolInputError("date_from не может быть позже date_to.")
        rows = [row for row in (self.by_node[gid] if gid is not None else self.rows)
                if (direction != "in" or row["dst"] == gid)
                and (direction != "out" or row["src"] == gid)
                and (date_from is None or row["date"] >= date_from)
                and (date_to is None or row["date"] <= date_to)
                and (min_amount is None or row["sum_kzt"] >= min_amount)]
        rows.sort(key=lambda r: (r["date"], r["input_row"]))
        end = offset + limit
        return {"source_id": "dataset:transactions", "matched_count": len(rows),
                "matched_total_kzt": _sum(rows), "offset": offset, "rows": rows[offset:end],
                "next_offset": end if end < len(rows) else None,
                "source_id_definition": "tx:N — неизменный номер строки transactions (с 1), не банковский ID.",
                "note": "Сумма относится ко всем совпавшим строкам, а не только к странице. Повторяющиеся строки сохранены."}

    def _pair_fact(self, source, target):
        rows = self.by_pair[source, target]
        return {"source_id": f"edge:{source}->{target}", "src": source, "dst": target,
                "n_transactions": len(rows), "sum_kzt": _sum(rows),
                "date_from": min(row["date"] for row in rows), "date_to": max(row["date"] for row in rows),
                "transaction_source_ids": [r["source_id"] for r in rows[:5]],
                "transaction_sources_truncated": len(rows) > 5}

    def _path(self, source, target):
        queue, previous = deque([source]), {source: None}
        while queue:
            current = queue.popleft()
            if current == target:
                path = []
                while current is not None:
                    path.append(current)
                    current = previous[current]
                return list(reversed(path))
            for neighbor in sorted(self.outgoing[current], key=int):
                if neighbor not in previous:
                    previous[neighbor] = current
                    queue.append(neighbor)
        return None

    def _connections(self, gid, other_gid=None):
        self._gid(gid)
        result = {"gid": gid, "source_id": "node:" + gid,
                  "incoming_neighbors_count": len(self.incoming[gid]),
                  "outgoing_neighbors_count": len(self.outgoing[gid]),
                  "incoming": [self._pair_fact(s, gid) for s in sorted(self.incoming[gid], key=int)[:12]],
                  "outgoing": [self._pair_fact(gid, t) for t in sorted(self.outgoing[gid], key=int)[:12]],
                  "neighbors_truncated": len(self.incoming[gid]) > 12 or len(self.outgoing[gid]) > 12,
                  "interpretation": "Только наблюдаемые связи. Совпадение контрагента и структурный путь не доказывают преступную связь или движение тех же денег."}
        if other_gid is not None:
            self._gid(other_gid)
            path = self._path(gid, other_gid)
            result.update(other_gid=other_gid, path_found=path is not None,
                          shortest_hops=len(path)-1 if path is not None else None,
                          path=path[:33] if path is not None else None,
                          path_truncated=bool(path and len(path) > 33),
                          path_edges=[self._pair_fact(s, t) for s, t in zip(path[:32], path[1:33])] if path else [])
            for name, values in [("common_senders", self.incoming[gid] & self.incoming[other_gid]),
                                 ("common_recipients", self.outgoing[gid] & self.outgoing[other_gid])]:
                result[name] = sorted(values, key=int)[:20]
                result[name + "_count"] = len(values)
            result["path_note"] = "Если пути нет, это означает отсутствие направленного пути в данной неполной выгрузке."
        return result

    def _cluster_details(self, cluster_id):
        if cluster_id not in self.clusters:
            raise LookupError(f"Кластер {cluster_id} отсутствует в отчёте.")
        cluster = self.clusters[cluster_id]
        nodes = [node for node in self.nodes.values() if node.get("cluster_id") == cluster_id]
        members = {str(node["gid"]) for node in nodes}
        internal, incoming, outgoing = [], [], []
        for tx in self.rows:
            a, b = tx["src"] in members, tx["dst"] in members
            if a and b:
                internal.append(tx)
            elif b:
                incoming.append(tx)
            elif a:
                outgoing.append(tx)
        ranked = sorted(nodes, key=lambda n: (-n.get("priority_score", 0), int(n["gid"])))
        return {"source_id": f"cluster:{cluster_id}", "cluster_id": cluster_id,
                "n_nodes": len(nodes), "n_seed": sum(bool(n.get("is_seed")) for n in nodes),
                "hypothesis": cluster.get("hypothesis", "")[:500],
                "internal_n_transactions": len(internal), "internal_kzt": _sum(internal),
                "external_incoming_kzt": _sum(incoming), "external_outgoing_kzt": _sum(outgoing),
                "top_nodes": [self._summary(n) for n in ranked[:20]],
                "nodes_truncated": len(nodes) > 20,
                "internal_transaction_sources": [r["source_id"] for r in internal[:10]],
                "interpretation": "Структурный кластер — гипотеза объединения; принадлежность к преступной группе не установлена."}

    def _recurring_patterns(self, gid=None, min_transactions=3, limit=10):
        if gid is not None:
            self._gid(gid)
        pairs = [(pair, rows) for pair, rows in self.by_pair.items() if gid is None or gid in pair]
        frequent = [pair for pair, rows in pairs if len(rows) >= min_transactions]
        rare = [pair for pair, rows in pairs if len(rows) == 1]
        frequent.sort(key=lambda pair: (-len(self.by_pair[pair]), -_sum(self.by_pair[pair]), int(pair[0]), int(pair[1])))
        rare.sort(key=lambda pair: (-_sum(self.by_pair[pair]), int(pair[0]), int(pair[1])))
        return {"source_id": "analysis:pair-frequency", "gid": gid,
                "frequent_definition": f"Наблюдаемая направленная пара с числом переводов >= {min_transactions}.",
                "rare_definition": "Ровно один наблюдаемый перевод по направленной паре в окне выгрузки.",
                "total_pairs": len(pairs), "frequent_pairs_count": len(frequent), "rare_pairs_count": len(rare),
                "frequent": [self._pair_fact(*pair) for pair in frequent[:limit]],
                "rare": [self._pair_fact(*pair) for pair in rare[:limit]],
                "truncated": len(frequent) > limit or len(rare) > limit,
                "interpretation": "Частота описывает выгрузку и не определяет подозрительность. Реальных номеров карт в данных нет; используются gid."}
