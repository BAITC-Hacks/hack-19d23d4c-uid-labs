from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
import csv
import hashlib
import json
import math


class DataError(ValueError):
    pass


def integer(value, name):
    if isinstance(value, (bool, float)):
        raise DataError(f"{name}: ожидается целое без преобразования через float: {value!r}")
    try:
        number = int(str(value))
    except (ValueError, TypeError) as error:
        raise DataError(f"{name}: неверное целое {value!r}") from error
    if name.endswith('gid') or name in ('src', 'dst'):
        if not -(2**63) <= number < 2**63:
            raise DataError(f"{name}: идентификатор вне int64")
    return number


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ('1', 'true'):
        return True
    if str(value).strip().lower() in ('0', 'false'):
        return False
    raise DataError(f"is_seed: ожидается true/false/1/0, получено {value!r}")


def money(value):
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise DataError(f"Некорректная сумма: {value!r}") from error
    if not math.isfinite(number) or number <= 0:
        raise DataError(f"Сумма должна быть положительной и конечной: {value!r}")
    return number


def day(value):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).date().isoformat()
    except ValueError as error:
        raise DataError(f"date: ожидается ISO-дата, получено {value!r}") from error


def aggregate(transactions):
    grouped = defaultdict(lambda: [0.0, 0])
    for row in transactions:
        values = grouped[row['src'], row['dst']]
        values[0] += row['sum_kzt']
        values[1] += 1
    return [{'src': a, 'dst': b, 'sum_kzt': value[0], 'n_tx': value[1], 'depth': 0}
            for (a, b), value in sorted(grouped.items())]


@dataclass
class Dataset:
    nodes: dict
    edges: list
    transactions: list
    meta: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def from_records(node_rows, edge_rows, tx_rows, meta=None):
    nodes = {}
    warnings = []
    for row in node_rows:
        gid = integer(row['gid'], 'gid')
        if gid in nodes:
            raise DataError(f"Повторный gid в nodes: {gid}")
        depth = integer(row['depth'], 'depth')
        if depth < 0:
            raise DataError('depth не может быть отрицательным')
        nodes[gid] = dict(gid=gid, depth=depth, is_seed=boolean(row['is_seed']))
    if not nodes:
        raise DataError('Таблица nodes пуста')
    tx = []
    for index, row in enumerate(tx_rows, 1):
        src, dst = integer(row['src'], 'src'), integer(row['dst'], 'dst')
        if src not in nodes or dst not in nodes:
            raise DataError(f"Транзакция {index}: неизвестный endpoint {src}->{dst}")
        tx.append(dict(tx_id=f'tx-{index:07d}', src=src, dst=dst,
                       date=day(row['date']), sum_kzt=money(row['sum_kzt'])))
    edges = []
    pairs = set()
    for row in edge_rows:
        src, dst = integer(row['src'], 'src'), integer(row['dst'], 'dst')
        if src not in nodes or dst not in nodes:
            raise DataError(f"Ребро: неизвестный endpoint {src}->{dst}")
        if (src, dst) in pairs:
            raise DataError(f"Повторная агрегированная пара {src}->{dst}")
        pairs.add((src, dst))
        count = integer(row['n_tx'], 'n_tx')
        if count < 1:
            raise DataError('n_tx должен быть положительным')
        edges.append(dict(src=src, dst=dst, sum_kzt=money(row['sum_kzt']),
                          n_tx=count, depth=integer(row.get('depth', 0), 'edge.depth')))
    actual = {(r['src'], r['dst']): r for r in aggregate(tx)}
    if pairs != set(actual):
        raise DataError('Наборы пар edges и агрегата transactions различаются')
    for edge in edges:
        check = actual[edge['src'], edge['dst']]
        if edge['n_tx'] != check['n_tx'] or not math.isclose(
                edge['sum_kzt'], check['sum_kzt'], rel_tol=1e-9, abs_tol=0.02):
            raise DataError(f"edges/transactions не согласуются: {edge['src']}->{edge['dst']}")
    if any(r['sum_kzt'] < 5000 for r in tx):
        warnings.append('Есть транзакции ниже 5 000 KZT; это отличается от описания кейса.')
    if any(r['date'][:7] != '2026-07' for r in tx):
        warnings.append('Период содержит даты вне июля 2026; границы окна вычислены по данным.')
    if any(n['depth'] > 4 for n in nodes.values()):
        warnings.append('Есть depth>4; они также считаются границей наблюдения.')
    if not any(n['is_seed'] for n in nodes.values()):
        warnings.append('Нет seed-клиентов: показатели связи с seed равны нулю.')
    if any(n['is_seed'] and n['depth'] != 0 for n in nodes.values()):
        warnings.append('Есть seed с depth != 0; проверьте схему выгрузки.')
    return Dataset(nodes, sorted(edges, key=lambda r: (r['src'], r['dst'])), tx,
                   dict(meta or {}), warnings)


def _read(path, fmt, required):
    if fmt == 'parquet':
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise DataError('Для Parquet установите зависимости: python -m pip install -r requirements.txt') from error
        table = pq.read_table(path)
        columns = table.column_names
        rows = table.to_pylist()
    else:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            rows = list(reader)
    missing = set(required) - set(columns)
    if missing:
        raise DataError(f'{path.name}: отсутствуют колонки {sorted(missing)}')
    return rows


def load(directory, fmt='auto'):
    directory = Path(directory)
    names = ('nodes', 'edges', 'transactions')
    if fmt == 'auto':
        present=[(directory / f'{name}.parquet').exists() for name in names]
        if any(present) and not all(present):
            raise DataError('Неполный набор Parquet: требуются nodes, edges и transactions. Для CSV задайте --format csv.')
        fmt = 'parquet' if all((directory / f'{name}.parquet').exists() for name in names) else 'csv'
    required = {'nodes': ('gid', 'depth', 'is_seed'),
                'edges': ('src', 'dst', 'sum_kzt', 'n_tx', 'depth'),
                'transactions': ('src', 'dst', 'date', 'sum_kzt')}
    rows, hashes = {}, {}
    for name in names:
        path = directory / f'{name}.{fmt}'
        if not path.exists():
            raise DataError(f'Не найден файл: {path}')
        rows[name] = _read(path, fmt, required[name])
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = directory / 'metadata.json'
    meta = json.loads(metadata.read_text(encoding='utf-8')) if metadata.exists() else {}
    meta.update(source_format=fmt, input_sha256=hashes)
    meta.setdefault('synthetic', False)
    return from_records(rows['nodes'], rows['edges'], rows['transactions'], meta)


def write_csv(path, rows, columns):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
