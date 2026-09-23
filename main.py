"""Run `python main.py` to calculate and validate all three competition CSVs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from time import perf_counter


def main() -> int:
    started = perf_counter()  # Includes dependency imports and output validation.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="След — локальная аналитика финансовых цепочек")
    parser.add_argument("--data", type=Path, default=root / "data", help="Папка с тремя Parquet-файлами")
    parser.add_argument("--out", type=Path, default=root, help="Папка для трёх CSV (по умолчанию корень проекта)")
    parser.add_argument("--explain", metavar="GID", help="Вывести JSON с основаниями и транзакциями узла")
    parser.add_argument("--common", nargs="+", metavar="GID", help="Найти общих получателей выбранных источников")
    parser.add_argument("--debug", action="store_true", help="Показывать traceback при ошибке")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        from analytics_core import analyze, explain_node, find_common_recipients, write_outputs
        result = analyze(args.data)
        extra = {}
        if args.explain:
            extra["explanation"] = explain_node(result, args.explain)
        if args.common:
            extra["common_recipients"] = find_common_recipients(result, args.common)
        paths = write_outputs(result, args.out)
        elapsed = perf_counter() - started
        print(f"Узлов: {len(result.nodes_roles)}; рёбер: {result.graph.number_of_edges()}; "
              f"транзакций: {len(result.transactions)}")
        print(f"Кластеров: {len(result.clusters)}; топ: {len(result.top_nodes)}; "
              f"граница 4 колен: {int(result.nodes_roles.boundary.sum())}")
        print("Роли:", result.nodes_roles.role_label.value_counts().to_dict())
        print("Проверка CSV после повторного чтения: OK; NaN/пустые значения: 0")
        for path in paths:
            print(f"  {path}")
        print(f"Полный расчёт, запись и проверка: {elapsed:.3f} с; лимит <60 с: {'OK' if elapsed < 60 else 'ПРЕВЫШЕН'}")
        if extra:
            print(json.dumps(extra, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if elapsed < 60 else 3
    except ImportError as exc:
        logging.error("Не хватает зависимости: %s. Выполните python -m pip install -r requirements.txt", exc)
        return 2
    except Exception as exc:
        logging.error("Расчёт завершился ошибкой: %s", exc, exc_info=args.debug)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
