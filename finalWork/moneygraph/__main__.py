import argparse
import json
from pathlib import Path
import sys
import time
from .data import load, DataError
from .pipeline import run
from .synthetic import generate


def main(argv=None):
    parser=argparse.ArgumentParser(description='MoneyGraph: локальный анализ неполной транзакционной сети')
    commands=parser.add_subparsers(dest='command',required=True)
    execute=commands.add_parser('run',help='Обработать три входных файла')
    execute.add_argument('--data',default='data')
    execute.add_argument('--out',default='results')
    execute.add_argument('--format',choices=['auto','parquet','csv'],default='auto')
    execute.add_argument('--top',type=int,default=20)
    execute.add_argument('--stability',action='store_true')
    execute.add_argument('--config',help='JSON с переопределениями порогов')
    demo=commands.add_parser('demo',help='Создать явно синтетические CSV и обработать их')
    demo.add_argument('--out',default='results/demo')
    demo.add_argument('--size',type=int,default=2248)
    demo.add_argument('--stability',action='store_true')
    experiment=commands.add_parser('benchmark',help='Контролируемое раскрытие исходящих переводов')
    experiment.add_argument('--data',default='data')
    experiment.add_argument('--out',default='results/benchmark')
    experiment.add_argument('--format',choices=['auto','parquet','csv'],default='auto')
    experiment.add_argument('--budget',type=int,default=10)
    experiment.add_argument('--random-runs',type=int,default=3)
    web=commands.add_parser('serve',help='Анализ данных и локальный интерфейс с настоящим ИИ')
    web.add_argument('--data',default='data')
    web.add_argument('--out',default='results/real')
    web.add_argument('--format',choices=['auto','parquet','csv'],default='auto')
    web.add_argument('--port',type=int,default=8520)
    web.add_argument('--env-file',default='.env')
    args=parser.parse_args(argv)
    try:
        started=time.perf_counter()
        if args.command=='serve':
            from .server import serve
            data=load(args.data,args.format)
            report=run(data,args.out,stability=True,started=started)
            serve(data,report,args.out,args.port,args.env_file)
            return 0
        elif args.command=='demo':
            source=generate(Path(args.out)/'input',args.size)
            data=load(source,'csv')
            report=run(data,args.out,stability=args.stability,started=started)
        elif args.command=='run':
            config=json.loads(Path(args.config).read_text(encoding='utf-8')) if args.config else None
            if config:
                from .engine import DEFAULTS
                unknown=set(config)-set(DEFAULTS)
                if unknown:
                    raise ValueError(f'Неизвестные параметры: {sorted(unknown)}')
                if 'priority_weights' in config and (len(config['priority_weights'])!=5 or
                     any(not isinstance(w,(int,float)) or w<0 for w in config['priority_weights']) or sum(config['priority_weights'])<=0):
                    raise ValueError('priority_weights: пять неотрицательных весов с положительной суммой')
                for key,value in config.items():
                    if key!='priority_weights' and (not isinstance(value,(int,float)) or value<0):
                        raise ValueError(f'{key}: ожидается неотрицательное число')
            report=run(load(args.data,args.format),args.out,args.top,args.stability,config,started)
        else:
            from .experiments import benchmark
            result=benchmark(load(args.data,args.format),args.out,args.budget,args.random_runs)
            print(json.dumps(dict(output=str(Path(args.out).resolve()),experiments=len(result)),ensure_ascii=False))
            return 0
        print(json.dumps(dict(output=str(Path(args.out).resolve()),nodes=report['meta']['n_nodes'],
                              synthetic=report['meta'].get('synthetic',False),runtime_seconds=report['meta']['runtime_seconds']),ensure_ascii=False))
        return 0
    except (DataError,ValueError,OSError,KeyError) as error:
        print(f'ОШИБКА: {error}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
