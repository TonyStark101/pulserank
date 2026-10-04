import argparse
import json

from .api import serve
from .demo import seed
from .storage import Store
from .streaming import EventTimePipeline, VerificationError


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pulserank", description="Real-time recommendation platform")
    parser.add_argument("--db", default="data/pulserank.db")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="seed a deterministic streaming-media scenario")
    run = commands.add_parser("serve", help="run the recommendation API and dashboard")
    run.add_argument("--port", type=int, default=8787)
    run.add_argument("--model", default="artifacts/model", help="trained model artifact directory")
    run.add_argument("--no-model", action="store_true", help="serve the heuristic model only")
    pipeline = commands.add_parser("pipeline", help="process accepted events into the local lakehouse")
    pipeline.add_argument("--lake", default="data/lake")
    pipeline.add_argument("--watermark-minutes", type=float, default=10)
    pipeline.add_argument("--batch-size", type=int, default=1000)
    repair = commands.add_parser("repair-late", help="backfill quarantined late events")
    repair.add_argument("--lake", default="data/lake")
    repair.add_argument("--watermark-minutes", type=float, default=10)
    verify = commands.add_parser("verify-pipeline", help="replay silver data and verify gold output")
    verify.add_argument("--lake", default="data/lake")
    verify.add_argument("--watermark-minutes", type=float, default=10)
    ml_demo = commands.add_parser("ml-demo", help="seed a deterministic training population")
    ml_demo.add_argument("--users", type=int, default=240)
    train_parser = commands.add_parser("train", help="train and evaluate retrieval and ranking models")
    train_parser.add_argument("--output", default="artifacts/model")
    train_parser.add_argument("--dimensions", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=35)
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--tracking-uri")
    train_parser.add_argument("--no-track", action="store_true")
    benchmark = commands.add_parser("benchmark-serving", help="measure end-to-end recommendation latency")
    benchmark.add_argument("--model", default="artifacts/model")
    benchmark.add_argument("--requests", type=int, default=1000)
    benchmark.add_argument("--output", default="benchmarks/serving-latency.json")
    analysis = commands.add_parser("analyze-experiment", help="compute CUPED-adjusted experiment outcomes")
    analysis.add_argument("--experiment", default="ranker-v1")
    analysis.add_argument("--window-days", type=float, default=7)
    args = parser.parse_args(argv)
    store = Store(args.db)
    if args.command == "demo":
        print(json.dumps(seed(store), indent=2))
    elif args.command == "serve":
        serve(store, port=args.port, model_path=None if args.no_model else args.model)
    elif args.command == "ml-demo":
        from .ml.data import seed_ml_population
        print(json.dumps(seed_ml_population(store, users=args.users), indent=2))
    elif args.command == "train":
        from .ml.training import train
        report, _, _ = train(
            store, output=args.output, dimensions=args.dimensions, epochs=args.epochs,
            seed=args.seed, tracking_uri=args.tracking_uri, track=not args.no_track,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "benchmark-serving":
        from .benchmark import benchmark_serving
        print(json.dumps(benchmark_serving(args.model, args.requests, args.output), indent=2, sort_keys=True))
    elif args.command == "analyze-experiment":
        from .experimentation import analyze_experiment
        print(json.dumps(analyze_experiment(store, args.experiment, args.window_days), indent=2, sort_keys=True))
    else:
        engine = EventTimePipeline(store, args.lake, args.watermark_minutes * 60)
        if args.command == "pipeline":
            result = engine.run_all(batch_size=args.batch_size)
        elif args.command == "repair-late":
            result = engine.repair_late()
        else:
            try:
                result = engine.verify()
            except VerificationError as exc:
                parser.error(str(exc))
        print(json.dumps(result, indent=2, sort_keys=True))
