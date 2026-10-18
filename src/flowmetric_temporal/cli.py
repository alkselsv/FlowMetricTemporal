"""Точка входа командной строки."""

from __future__ import annotations

import argparse
from pathlib import Path

from .bootstrap import bootstrap_annotations
from .end_to_end import run_current_data
from .extract import build_dataset
from .infer import predict_manifest
from .train import train_model


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="flowmetric-temporal")
    commands = root.add_subparsers(dest="command", required=True)

    extract = commands.add_parser(
        "extract", help="создать тензоры эпизодов из JSONL с аннотациями"
    )
    extract.add_argument("--annotations", type=Path, required=True)
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--frames", type=int, default=16)
    extract.add_argument("--crop-size", type=int, default=112)

    train = commands.add_parser("train", help="обучить временной классификатор")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--device", default="auto")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=17)

    predict = commands.add_parser("predict", help="классифицировать извлечённые эпизоды")
    predict.add_argument("--manifest", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--out", type=Path, required=True)
    predict.add_argument("--split")
    predict.add_argument("--batch-size", type=int, default=16)
    predict.add_argument("--device", default="auto")

    bootstrap = commands.add_parser(
        "bootstrap-current",
        help="создать проверяемые аннотации из контрольного набора FlowMetric",
    )
    bootstrap.add_argument("--flowmetric-root", type=Path, required=True)
    bootstrap.add_argument("--out", type=Path, required=True)
    bootstrap.add_argument(
        "--tracks",
        type=Path,
        default=Path("data/output/service_verifier_tables_run/tracks.json"),
        help="путь к tracks.json относительно корня FlowMetric",
    )
    bootstrap.add_argument("--max-weak-per-class", type=int, default=6)

    current = commands.add_parser(
        "current-data",
        help="разметить данные FlowMetric, извлечь признаки, обучить модель и получить предсказания",
    )
    current.add_argument("--flowmetric-root", type=Path, required=True)
    current.add_argument("--out", type=Path, required=True)
    current.add_argument(
        "--tracks",
        type=Path,
        default=Path("data/output/service_verifier_tables_run/tracks.json"),
    )
    current.add_argument("--frames", type=int, default=16)
    current.add_argument("--crop-size", type=int, default=112)
    current.add_argument("--epochs", type=int, default=30)
    current.add_argument("--batch-size", type=int, default=8)
    current.add_argument("--device", default="auto")
    current.add_argument("--prepare-only", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "extract":
        path = build_dataset(
            args.annotations,
            args.out,
            frames_per_episode=args.frames,
            crop_size=args.crop_size,
        )
        print(f"записано: {path}")
    elif args.command == "train":
        metrics = train_model(
            args.manifest,
            args.out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device_name=args.device,
            workers=args.workers,
            seed=args.seed,
        )
        print(f"лучшие метрики на проверочной выборке: {metrics}")
    elif args.command == "predict":
        predict_manifest(
            args.manifest,
            args.checkpoint,
            args.out,
            split=args.split,
            batch_size=args.batch_size,
            device_name=args.device,
        )
        print(f"записано: {args.out}")
    elif args.command == "bootstrap-current":
        result = bootstrap_annotations(
            args.flowmetric_root,
            args.out,
            tracks_relative=args.tracks,
            max_weak_per_class=args.max_weak_per_class,
        )
        print(result)
    else:
        result = run_current_data(
            args.flowmetric_root,
            args.out,
            tracks_relative=args.tracks,
            frames_per_episode=args.frames,
            crop_size=args.crop_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device_name=args.device,
            prepare_only=args.prepare_only,
        )
        print(result)


if __name__ == "__main__":
    main()
