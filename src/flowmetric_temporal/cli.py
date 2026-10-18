"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .extract import build_dataset
from .infer import predict_manifest
from .train import train_model


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="flowmetric-temporal")
    commands = root.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="create episode tensors from annotation JSONL")
    extract.add_argument("--annotations", type=Path, required=True)
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--frames", type=int, default=16)
    extract.add_argument("--crop-size", type=int, default=112)

    train = commands.add_parser("train", help="train the temporal classifier")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--device", default="auto")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=17)

    predict = commands.add_parser("predict", help="classify extracted episodes")
    predict.add_argument("--manifest", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--out", type=Path, required=True)
    predict.add_argument("--split")
    predict.add_argument("--batch-size", type=int, default=16)
    predict.add_argument("--device", default="auto")
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
        print(f"wrote {path}")
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
        print(f"best validation metrics: {metrics}")
    else:
        predict_manifest(
            args.manifest,
            args.checkpoint,
            args.out,
            split=args.split,
            batch_size=args.batch_size,
            device_name=args.device,
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

