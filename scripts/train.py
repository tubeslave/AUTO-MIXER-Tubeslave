from __future__ import annotations

import argparse
import time


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the sample training loop.")
    parser.add_argument("--epochs", type=int, default=3)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"epoch={epoch} loss={1.0 / (epoch + 1):.4f}")
        time.sleep(0.2)
    print("Training finished.")


if __name__ == "__main__":
    main()
