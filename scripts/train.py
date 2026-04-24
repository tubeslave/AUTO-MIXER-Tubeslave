from __future__ import annotations

import time


def main() -> None:
    print("Starting training...")
    for epoch in range(3):
        print(f"epoch={epoch} loss={1.0 / (epoch + 1):.4f}")
        time.sleep(0.2)
    print("Training finished.")


if __name__ == "__main__":
    main()
