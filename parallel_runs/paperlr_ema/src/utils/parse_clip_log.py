"""Parse a CLIP training log.txt and report the best epoch by val mean per-property accuracy."""
import argparse
import json
import re
from pathlib import Path

VAL_RE = re.compile(
    r"VAL accuracies \[hardness, roughness, texture, combined\]: "
    r"([0-9.eE+-]+), ([0-9.eE+-]+), ([0-9.eE+-]+), ([0-9.eE+-]+)"
)
TEST_RE = re.compile(
    r"TEST accuracies \[hardness, roughness, texture, combined\]: "
    r"([0-9.eE+-]+), ([0-9.eE+-]+), ([0-9.eE+-]+), ([0-9.eE+-]+)"
)


def parse(log_path: Path):
    text = log_path.read_text()
    val_hits = [tuple(float(x) for x in m.groups()) for m in VAL_RE.finditer(text)]
    test_hits = [tuple(float(x) for x in m.groups()) for m in TEST_RE.finditer(text)]
    if not val_hits:
        return None
    best_idx = max(range(len(val_hits)), key=lambda i: sum(val_hits[i][:3]) / 3)
    val = val_hits[best_idx]
    test = test_hits[best_idx] if best_idx < len(test_hits) else (float("nan"),) * 4
    return {
        "epoch": best_idx + 1,
        "val_hardness": val[0],
        "val_roughness": val[1],
        "val_texture": val[2],
        "val_combined": val[3],
        "val_mean": sum(val[:3]) / 3,
        "test_hardness": test[0],
        "test_roughness": test[1],
        "test_texture": test[2],
        "test_combined": test[3],
        "test_mean": sum(test[:3]) / 3,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("--field", default=None,
                        help="Print only this field (e.g., val_mean). Default: full JSON.")
    args = parser.parse_args()
    result = parse(Path(args.log_path))
    if result is None:
        raise SystemExit(f"No VAL lines found in {args.log_path}")
    if args.field:
        print(result[args.field])
    else:
        print(json.dumps(result, indent=2))
