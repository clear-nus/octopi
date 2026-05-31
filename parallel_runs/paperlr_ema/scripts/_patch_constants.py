"""Rewrite TRAIN_OBJECTS and VAL_OBJECTS list literals in src/utils/constants.py
from a fold JSON. Leaves everything else (TEST_OBJECTS, label dicts, etc.) untouched.

Usage:
    python scripts/_patch_constants.py scripts/kfold/fold_0.json
"""
import json
import re
import sys
from pathlib import Path

CONSTANTS = Path("src/utils/constants.py")


def render_list(name, items):
    body = ",\n    ".join(repr(x) for x in items)
    return f"{name} = [\n    {body},\n]"


def replace_list_block(src: str, name: str, items):
    # Match: NAME = [   ...balanced brackets...  ]
    # Use a non-greedy match across newlines; constants.py only has one list named X.
    pattern = re.compile(rf"^{name}\s*=\s*\[[^\]]*?\]", re.MULTILINE | re.DOTALL)
    if not pattern.search(src):
        raise SystemExit(f"could not find list assignment for {name}")
    return pattern.sub(render_list(name, items), src, count=1)


def main():
    fold_path = Path(sys.argv[1])
    fold = json.loads(fold_path.read_text())
    src = CONSTANTS.read_text()
    src = replace_list_block(src, "TRAIN_OBJECTS", fold["train_objects"])
    src = replace_list_block(src, "VAL_OBJECTS", fold["val_objects"])
    CONSTANTS.write_text(src)
    print(f"patched constants.py from {fold_path}: "
          f"train={len(fold['train_objects'])} val={len(fold['val_objects'])}")


if __name__ == "__main__":
    main()
