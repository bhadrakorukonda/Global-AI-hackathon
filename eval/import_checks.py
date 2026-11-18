"""Copy the team's row checks into eval/dataset.jsonl (marks checked rows verified).

Input: a JSON export of the Row Checker's `checks`, either an object keyed by row id
({"version-python-latest": {"status": "checked", "checked_on": "2026-11-19", ...}}) or a list of objects with an "id".

    uv run python eval/import_checks.py checks.json            # preview, writes nothing
    uv run python eval/import_checks.py checks.json --write    # update dataset.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from staleguard.evaluation import RowCheck, apply_checks, load_dataset

DATASET = Path(__file__).parent / "dataset.jsonl"


def read_checks(path: Path) -> dict[str, RowCheck]:
    """Parse either export shape into {row id: RowCheck}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    pairs = raw.items() if isinstance(raw, dict) else ((r["id"], r) for r in raw)
    return {rid: RowCheck.model_validate(body) for rid, body in pairs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checks", type=Path)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--write", action="store_true", help="rewrite the dataset file (default: preview only)")
    args = ap.parse_args()

    items = load_dataset(args.dataset, include_unverified=True)
    updated, unchecked = apply_checks(items, read_checks(args.checks))
    verified = [i for i in updated if i.verified]
    changed = [u.id for u, o in zip(updated, items, strict=True)
               if u.verified and (u.label != o.label or u.ground_truth != o.ground_truth)]
    print(f"{len(verified)} verified ({sum(i.label == 'stale' for i in verified)} stale, "
          f"{sum(i.label == 'current' for i in verified)} current), "
          f"{len(items) - len(verified) - len(unchecked)} dropped, {len(unchecked)} not checked yet")
    if changed:
        print("checker changed label or ground truth on:", ", ".join(changed))
    if not args.write:
        print("preview only; pass --write to update", args.dataset)
        return

    header = [ln for ln in args.dataset.read_text(encoding="utf-8").splitlines() if ln.lstrip().startswith("//")]
    rows = [json.dumps(i.model_dump(by_alias=True), ensure_ascii=False) for i in updated]
    args.dataset.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    print("wrote", args.dataset)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
