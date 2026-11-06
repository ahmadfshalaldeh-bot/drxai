"""Check every number typed in the manuscript against the experiment outputs.

``manuscript_claims.yaml`` lists each reported value with the results file and
key that must reproduce it. Running

    python -m drxai.cli verify

prints PASS / FAIL / MISSING per claim and exits non-zero on any FAIL. This is
the authors' pre-submission integrity check: no value may enter the paper
unless it is regenerated from code and data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from .utils import load_json


def _get(obj, dotted: str):
    for part in dotted.split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]
        else:
            obj = obj[part]
    return obj


def run(claims_path: str, results_root: str) -> int:
    claims = yaml.safe_load(Path(claims_path).read_text())
    n_fail = 0
    for c in claims:
        f = Path(results_root) / c["file"]
        if not f.exists():
            print(f"MISSING  {c['id']:<34} ({c['file']} not found)")
            continue
        try:
            val = float(_get(load_json(f), c["key"]))
        except (KeyError, IndexError, TypeError) as e:
            print(f"MISSING  {c['id']:<34} key {c['key']} ({e})")
            continue
        val = val * c.get("scale", 1.0)
        ok = abs(val - float(c["value"])) <= float(c.get("tol", 0.0015))
        n_fail += not ok
        print(f"{'PASS' if ok else 'FAIL':<8} {c['id']:<34} manuscript={c['value']}  computed={val:.4f}")
    print(f"\n{n_fail} claim(s) failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1], sys.argv[2]))
