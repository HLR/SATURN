"""Score release runs: accuracy per run, the mean over seeds, and (optionally) id-paired agreement
with reference result files.

    python scripts/release_score.py [release/saturn/experiments] [--reference DIR]

Scores the runs whose file names carry the planner tag "_unified_" (the release runners write it).

--reference DIR: a directory holding the reported runs' result files under the same names the release
runners write ({mindcube_<slice>|mmsi}_s<seed>_*.json, force3d_{ref,puzzle}_nogt_s<seed>_*.json); each release run
is then paired id by id with the newest reference file of the same name.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
from collections import Counter, defaultdict

Q = "Qwen/Qwen3-VL-8B-Instruct"
PAPER = {"mindcube": 78.06, "mmsi": 48.77, "force3d-ref": 81.24, "force3d-puzzle": 85.88}   # published tables


def load(path):
    d = json.load(open(path))
    r = d.get("results", d) if isinstance(d, dict) else d
    return {str(x["id"]): bool(x.get("correct_final_answer")) for x in r if "id" in x}


def runs(root, mode="unified"):
    """{(bench, slice, seed): newest results file} whose name carries the planner tag ``mode``"""
    out = {}
    for path in sorted(glob.glob(f"{root}/*/{Q}/*.json"), key=os.path.getmtime):
        name = os.path.basename(path)
        if f"_{mode}_" not in name:
            continue
        m = (re.match(r"(mindcube)_(among|around|rotation)_s(\d)_", name) or re.match(r"(mmsi)()_s(\d)_", name)
             or re.match(r"(force3d)_(ref|puzzle)_nogt_s(\d)_", name))
        if m:
            out[(m.group(1), m.group(2), m.group(3) or "0")] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="?",
                    default="release/saturn/experiments" if os.path.isdir("release/saturn") else "experiments")
    ap.add_argument("--reference")
    a = ap.parse_args()
    R = runs(a.experiments)
    ref = runs(a.reference) if a.reference else {}
    pooled = defaultdict(lambda: defaultdict(dict))   # bench -> seed -> {id: correct}
    print(f"{'run':28s} {'n':>5s} {'acc':>6s}   paired with reference")
    for (bench, part, seed), path in sorted(R.items()):
        res = load(path)
        key = f"{bench}-{part}" if bench == "force3d" else bench
        pooled[key][seed].update({f"{part}:{i}": v for i, v in res.items()})
        line = f"{bench + (' ' + part if part else '') + ' s' + seed:28s} {len(res):5d} {100 * sum(res.values()) / max(1, len(res)):6.1f}"
        if (bench, part, seed) in ref:
            r = load(ref[(bench, part, seed)])
            ids = sorted(set(r) & set(res))
            c = Counter((r[i], res[i]) for i in ids)
            line += (f"   n={len(ids)} reference {100 * sum(r[i] for i in ids) / max(1, len(ids)):.1f}"
                     f"  flips: release-only {c[(False, True)]}, reference-only {c[(True, False)]}")
        print(line)
    print()
    for key, seeds in sorted(pooled.items()):
        accs = [100 * sum(v.values()) / len(v) for v in seeds.values() if v]
        spread = f" +/- {st.stdev(accs):.2f}" if len(accs) > 1 else ""
        print(f"{key:16s} {len(accs)} seed(s): {st.mean(accs):.2f}{spread}   (published: {PAPER.get(key, '-')})")


if __name__ == "__main__":
    main()
