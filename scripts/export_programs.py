#!/usr/bin/env python3
"""Export {sample id: program} from SATURN results files, for replay with --programs_by_id.

    python scripts/export_programs.py OUT.json RESULTS.json [RESULTS.json ...]

The results store each program wrapped in the program header (CODE_TEMPLATE); this
keeps only the program body the code LLM wrote (the final one, after any retry).
"""
import json
import sys
import textwrap

HEADER_END = "formula_helpers(score, scene)"   # in both header forms: `camera = ...` or `_h = ...`


def program_body(wrapped: str) -> str:
    lines = wrapped.split("\n")
    start = next(i for i, line in enumerate(lines) if HEADER_END in line) + 1
    # `_h` header form: the next line binds the helpers from _h (camera, view, ...)
    if start < len(lines) and "= _h[" in lines[start]:
        start += 1
    return textwrap.dedent("\n".join(lines[start:])).strip()


def main(out, results_files):
    programs = {}
    for path in results_files:
        data = json.load(open(path))
        data = data if isinstance(data, list) else data["results"]
        for rec in data:
            if rec.get("program_code"):
                programs[str(rec["id"])] = program_body(rec["program_code"])
    with open(out, "w", encoding="utf-8") as f:
        json.dump(programs, f, indent=1, sort_keys=True)
    print(f"{len(programs)} programs -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
