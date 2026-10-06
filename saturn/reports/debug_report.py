#!/usr/bin/env python3
"""Generate self-contained HTML debug reports for SATURN pipeline runs.

For each sample in a result JSON, produces an HTML file with every
intermediate pipeline stage: question, images, planner, assembled prompt,
generated code, execution result, 3D scene and timing. Also produces an
index.html linking to all reports.

Usage:
    python -m saturn.reports.debug_report \
        --result_json experiments/mmsi/Qwen/Qwen3-VL-8B-Instruct/mmsi_s0_<timestamp>.json \
        --prompt_file prompts/vqa.txt --output_dir reports/mmsi_s0
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate HTML debug reports for SATURN results."
    )
    p.add_argument("--result_json", required=True, help="Path to result JSON file.")
    p.add_argument(
        "--prompt_file", default=None, help="Prompt template .txt used for this run."
    )
    p.add_argument("--program_cache", default=None, help="Program cache JSON.")
    p.add_argument("--planner_cache", default=None, help="Planner cache JSON.")
    p.add_argument(
        "--output_dir", default=None, help="Directory for HTML reports (default: auto)."
    )
    p.add_argument(
        "--image_dir",
        default=None,
        help="Directory containing sample images ({id}_{n}.jpg). Default: where the MMSI loader extracts them "
             "($MMSI_DATA_ROOT/images, else data/mmsi/images).",
    )
    p.add_argument(
        "--scene_dir",
        default=None,
        help="Directory containing scene JSONs ({id}.json). Auto-detected if omitted.",
    )
    p.add_argument(
        "--samples",
        default=None,
        help="Comma-separated sample indices to generate (default: all).",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# HTML rendering
# ═══════════════════════════════════════════════════════════════════════

from saturn.log import configure  # noqa: E402
from saturn.reports.assets import INDEX_CSS, template_block  # noqa: E402
from saturn.reports.helpers import default_mmsi_image_dir, esc  # noqa: E402
from saturn.reports.sections import SECTIONS, build_context  # noqa: E402


def render_sample_report(
    sample: Dict,
    idx: int,
    total: int,
    *,
    result_json: str,
    prompt_template: Optional[str],
    program_cache: Optional[Dict],
    planner_cache: Optional[Dict],
    image_dir: str,
    scene_dir: Optional[str],
) -> str:
    """Render a full HTML report for one sample."""
    ctx = build_context(
        sample,
        idx,
        total,
        result_json=result_json,
        prompt_template=prompt_template,
        program_cache=program_cache,
        planner_cache=planner_cache,
        image_dir=image_dir,
        scene_dir=scene_dir,
    )
    parts: List[str] = []
    for section in SECTIONS:
        parts.extend(section(ctx))
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Index page
# ═══════════════════════════════════════════════════════════════════════


# ── MMSI-Bench taxonomy ──────────────────────────────────────────────
# Maps raw `question_type` strings (as stored in MMSI_Bench.parquet and
# echoed in result JSONs) to (supercategory, short_subcategory_label).
MMSI_TAXONOMY: Dict[str, tuple] = {
    # Positional Relationship family
    "Positional Relationship (Cam.–Cam.)": ("Positional Relationship", "cam-cam"),
    "Positional Relationship (Cam.–Obj.)": ("Positional Relationship", "cam-obj"),
    "Positional Relationship (Cam.–Reg.)": ("Positional Relationship", "cam-reg"),
    "Positional Relationship (Obj.–Obj.)": ("Positional Relationship", "obj-obj"),
    "Positional Relationship (Obj.–Reg.)": ("Positional Relationship", "obj-reg"),
    "Positional Relationship (Reg.–Reg.)": ("Positional Relationship", "reg-reg"),
    # Attribute family
    "Attribute (Appr.)": ("Attribute", "appr"),
    "Attribute (Meas.)": ("Attribute", "meas"),
    # Motion family
    "Motion (Cam.)": ("Motion", "motion-cam"),
    "Motion (Obj.)": ("Motion", "motion-obj"),
    # Multi-step reasoning (stands alone)
    "MSR": ("Multi-Step Reasoning", "MSR"),
}

# Order used when rendering the taxonomy table.
MMSI_SUPERCATEGORY_ORDER = [
    "Positional Relationship",
    "Attribute",
    "Motion",
    "Multi-Step Reasoning",
]


def _classify_mmsi(qtype: str, fallback: Optional[str] = None) -> tuple:
    """Return (supercategory, subcategory) for a raw MMSI question_type.

    Accepts either the long-form MMSI string (e.g. "Positional Relationship
    (Cam.–Reg.)") or the short-name subcategory (e.g. "cam-reg"). Falls
    back to the provided ``fallback`` short-name if ``qtype`` is
    missing/empty. Otherwise returns ('Other', qtype) so the row still
    renders.
    """
    if qtype in MMSI_TAXONOMY:
        return MMSI_TAXONOMY[qtype]
    # Short-name lookup (invert the taxonomy).
    for raw, (super_cat, sub_cat) in MMSI_TAXONOMY.items():
        if sub_cat == qtype:
            return (super_cat, sub_cat)
    if fallback:
        for raw, (super_cat, sub_cat) in MMSI_TAXONOMY.items():
            if sub_cat == fallback:
                return (super_cat, sub_cat)
    return ("Other", qtype or fallback or "?")


def _render_mmsi_taxonomy_table(
    results: List[Dict], fallback_category: Optional[str] = None
) -> str:
    """Render an MMSI supercategory → subcategory breakdown table.

    Returns empty string if the results don't look like MMSI data (no
    sample maps to a known MMSI question_type).
    """
    # Bucket: super -> sub -> [correct, wrong, error]
    buckets: Dict[str, Dict[str, List[int]]] = {}
    any_mmsi = False
    for r in results:
        qt = r.get("question_type", "")
        super_cat, sub_cat = _classify_mmsi(qt, fallback=fallback_category)
        if super_cat != "Other":
            any_mmsi = True
        buckets.setdefault(super_cat, {}).setdefault(sub_cat, [0, 0, 0])
        correct = r.get("correct_final_answer")
        if correct is True:
            buckets[super_cat][sub_cat][0] += 1
        elif correct is False:
            buckets[super_cat][sub_cat][1] += 1
        else:
            buckets[super_cat][sub_cat][2] += 1

    if not any_mmsi:
        return ""

    # Render ordered
    rows = []
    super_order = [s for s in MMSI_SUPERCATEGORY_ORDER if s in buckets]
    super_order += [s for s in buckets if s not in super_order]

    grand_c = grand_w = grand_e = 0
    for super_cat in super_order:
        subs = buckets[super_cat]
        # Supercategory totals
        s_c = sum(v[0] for v in subs.values())
        s_w = sum(v[1] for v in subs.values())
        s_e = sum(v[2] for v in subs.values())
        s_n = s_c + s_w + s_e
        s_acc = 100 * s_c / s_n if s_n else 0.0
        grand_c += s_c
        grand_w += s_w
        grand_e += s_e

        rows.append(
            f"<tr class='super-row'>"
            f"<td colspan='2'><b>{esc(super_cat)}</b></td>"
            f"<td>{s_n}</td>"
            f"<td class='correct'>{s_c}</td>"
            f"<td class='wrong'>{s_w}</td>"
            f"<td class='error'>{s_e}</td>"
            f"<td><b>{s_acc:.1f}%</b></td>"
            f"</tr>"
        )
        # Subcategory rows (alphabetical within the supercategory)
        for sub_cat in sorted(subs.keys()):
            c, w, e = subs[sub_cat]
            n_sub = c + w + e
            acc_sub = 100 * c / n_sub if n_sub else 0.0
            rows.append(
                f"<tr class='sub-row'>"
                f"<td></td>"
                f"<td>{esc(sub_cat)}</td>"
                f"<td>{n_sub}</td>"
                f"<td class='correct'>{c}</td>"
                f"<td class='wrong'>{w}</td>"
                f"<td class='error'>{e}</td>"
                f"<td>{acc_sub:.1f}%</td>"
                f"</tr>"
            )

    # Grand total
    grand_n = grand_c + grand_w + grand_e
    grand_acc = 100 * grand_c / grand_n if grand_n else 0.0
    rows.append(
        f"<tr class='total-row'>"
        f"<td colspan='2'><b>TOTAL</b></td>"
        f"<td><b>{grand_n}</b></td>"
        f"<td class='correct'><b>{grand_c}</b></td>"
        f"<td class='wrong'><b>{grand_w}</b></td>"
        f"<td class='error'><b>{grand_e}</b></td>"
        f"<td><b>{grand_acc:.1f}%</b></td>"
        f"</tr>"
    )

    return (
        "<h2>MMSI-Bench Taxonomy Breakdown</h2>\n"
        "<table class='summary-table taxonomy-table'>\n"
        "<tr><th>Supercategory</th><th>Subcategory</th>"
        "<th>N</th><th>Correct</th><th>Wrong</th><th>Err</th>"
        "<th>Acc</th></tr>\n"
        + "\n".join(rows)
        + "\n</table>\n"
    )


def render_index(results: List[Dict], result_json: str, output_dir: str,
                 fallback_category: Optional[str] = None) -> str:
    """Render the index.html page with summary table."""
    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct_final_answer"))
    n_wrong = sum(1 for r in results if r.get("correct_final_answer") is False)
    n_error = sum(1 for r in results if r.get("correct_final_answer") is None)
    n_rf = sum(1 for r in results if r.get("planner"))
    n_vlm = sum(1 for r in results if r.get("vlm_fallback"))
    acc = 100 * n_correct / max(n, 1)
    mmsi_taxonomy_html = _render_mmsi_taxonomy_table(results, fallback_category=fallback_category)

    rows = []
    for i, r in enumerate(results):
        correct = r.get("correct_final_answer")
        cls = "correct" if correct else ("wrong" if correct is False else "error")
        status = "OK" if correct else ("WRONG" if correct is False else "ERR")
        fname = f"sample_{r['id']}.html"
        has_planner = "Y" if r.get("planner") else "N"
        qtype = r.get("question_type", "?")
        rows.append(f"""<tr>
<td><a href="{fname}">{esc(str(r["id"]))}</a></td>
<td class="{cls}">{status}</td>
<td>{esc(str(r.get("final_answer_text", "?")))}</td>
<td>{esc(r.get("ground_truth_answer", "?"))}</td>
<td>{esc(qtype)}</td>
<td>{has_planner}</td>
<td>{r.get("objects_count", 0)}</td>
<td>{r.get("retry_count", 0)}</td>
<td title="{esc(r["query"][:120])}">{esc(r["query"][:80])}{"..." if len(r["query"]) > 80 else ""}</td>
</tr>""")

    return template_block("report.html", "index").substitute(
        stem=esc(Path(result_json).stem),
        css=INDEX_CSS,
        result_json=esc(result_json),
        n=n,
        acc=f"{acc:.0f}",
        n_correct=n_correct,
        n_wrong=n_wrong,
        n_error=n_error,
        n_rf=n_rf,
        n_vlm=n_vlm,
        mmsi_taxonomy_html=mmsi_taxonomy_html,
        rows="".join(rows),
    )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    configure()
    args = parse_args()

    # Load inputs
    with open(args.result_json) as f:
        result_data = json.load(f)
    results = result_data["results"]
    print(f"Loaded {len(results)} samples from {args.result_json}")

    prompt_template = None
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file) as f:
            prompt_template = f.read()
        print(f"Prompt template: {args.prompt_file} ({len(prompt_template):,} chars)")

    program_cache = None
    if args.program_cache and os.path.exists(args.program_cache):
        with open(args.program_cache) as f:
            program_cache = json.load(f)
        print(f"Program cache: {args.program_cache} ({len(program_cache)} entries)")

    planner_cache = None
    if args.planner_cache and os.path.exists(args.planner_cache):
        with open(args.planner_cache) as f:
            planner_cache = json.load(f)
        print(f"Planner cache: {args.planner_cache} ({len(planner_cache)} entries)")

    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        stem = Path(args.result_json).stem
        output_dir = f"reports/{stem}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}/")

    # Filter samples
    if args.samples:
        indices = [int(x.strip()) for x in args.samples.split(",")]
    else:
        indices = list(range(len(results)))

    # Generate per-sample reports
    for i, idx in enumerate(indices):
        sample = results[idx]
        print(f"  [{i + 1}/{len(indices)}] id={sample['id']}...", end="", flush=True)
        html = render_sample_report(
            sample,
            idx,
            len(results),
            result_json=args.result_json,
            prompt_template=prompt_template,
            program_cache=program_cache,
            planner_cache=planner_cache,
            image_dir=args.image_dir or default_mmsi_image_dir(),
            scene_dir=args.scene_dir,
        )
        fname = f"sample_{sample['id']}.html"
        with open(os.path.join(output_dir, fname), "w") as f:
            f.write(html)
        correct = sample.get("correct_final_answer")
        tag = "ok" if correct else ("WRONG" if correct is False else "ERR")
        print(f" {tag}")

    # Rewrite reports to have correct prev/next links
    for i, idx in enumerate(indices):
        sample = results[idx]
        fname = os.path.join(output_dir, f"sample_{sample['id']}.html")
        with open(fname) as f:
            content = f.read()

        # Fix prev link
        if i > 0:
            prev_id = results[indices[i - 1]]["id"]
            content = content.replace(
                f"sample_{sample['id']}_prev.html",
                f"sample_{prev_id}.html",
            )
        else:
            # Remove broken prev link
            content = re.sub(r"<a href='sample_[^']*_prev\.html'>Prev</a>", "", content)

        # Fix next link
        if i < len(indices) - 1:
            next_id = results[indices[i + 1]]["id"]
            content = content.replace(
                f"sample_{sample['id']}_next.html",
                f"sample_{next_id}.html",
            )
        else:
            content = re.sub(r"<a href='sample_[^']*_next\.html'>Next</a>", "", content)

        with open(fname, "w") as f:
            f.write(content)

    # Generate index
    # Try to infer a short-name fallback category from the result-file
    # description (e.g. "mmsi cam-reg" → "cam-reg") or the file path.
    fallback_cat: Optional[str] = None
    desc = (result_data.get("description") or "").strip()
    if desc:
        parts = desc.split()
        if parts:
            tail = parts[-1]
            if tail in {sub for (_sup, sub) in MMSI_TAXONOMY.values()}:
                fallback_cat = tail
    if fallback_cat is None:
        stem = Path(args.result_json).stem  # e.g. "mmsi_s0_cam-reg"
        for sub in {s for (_sup, s) in MMSI_TAXONOMY.values()}:
            if stem.endswith("_" + sub) or stem.endswith("-" + sub):
                fallback_cat = sub
                break
    index_html = render_index(results, args.result_json, output_dir,
                              fallback_category=fallback_cat)
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(index_html)
    print(f"\nDone! Open {output_dir}/index.html")


if __name__ == "__main__":
    main()
