"""Per-section renderers for the per-sample debug report.

Each ``section_*`` function renders exactly one ``<h2>`` block of the report
from a shared :class:`ReportContext` (built by ``build_context``) and returns
its HTML fragments as a list of parts; ``render_sample_report`` joins all parts
with newlines.
"""

from __future__ import annotations

import base64
import io
import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from saturn.reports.assets import CSS, script_tag, template_block
from saturn.reports.helpers import (
    _collect_raw_azimuths_by_label,
    _extract_user_code,
    _json_default,
    b64_image,
    build_plotly_traces_json,
    esc,
    load_orientation_trace_for_sample,
    load_scene_for_sample,
    resolve_image_paths_for_sample,
)
from saturn.codegen.generator import CodeGenerator
from saturn.planning.query_planner import QueryPlanner
from saturn.scene.scene import Scene


@dataclass
class ReportContext:
    """Everything more than one section reads. Built once by :func:`build_context`."""

    sample: Dict
    idx: int
    total: int
    result_json: str
    prompt_template: Optional[str]
    program_cache: Optional[Dict]
    planner_cache: Optional[Dict]
    image_dir: str
    scene_dir: Optional[str]
    item_id: str
    correct: Optional[bool]
    status_badge: str
    planner_badge: str
    planner_data: Optional[Dict]
    planner_time: Any
    query_text: str
    grounding_results: List[Dict]
    scene_data: Optional[Dict]
    scene_path: Optional[str]
    scene_obj: Optional[Scene]
    orient_trace: Optional[Dict]
    raw_az_by_label: Dict[str, List[Dict]]
    task_formalization: str
    clarified_query_block: str
    scene_facts_block: str
    object_groundings_block: str = ""


def build_context(
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
) -> ReportContext:
    """Compute the shared per-sample state (status badges, planner/RF data, scene)."""
    item_id = str(sample["id"])
    correct = sample.get("correct_final_answer")
    status_badge = (
        '<span class="badge badge-ok">CORRECT</span>'
        if correct
        else '<span class="badge badge-wrong">WRONG</span>'
        if correct is False
        else '<span class="badge badge-err">ERROR</span>'
    )
    planner_data = sample.get("planner")
    planner_time = sample.get("planner_time_s", "?")
    planner_badge = '<span class="badge badge-rf">PLANNER</span>' if planner_data else ""
    query_text = sample.get("query", "")
    grounding_results = sample.get("grounding_results", [])
    task_formalization = ""
    clarified_query_block = ""
    scene_obj = None
    # Load the dumped scene for the facts block and the 3D view
    scene_data, scene_path = load_scene_for_sample(item_id, scene_dir, result_json)
    if scene_data:
        scene_obj = Scene.from_dict(scene_data)
    # Optional: load orientation trace (raw OA azimuths etc.) for this sample
    orient_trace = load_orientation_trace_for_sample(item_id, result_json)
    raw_az_by_label = _collect_raw_azimuths_by_label(orient_trace) if orient_trace else {}

    # Build task_formalization block from planner output for debug display.
    if planner_data:
        groundings = planner_data.get("object_groundings", [])
        if groundings:
            g_lines = ["OBJECT GROUNDINGS:"]
            for g in groundings:
                phrase = g.get("phrase", "").strip()
                description = g.get("description", "").strip()
                if phrase and description:
                    g_cam = g.get("cam_id")
                    if g_cam is not None:
                        g_lines.append(f"- {phrase}: {description} [cam_id={g_cam}]")
                    else:
                        g_lines.append(f"- {phrase}: {description}")
            if len(g_lines) > 1:
                task_formalization = "\n".join(g_lines)

    # Rebuild the blocks codegen received (pipeline/plan.py, pipeline/generate.py).
    object_groundings_block = ""
    if planner_data:
        object_groundings_block = QueryPlanner.format_groundings_block(
            planner_data.get("object_groundings") or []
        ) or ""

    # Reconstruct the scene_facts_block codegen received.
    scene_facts_block = ""
    if scene_obj is not None:
        try:
            if planner_data:
                scene_obj.set_planner_context(setup_caption=planner_data.get("setup_caption"))
            scene_facts_block = scene_obj.dump_facts_str()
        except Exception:
            scene_facts_block = ""
    return ReportContext(
        sample=sample,
        idx=idx,
        total=total,
        result_json=result_json,
        prompt_template=prompt_template,
        program_cache=program_cache,
        planner_cache=planner_cache,
        image_dir=image_dir,
        scene_dir=scene_dir,
        item_id=item_id,
        correct=correct,
        status_badge=status_badge,
        planner_badge=planner_badge,
        planner_data=planner_data,
        planner_time=planner_time,
        query_text=query_text,
        grounding_results=grounding_results,
        scene_data=scene_data,
        scene_path=scene_path,
        scene_obj=scene_obj,
        orient_trace=orient_trace,
        raw_az_by_label=raw_az_by_label,
        task_formalization=task_formalization,
        object_groundings_block=object_groundings_block,
        clarified_query_block=clarified_query_block,
        scene_facts_block=scene_facts_block,
    )


def section_header(ctx: ReportContext) -> List[str]:
    """Header (nav + title)."""
    sample = ctx.sample
    idx = ctx.idx
    total = ctx.total
    item_id = ctx.item_id
    status_badge = ctx.status_badge
    planner_badge = ctx.planner_badge
    parts: List[str] = []
    # ── Header ──
    parts.append(template_block("report.html", "header").substitute(
        item_id=item_id,
        css=CSS,
        prev_link="<a href='sample_" + str(sample.get("id", "")) + "_prev.html'>Prev</a>" if idx > 0 else "",
        next_link="<a href='sample_" + str(sample.get("id", "")) + "_next.html'>Next</a>" if idx < total - 1 else "",
        sample_no=idx + 1,
        total=total,
        status_badge=status_badge,
        planner_badge=planner_badge,
    ))
    return parts


def section_question(ctx: ReportContext) -> List[str]:
    """1. Question & Ground Truth (+ VLM discovery)."""
    sample = ctx.sample
    parts: List[str] = []
    # ── 1. Question & Ground Truth ──
    parts.append(f"""<h2>1. Question & Ground Truth</h2>
<div class="card">
<dl class="kv">
<dt>Question Type</dt><dd>{esc(sample.get("question_type", "?"))}</dd>
<dt>Ground Truth</dt><dd><strong>{esc(sample.get("ground_truth_answer", "?"))}</strong></dd>
<dt>Predicted</dt><dd><strong>{esc(str(sample.get("final_answer_text", "?")))}</strong></dd>
<dt>VLM Fallback</dt><dd>{"Yes" if sample.get("vlm_fallback") else "No"}</dd>
<dt>Target Objects</dt><dd>{esc(', '.join(r.get('description', r['phrase']) for r in sample.get('grounding_results', [])) or str(sample.get('scene_keywords', sample.get('focus_prompts', []))))}</dd>
<dt>Objects Found</dt><dd>{sample.get("objects_count", 0)} (after predetect: {sample.get("objects_count_after_predetect", "?")}, after NMS: {sample.get("objects_count_after_nms", "?")})</dd>
<dt>Cameras</dt><dd>{sample.get("num_cameras", 0)}</dd>
<dt>Retries</dt><dd>{sample.get("retry_count", 0)}</dd>
</dl>
<pre style="margin-top:12px">{esc(sample.get("query", ""))}</pre>
</div>""")

    # VLM discovery
    if sample.get("vlm_discovery_used"):
        parts.append(f"""<div class="card">
<strong>VLM Object Discovery:</strong> {esc(str(sample.get("vlm_discovered_objects", [])))}
</div>""")
    return parts


def section_input_images(ctx: ReportContext) -> List[str]:
    """2. Input Images."""
    item_id = ctx.item_id
    image_dir = ctx.image_dir
    parts: List[str] = []
    # ── 2. Input Images ──
    parts.append('<h2>2. Input Images</h2><div class="card">')
    image_paths = resolve_image_paths_for_sample(item_id, image_dir)
    if image_paths:
        parts.append('<div class="images-row">')
        for ip in image_paths:
            data_uri = b64_image(str(ip))
            parts.append(f'<img src="{data_uri}" alt="{ip.name}" title="{ip.name}">')
        parts.append("</div>")
    else:
        parts.append(
            f'<p style="color:var(--muted)">Images not found for {esc(item_id)} '
            f'(checked MindCube index and {esc(str(image_dir))}/{item_id}_*.jpg)</p>'
        )
    parts.append("</div>")
    return parts


def section_pose_constraints(ctx: ReportContext) -> List[str]:
    """3.5 Pose Constraints."""
    sample = ctx.sample
    parts: List[str] = []
    # ── 3.5 Pose Constraints (only present on --use_pose_constraints runs) ──
    # A list of dicts ({"type":"rotation", "from_cam":int, "to_cam":int,
    # "yaw":float, "axis":str}) or ({"type":"same_position", "cams":[int,...]}).
    if "pose_constraints_extracted" in sample:
        constraints = sample.get("pose_constraints_extracted") or []
        extract_time = sample.get("pose_constraints_extract_time_s")
        parts.append('<h2>3.5 Pose Constraints</h2><div class="card">')
        if extract_time is not None:
            time_str = f" &nbsp; <strong>Extract time:</strong> {extract_time}s"
        else:
            time_str = ""
        if not constraints:
            parts.append(
                f'<p><span class="badge" style="background:#6b7280">NO CONSTRAINTS</span>{time_str}</p>'
                '<p style="color:var(--muted);font-size:0.88rem">'
                "Either the question did not contain pose keywords, or the "
                "extractor returned an empty list. Scene poses are unchanged "
                "from VGGT.</p>"
            )
        else:
            parts.append(
                f'<p><span class="badge" style="background:#22c55e">APPLIED {len(constraints)}</span>{time_str}</p>'
                '<p style="color:var(--muted);font-size:0.88rem">'
                "Each constraint was applied to <code>scene.constraint.*</code> "
                "before grounding/codegen, mutating the corresponding "
                "<code>scene.cameras[i].extrinsics</code>.</p>"
                "<table><tr><th>#</th><th>Type</th><th>Cameras</th>"
                "<th>Params</th></tr>"
            )
            for i, rec in enumerate(constraints):
                ctype = esc(rec.get("type", "?"))
                if rec.get("type") == "rotation":
                    cams = f"cam {rec.get('from_cam', '?')} → cam {rec.get('to_cam', '?')}"
                    params = (
                        f"yaw={esc(rec.get('yaw', '?'))}, "
                        f"axis={esc(rec.get('axis', 'up'))}"
                    )
                elif rec.get("type") == "same_position":
                    cams = ", ".join(f"cam {c}" for c in rec.get("cams", []))
                    params = "—"
                else:
                    cams = "—"
                    params = esc(json.dumps(rec, default=_json_default))
                parts.append(
                    f"<tr><td>{i}</td><td><code>{ctype}</code></td>"
                    f"<td>{cams}</td><td>{params}</td></tr>"
                )
            parts.append("</table>")
        parts.append("</div>")
    return parts


def _planner_status_badge(planner_data: Dict, groundings: List[Dict]) -> str:
    """FAILED if the planner flagged a failure, HIT if it produced groundings, else MISS."""
    status = (planner_data.get("status") or "").lower()
    if status == "failed":
        return '<span class="badge" style="background:#ef4444">FAILED</span>'
    if groundings:
        return '<span class="badge" style="background:#22c55e">HIT</span>'
    return '<span class="badge" style="background:#f59e0b">MISS</span>'


def _planner_heading(ctx: ReportContext, groundings: List[Dict]) -> List[str]:
    """The section title (with the DETECT badge) and the status / time line."""
    needs_det = ctx.sample.get("needs_detection", True)
    det_badge = ' <span class="badge" style="background:#22c55e">DETECT</span>' if needs_det else ' <span class="badge" style="background:#6b7280">NO DETECT</span>'
    status_badge = _planner_status_badge(ctx.planner_data, groundings)
    return [
        f"<h2>4. Planner{det_badge}</h2>"
        '<div class="card">',
        f"<p><strong>Status:</strong> {status_badge} &nbsp; "
        f"<strong>Time:</strong> {ctx.planner_time}s</p>",
    ]


def _scene_keywords_row(scene_kws: List) -> str:
    """Table row of the descriptive keywords used for pre-detection, as pills."""
    kw_pills = " ".join(
        f'<span class="badge" style="background:#0ea5e9;margin:2px">{esc(str(k))}</span>'
        for k in scene_kws
    )
    return (
        f"<tr><th>scene_keywords<br/>"
        f'<span style="font-weight:400;color:var(--muted);font-size:0.8rem">'
        f"({len(scene_kws)} items)</span></th>"
        f"<td>{kw_pills}</td></tr>"
    )


def _grounding_kind_badge(g: Dict) -> str:
    if g.get("is_region"):
        return '<span class="badge" style="background:#a855f7">REGION</span>'
    return '<span class="badge" style="background:#6b7280">OBJECT</span>'


def _planner_groundings_row(groundings: List[Dict]) -> str:
    """Table row of the planner's first-pass object groundings.

    Phrases revised during planner-retry keep their original description here; the
    revised description is in the grounding_results row.
    """
    if not groundings:
        return (
            "<tr><th>object_groundings</th>"
            '<td><span style="color:var(--muted)">none</span></td></tr>'
        )
    rows = "".join(
        f"<tr><td style='width:200px'><code>{esc(g.get('phrase', ''))}</code></td>"
        f"<td>{esc(g.get('description', ''))}</td>"
        f"<td>cam {g.get('cam_id', '?')}</td>"
        f"<td>{_grounding_kind_badge(g)}</td></tr>"
        for g in groundings
    )
    inner = (
        '<table style="margin:0;width:100%;font-size:0.88rem">'
        "<thead><tr><th>phrase</th><th>description (first-pass)</th>"
        "<th>camera</th><th>kind</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
    return (
        f"<tr><th>object_groundings<br/>"
        f'<span style="font-weight:400;color:var(--muted);font-size:0.8rem">'
        f"({len(groundings)} items)</span></th>"
        f"<td>{inner}</td></tr>"
    )


def _score_html(s) -> str:
    """A grounding score as HTML: three decimals, or the escaped text of a non-number."""
    try:
        return f"{float(s):.3f}"
    except (TypeError, ValueError):
        return esc(str(s))


def _grounding_score_cell(s) -> str:
    """Score cell colored green (>= 0.7), amber (>= 0.3) or red; non-numbers shown as text."""
    try:
        sv = float(s)
    except (TypeError, ValueError):
        return f"<td>{_score_html(s)}</td>"
    if sv >= 0.7:
        bg = "#16a34a"
    elif sv >= 0.3:
        bg = "#f59e0b"
    else:
        bg = "#ef4444"
    return f'<td style="background:{bg};color:white;font-weight:600;text-align:right">{_score_html(sv)}</td>'


def _grounding_verified_cell(v) -> str:
    if v:
        return '<td style="color:#16a34a;font-weight:600">YES</td>'
    return '<td style="color:#ef4444;font-weight:600">NO</td>'


def _final_description_cell(gr: Dict, first_desc_by_phrase: Dict[str, str]) -> str:
    """The post-retry description, highlighted when it differs from the planner's first pass."""
    key = (gr.get("phrase") or "").strip().lower()
    first = first_desc_by_phrase.get(key, "")
    final = (gr.get("description") or "").strip()
    if not final:
        return '<td><span style="color:var(--muted)">—</span></td>'
    if final == first:
        return '<td><span style="color:var(--muted)">(unchanged)</span></td>'
    return (
        f'<td style="background:#fef3c7;font-size:0.85rem">'
        f'<span style="color:#92400e;font-weight:600">↻ revised:</span> '
        f'{esc(final[:200])}{"…" if len(final) > 200 else ""}</td>'
    )


def _grounding_diagnostic_cell(gr: Dict) -> str:
    diag = (gr.get("diagnostic") or "").strip()
    if not diag:
        return '<td><span style="color:var(--muted)">—</span></td>'
    if diag.lower() == "salvaged":
        return '<td><span class="badge" style="background:#f59e0b">SALVAGED</span></td>'
    return f'<td><span class="badge" style="background:#ef4444">{esc(diag[:60])}</span></td>'


def _verify_feedback_cell(gr: Dict) -> str:
    fb = (gr.get("verify_feedback") or "").strip()
    if not fb:
        return '<td><span style="color:var(--muted)">—</span></td>'
    return (
        f'<td style="background:#eef2ff;font-size:0.83rem;'
        f'max-width:380px">'
        f'<span style="color:#3730a3;font-weight:600">VLM:</span> '
        f'{esc(fb[:400])}{"…" if len(fb) > 400 else ""}</td>'
    )


def _grounding_results_row(grounding_results: List[Dict], groundings: List[Dict]) -> str:
    """Table row of the ObjectGrounder results, one inner row per phrase."""
    first_desc_by_phrase = {
        (g.get("phrase") or "").strip().lower(): (g.get("description") or "").strip()
        for g in groundings
    }
    gr_rows = "".join(
        f"<tr><td><code>{esc(gr.get('phrase', ''))}</code></td>"
        f"{_grounding_verified_cell(gr.get('verified'))}"
        f"{_grounding_score_cell(gr.get('score', 0))}"
        f"<td>{gr.get('attempts', 0)}</td>"
        f"<td>{gr.get('obj_indices', [])}</td>"
        f"{_final_description_cell(gr, first_desc_by_phrase)}"
        f"{_grounding_diagnostic_cell(gr)}"
        f"{_verify_feedback_cell(gr)}</tr>"
        for gr in grounding_results
    )
    gr_inner = (
        '<table style="margin:0;width:100%;font-size:0.88rem">'
        "<thead><tr><th>phrase</th><th>verified</th><th>score</th>"
        "<th>attempts</th><th>obj_ids</th>"
        "<th>description (final, post-retry)</th>"
        "<th>diagnostic</th>"
        "<th>verify_feedback</th></tr></thead>"
        f"<tbody>{gr_rows}</tbody></table>"
    )
    return (
        f"<tr><th>grounding_results<br/>"
        f'<span style="font-weight:400;color:var(--muted);font-size:0.8rem">'
        f"({len(grounding_results)} items)</span></th>"
        f"<td>{gr_inner}</td></tr>"
    )


def _planner_raw_and_thinking(ctx: ReportContext):
    """The planner's raw response and thinking trace, from the result or else the planner cache."""
    planner_data = ctx.planner_data
    planner_cache_entry = {}
    if ctx.planner_cache and ctx.query_text in ctx.planner_cache:
        planner_cache_entry = ctx.planner_cache[ctx.query_text]
    raw_response = planner_data.get("raw", "") or planner_cache_entry.get("raw", "") or planner_cache_entry.get("raw_response", "")
    thinking_trace = planner_data.get("thinking", "") or planner_cache_entry.get("thinking", "")
    return raw_response, thinking_trace


def _raw_planner_output_row(raw_response: str) -> str:
    return (
        f"<tr><th>raw_planner_output</th>"
        f'<td><pre style="margin:0;background:#1e1e2e;color:#cdd6f4;padding:12px;border-radius:6px;'
        f'white-space:pre-wrap;max-height:400px;overflow-y:auto">{esc(raw_response)}</pre></td></tr>'
    )


def _thinking_trace_details(thinking_trace: str) -> List[str]:
    return [
        "<details><summary>Planner Thinking Trace</summary>",
        f'<p><strong>Thinking trace ({len(thinking_trace)} chars):</strong></p>'
        f'<pre style="background:#1e1e2e;color:#cdd6f4;padding:12px;border-radius:6px;'
        f'white-space:pre-wrap;max-height:400px;overflow-y:auto">{esc(thinking_trace)}</pre>',
        "</details>",
    ]


def _planner_json_details(planner_data: Dict) -> List[str]:
    """The exact planner output as JSON, collapsed."""
    return [
        "<details><summary>Exact Planner Output (full JSON)</summary>",
        f'<pre style="background:#1e1e2e;color:#cdd6f4;padding:12px;border-radius:6px;'
        f'white-space:pre-wrap;max-height:500px;overflow-y:auto">{esc(json.dumps(planner_data, indent=2, default=_json_default))}</pre>',
        "</details>",
    ]


def section_planner(ctx: ReportContext) -> List[str]:
    """4. Planner."""
    planner_data = ctx.planner_data
    if not planner_data:
        return ['<h2>4. Planner</h2><div class="card"><p style="color:var(--muted)">'
                "No planner for this benchmark: the question's objects are grounded directly.</p></div>"]
    groundings = planner_data.get("object_groundings", []) or []
    parts = _planner_heading(ctx, groundings)
    parts.append("<table>")
    scene_kws = ctx.sample.get("scene_keywords") or []
    if scene_kws:
        parts.append(_scene_keywords_row(scene_kws))
    parts.append(_planner_groundings_row(groundings))
    if ctx.grounding_results:
        parts.append(_grounding_results_row(ctx.grounding_results, groundings))
    raw_response, thinking_trace = _planner_raw_and_thinking(ctx)
    if raw_response:
        parts.append(_raw_planner_output_row(raw_response))
    parts.append("</table>")
    if thinking_trace:
        parts.extend(_thinking_trace_details(thinking_trace))
    parts.extend(_planner_json_details(planner_data))
    parts.append("</div>")
    return parts


def section_disambiguation(ctx: ReportContext) -> List[str]:
    """4.5 Detection Disambiguation."""
    sample = ctx.sample
    parts: List[str] = []
    # ── 4.5 Detection Disambiguation ──
    disambiguation_logs = sample.get("disambiguation", []) or []
    if disambiguation_logs:
        parts.append('<h2>4.5 Detection Disambiguation</h2><div class="card">')
        parts.append(
            f'<p style="color:var(--muted);font-size:0.85rem">'
            f'{len(disambiguation_logs)} disambiguation attempt(s) recorded.</p>'
        )
        for i, entry in enumerate(disambiguation_logs, start=1):
            phrase = str(entry.get("phrase", "?"))
            stage = str(entry.get("stage", "?") or "?")
            status = str(entry.get("status", "?") or "?")
            selected_obj_id = entry.get("selected_obj_id")
            selected_yes_prob = entry.get("selected_yes_prob")
            summary = (
                f"Attempt {i}: {phrase} [{stage}]"
                f" -> kept obj {selected_obj_id}"
                f" (P(yes)={selected_yes_prob:.3f})"
                if isinstance(selected_yes_prob, (int, float)) and selected_obj_id is not None
                else f"Attempt {i}: {phrase} [{stage}] ({status})"
            )
            parts.append(
                f'<details{" open" if len(disambiguation_logs) == 1 else ""}>'
                f'<summary>{esc(summary)}</summary>'
            )
            parts.append("<table>")
            parts.append(
                f"<tr><th style=\"width:180px\">description</th>"
                f"<td>{esc(str(entry.get('description', '')))}</td></tr>"
            )
            parts.append(
                f"<tr><th>status</th><td><code>{esc(status)}</code></td></tr>"
            )
            if entry.get("prompt"):
                parts.append(
                    f"<tr><th>prompt</th>"
                    f'<td><pre style="margin:0;background:transparent;white-space:pre-wrap">{esc(str(entry.get("prompt", "")))}</pre></td></tr>'
                )
            candidates = entry.get("candidates", []) or []
            if candidates:
                candidate_rows = []
                for cand in candidates:
                    best_view = cand.get("best_view")
                    view_label = (
                        f"Image {int(best_view) + 1}"
                        if isinstance(best_view, int)
                        else "—"
                    )
                    selected = "yes" if cand.get("obj_id") == entry.get("selected_obj_id_before_reindex") else ""
                    candidate_rows.append(
                        f"<tr>"
                        f"<td><code>{esc(str(cand.get('obj_id', '?')))}</code></td>"
                        f"<td>{esc(view_label)}</td>"
                        f"<td>{cand.get('best_view_score', 0.0):.3f}</td>"
                        f"<td>{cand.get('yes_prob', 0.0):.3f}</td>"
                        f"<td>{esc(str(cand.get('status', '?')))}</td>"
                        f"<td>{selected}</td>"
                        f"</tr>"
                    )
                candidate_table = (
                    '<table style="margin:0;width:100%;font-size:0.88rem">'
                    '<thead><tr><th>obj_id</th><th>best_view</th><th>det_score</th><th>yes_prob</th><th>status</th><th>selected</th></tr></thead>'
                    f"<tbody>{''.join(candidate_rows)}</tbody></table>"
                )
                parts.append(f"<tr><th>candidates</th><td>{candidate_table}</td></tr>")
            removed_ids = entry.get("removed_obj_ids", []) or []
            parts.append(
                f"<tr><th>removed_obj_ids</th><td>{esc(str(removed_ids))}</td></tr>"
            )
            parts.append("</table>")
            parts.append("</details>")
        parts.append("</div>")
    return parts


def section_assembled_prompt(ctx: ReportContext) -> List[str]:
    """5. Assembled Prompt."""
    prompt_template = ctx.prompt_template
    query_text = ctx.query_text
    task_formalization = ctx.task_formalization
    clarified_query_block = ctx.clarified_query_block
    scene_facts_block = ctx.scene_facts_block
    parts: List[str] = []
    # ── 5. Assembled Prompt ──
    parts.append('<h2>5. Assembled Prompt</h2><div class="card">')
    if prompt_template:
        assembled = prompt_template.replace("{query}", query_text)
        assembled = assembled.replace("{object_groundings_block}", ctx.object_groundings_block)
        assembled = assembled.replace("{task_formalization}", task_formalization)
        assembled = assembled.replace("{scene_facts_block}", scene_facts_block)
        assembled = assembled.replace("{clarified_query_block}", clarified_query_block)
        parts.append(
            f'<p style="color:var(--muted);font-size:0.85rem">Total length: {len(assembled):,} chars</p>'
        )

        # Show the planner injected blocks
        object_groundings_block = ctx.object_groundings_block
        if object_groundings_block or clarified_query_block:
            parts.append(
                "<details open><summary>Planner Injected Blocks</summary>"
            )
            if object_groundings_block:
                parts.append(
                    f"<p><strong>object_groundings_block:</strong></p><pre>{esc(object_groundings_block)}</pre>"
                )
            if clarified_query_block:
                parts.append(
                    f"<p><strong>clarified_query_block:</strong></p><pre>{esc(clarified_query_block)}</pre>"
                )
            parts.append("</details>")

        # Show the full prompt (collapsed)
        parts.append("<details><summary>Full Prompt (click to expand)</summary>")
        parts.append(f"<pre>{esc(assembled)}</pre>")
        parts.append("</details>")
    else:
        parts.append('<p style="color:var(--muted)">Prompt file not provided.</p>')
    parts.append("</div>")
    return parts


# id(program_cache) -> (cache, {finalized program: key}); built once per cache.
_PROGRAM_TEXT_INDEX: Dict[int, Any] = {}


def _find_program_cache_key(ctx: ReportContext):
    """Return (key, label) of the cache entry codegen used for this sample.

    Programs are keyed by ``prog-v2:<sha256>`` of the fully rendered request,
    which the report cannot rebuild, so it tries: the key recorded in the
    result, then the entry whose finalized program equals the program the
    sample actually ran. Returns (None, <what was tried>) on a miss.
    """
    cache = ctx.program_cache
    recorded = ctx.sample.get("program_cache_key")
    if recorded and recorded in cache:
        return recorded, "recorded in result"
    ran = ctx.sample.get("raw_llm_code")
    if ran:
        entry = _PROGRAM_TEXT_INDEX.get(id(cache))
        if entry is None or entry[0] is not cache:
            index = {}
            for key, code in cache.items():
                if code:
                    index.setdefault(CodeGenerator._finalize(code), key)
            entry = _PROGRAM_TEXT_INDEX[id(cache)] = (cache, index)
        key = entry[1].get(ran)
        if key is not None:
            return key, "matched by program text"
    return None, "recorded key + program text"


def section_program_cache(ctx: ReportContext) -> List[str]:
    """6. Program Cache Lookup."""
    program_cache = ctx.program_cache
    parts: List[str] = []
    # ── 6. Program Cache ──
    parts.append('<h2>6. Program Cache Lookup</h2><div class="card">')
    if program_cache:
        hit_key, key_label = _find_program_cache_key(ctx)
        if hit_key is not None:
            parts.append(
                f'<p><span class="badge badge-ok">CACHE HIT</span> '
                f"(key: {esc(key_label)}, hash: <code>{esc(hit_key[-11:])}</code>)</p>"
            )
            parts.append(
                f"<details><summary>Cached code</summary><pre>{esc(program_cache[hit_key])}</pre></details>"
            )
        else:
            parts.append(
                f'<p><span class="badge badge-miss">CACHE MISS</span> '
                f"(tried: {esc(key_label)})</p>"
            )
        parts.append(
            f'<p style="color:var(--muted);font-size:0.82rem">{len(program_cache)} entries in cache</p>'
        )
    else:
        parts.append('<p style="color:var(--muted)">Program cache not provided.</p>')
    parts.append("</div>")
    return parts


def section_generated_code(ctx: ReportContext) -> List[str]:
    """7. Generated Code."""
    sample = ctx.sample
    parts: List[str] = []
    # ── 7. Generated Code ──
    full_code = sample.get("program_code", "")
    parts.append('<h2>7. Generated Code</h2><div class="card">')
    if full_code:
        user_code = _extract_user_code(full_code)

        # Count scene.ground() and scene.detect() calls (in user code only)
        n_ground = len(re.findall(r"\bscene\.ground\s*\(", user_code))
        n_detect = len(re.findall(r"\bscene\.detect\s*\(", user_code))
        n_score = len(re.findall(r"\.score\s*\(", user_code))
        parts.append(
            f'<p style="color:var(--muted);font-size:0.88rem">'
            f"API calls: "
            f'<span class="scene-detect">scene.detect</span>={n_detect} &nbsp; '
            f'<span class="scene-ground">scene.ground</span>={n_ground} &nbsp; '
            f".score={n_score}</p>"
        )

        # Escape first, then wrap API calls in highlight spans.
        code_html = esc(user_code)
        code_html = re.sub(
            r"(scene\.ground)(?=\s*\()",
            r'<span class="scene-ground">\1</span>',
            code_html,
        )
        code_html = re.sub(
            r"(scene\.detect)(?=\s*\()",
            r'<span class="scene-detect">\1</span>',
            code_html,
        )
        parts.append(f'<pre class="code-python">{code_html}</pre>')
        parts.append("<details><summary>Full wrapped code (with template)</summary>")
        parts.append(f"<pre>{esc(full_code)}</pre></details>")
    else:
        err = sample.get("error", "")
        parts.append(f'<p style="color:var(--red)">No code generated. {esc(err)}</p>')
    parts.append("</div>")
    return parts


def section_retries(ctx: ReportContext) -> List[str]:
    """7. Retry code."""
    sample = ctx.sample
    parts: List[str] = []
    # Retries
    retry_count = sample.get("retry_count", 0)
    if retry_count > 0:
        parts.append('<h2>7. Retry Code</h2><div class="card">')
        parts.append(f"<p>{retry_count} retries</p>")
        for r in range(1, retry_count + 1):
            rcode = sample.get(f"retry_{r}_code", "")
            if rcode:
                parts.append(
                    f"<details><summary>Retry {r}</summary><pre>{esc(rcode)}</pre></details>"
                )
        parts.append("</div>")
    return parts


def section_execution(ctx: ReportContext) -> List[str]:
    """8. Execution & Answer."""
    sample = ctx.sample
    correct = ctx.correct
    parts: List[str] = []
    # ── 8. Execution & Answer ──
    parts.append('<h2>8. Execution & Answer</h2><div class="card">')
    parts.append(f"""<dl class="kv">
<dt>Execution Successful</dt><dd>{"Yes" if sample.get("execution_successful") else "No"}</dd>
<dt>Code Answer</dt><dd><strong>{esc(str(sample.get("code_generated_answer", "?")))}</strong></dd>
<dt>Final Answer</dt><dd><strong>{esc(str(sample.get("final_answer_text", "?")))}</strong></dd>
<dt>Ground Truth</dt><dd><strong>{esc(sample.get("ground_truth_answer", "?"))}</strong></dd>
<dt>Result</dt><dd>{"<span class='badge badge-ok'>CORRECT</span>" if correct else "<span class='badge badge-wrong'>WRONG</span>" if correct is False else "<span class='badge badge-err'>ERROR</span>"}</dd>
</dl>""")
    if sample.get("error"):
        parts.append(
            f'<details open><summary>Error</summary><pre style="color:var(--red)">{esc(sample["error"])}</pre></details>'
        )
    if sample.get("vlm_fallback"):
        parts.append(
            '<p style="color:var(--amber);margin-top:8px">VLM fallback was used (code execution failed).</p>'
        )
    parts.append("</div>")
    return parts


def section_score_cache(ctx: ReportContext) -> List[str]:
    """9. Score Cache."""
    sample = ctx.sample
    item_id = ctx.item_id
    scene_data = ctx.scene_data
    parts: List[str] = []
    # ── 9. Score Cache Visualization ──
    score_cache = sample.get("score_cache")
    parts.append('<h2>9. Score Cache</h2><div class="card">')
    if score_cache and isinstance(score_cache, list):
        # Collect object labels from scene data for axis labels
        obj_labels = []
        if scene_data:
            for i, obj_d in enumerate(scene_data.get("objects", [])):
                obj_labels.append(f"[{i}] {obj_d.get('label', '?')}")

        score_calls = []
        for entry in score_cache:
            action = entry.get("action", "")
            if action == "__init__" and "inputs" in entry:
                # This is a score() call — extract the question and the tensor
                args = entry["inputs"].get("args", [])
                question_str = args[0] if args else "?"
                result = entry.get("result")
                if result is not None:
                    score_calls.append({"question": question_str, "scores": result})

        if score_calls:
            parts.append(f"<p>{len(score_calls)} score() calls captured</p>")
            for sc_idx, sc in enumerate(score_calls):
                q = sc["question"]
                scores = sc["scores"]
                parts.append(
                    f"<details{'  open' if sc_idx == 0 else ''}>"
                    f"<summary>score(): {esc(q[:120])}</summary>"
                )

                # Handle 1D (single-object) scores
                if (
                    isinstance(scores, list)
                    and scores
                    and not isinstance(scores[0], list)
                ):
                    # 1D tensor — bar chart
                    n_scores = len(scores)
                    labels = (
                        obj_labels[:n_scores]
                        if len(obj_labels) >= n_scores
                        else [f"[{i}]" for i in range(n_scores)]
                    )
                    # Find argmax
                    max_idx = int(np.argmax(scores))
                    colors = [
                        "#2563eb" if i != max_idx else "#dc2626"
                        for i in range(n_scores)
                    ]
                    div_id = f"score_{item_id}_{sc_idx}"
                    parts.append(
                        f'<div id="{div_id}" style="width:100%;height:{max(200, n_scores * 28 + 60)}px"></div>'
                    )
                    parts.append(script_tag(template_block("report.js", "score_bar").substitute(
                        labels_json=json.dumps(labels, default=_json_default),
                        x_json=json.dumps([round(s, 4) for s in scores], default=_json_default),
                        colors_json=json.dumps(colors, default=_json_default),
                        text_json=json.dumps([f"{s:.4f}" for s in scores], default=_json_default),
                        xmax=f"{max(max(scores) * 1.15, 0.01):.4f}",
                        div_id=div_id,
                    )))
                    # Text summary
                    parts.append(
                        f'<p style="font-size:0.85rem;color:var(--muted)">'
                        f'Argmax: <strong style="color:var(--red)">{labels[max_idx]}</strong> '
                        f"(score={scores[max_idx]:.4f})</p>"
                    )
                elif (
                    isinstance(scores, list) and scores and isinstance(scores[0], list)
                ):
                    # 2D tensor (pairwise) — show as heatmap
                    n = len(scores)
                    labels_n = (
                        obj_labels[:n]
                        if len(obj_labels) >= n
                        else [f"[{i}]" for i in range(n)]
                    )
                    div_id = f"score_{item_id}_{sc_idx}"
                    parts.append(
                        f'<div id="{div_id}" style="width:100%;height:{max(300, n * 40 + 100)}px"></div>'
                    )
                    parts.append(script_tag(template_block("report.js", "heatmap").substitute(
                        z_json=json.dumps(scores, default=_json_default),
                        labels_json=json.dumps(labels_n, default=_json_default),
                        div_id=div_id,
                    )))
                else:
                    parts.append(
                        f"<pre>{esc(json.dumps(scores, indent=2, default=_json_default)[:500])}</pre>"
                    )
                parts.append("</details>")
        else:
            parts.append(
                '<p style="color:var(--muted)">Score cache present but no score() calls found.</p>'
            )

        # Show full cache (collapsed)
        parts.append(
            "<details><summary>Raw score cache (all operations)</summary>"
            f"<pre>{esc(json.dumps(score_cache, indent=2, default=_json_default)[:10000])}</pre></details>"
        )
    else:
        parts.append(
            '<p style="color:var(--muted)">No score cache (run needs serialization update).</p>'
        )
    parts.append("</div>")
    return parts


def _object_grounding_map(grounding_results: List[Dict]) -> Dict[int, Dict]:
    """Object index -> the grounding result that selected it (the last one, if several)."""
    grounding_map = {}
    for gr in grounding_results:
        for idx in gr.get("obj_indices", []):
            grounding_map[idx] = gr
    return grounding_map


def _per_view_scores_str(obj_d: Dict) -> str:
    pv_scores = obj_d.get("per_view_scores", {})
    return ", ".join(
        f"v{k}:{v:.2f}"
        for k, v in sorted(pv_scores.items(), key=lambda x: int(x[0]))
    )


def _per_view_bboxes_str(obj_d: Dict) -> str:
    pv_bboxes = obj_d.get("per_view_bboxes", {})
    bbox_strs = []
    for k, v in sorted(pv_bboxes.items(), key=lambda x: int(x[0])):
        try:
            bbox_strs.append(f"v{k}:[{v[0]:.0f},{v[1]:.0f},{v[2]:.0f},{v[3]:.0f}]"
                             if isinstance(v, (list, tuple)) and len(v) >= 4 else f"v{k}:{v}")
        except (TypeError, ValueError):  # a coordinate that is not a number
            bbox_strs.append(f"v{k}:{v}")
    return "<br/>".join(bbox_strs) if bbox_strs else "—"


def _object_grounding_cells(gr: Optional[Dict]):
    """(grounded phrase, verified cell HTML) for an object's grounding result, if any."""
    if not gr:
        return "—", '<span style="color:var(--muted)">—</span>'
    g_phrase = gr.get("phrase", "—")
    g_verified = gr.get("verified", False)
    verified_badge = '<span class="badge" style="background:#16a34a">YES</span>' if g_verified else '<span class="badge" style="background:#ef4444">NO</span>'
    g_score = gr.get("score", 0)
    return g_phrase, f'{verified_badge}<br/>score={_score_html(g_score)}'


def _detected_object_row(i: int, obj_d: Dict, gr: Optional[Dict]) -> str:
    meta = obj_d.get("metadata") or {}
    src = meta.get("source_keyword", "?")
    score_str = _per_view_scores_str(obj_d)
    bbox_str = _per_view_bboxes_str(obj_d)
    g_phrase, verified_cell = _object_grounding_cells(gr)
    return (
        f"<tr><td>{i}</td><td><strong>{esc(obj_d.get('label', ''))}</strong></td>"
        f"<td>{esc(src)}</td>"
        f"<td>{obj_d.get('views', [])}</td>"
        f"<td>{score_str}</td>"
        f'<td style="font-size:0.82rem">{esc(bbox_str)}</td>'
        f"<td>{esc(g_phrase)}</td>"
        f"<td>{verified_cell}</td></tr>"
    )


def _detected_objects_table(objects: List[Dict], grounding_results: List[Dict]) -> List[str]:
    """Header, one row per object (detection details + grounding verdict), closing tag."""
    grounding_map = _object_grounding_map(grounding_results)
    parts = [
        "<table><tr><th>Idx</th><th>Label</th><th>Source</th>"
        "<th>Views</th><th>Per-View Scores</th><th>Per-View Bboxes</th>"
        "<th>Grounded Phrase</th><th>Verified</th></tr>"
    ]
    for i, obj_d in enumerate(objects):
        parts.append(_detected_object_row(i, obj_d, grounding_map.get(i)))
    parts.append("</table>")
    return parts


# RGB color of object i in the overlay images is _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)].
_OVERLAY_COLORS = [
    (220, 38, 38),
    (37, 99, 235),
    (22, 163, 74),
    (234, 179, 8),
    (168, 85, 247),
    (249, 115, 22),
    (14, 165, 233),
    (236, 72, 153),
    (132, 204, 22),
    (99, 102, 241),
]


def _pil_available() -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    except ImportError:
        return False
    return True


def _decode_mask(mask_data) -> Optional[np.ndarray]:
    """A per-view mask as a uint8 array, from a {"data": base64 zlib bytes, "shape"} dict or a
    nested list / array; None for any other form."""
    if isinstance(mask_data, dict) and "data" in mask_data:
        compressed = base64.b64decode(mask_data["data"])
        flat = zlib.decompress(compressed)
        return np.frombuffer(flat, dtype=np.uint8).reshape(mask_data["shape"])
    if isinstance(mask_data, (list, np.ndarray)):
        return np.array(mask_data, dtype=np.uint8)
    return None


def _draw_mask_overlays(img, objects: List[Dict], view_idx: int):
    """Composite each object's mask for this view onto the RGBA image as a translucent color."""
    from PIL import Image

    for obj_i, obj_d in enumerate(objects):
        pv_masks = obj_d.get("per_view_masks", {})
        mask_data = pv_masks.get(str(view_idx))
        if mask_data is None:
            continue
        color = _OVERLAY_COLORS[obj_i % len(_OVERLAY_COLORS)]
        mask_resized = _decode_mask(mask_data)
        if mask_resized is None:
            continue
        if mask_resized.shape[:2] != (img.height, img.width):
            mask_pil = Image.fromarray(mask_resized * 255)
            mask_pil = mask_pil.resize((img.width, img.height), Image.NEAREST)
            mask_resized = np.array(mask_pil) > 127
        mask_bool = mask_resized.astype(bool)
        overlay_arr = np.zeros((img.height, img.width, 4), dtype=np.uint8)
        overlay_arr[mask_bool, 0] = color[0]
        overlay_arr[mask_bool, 1] = color[1]
        overlay_arr[mask_bool, 2] = color[2]
        overlay_arr[mask_bool, 3] = 80  # semi-transparent
        overlay = Image.fromarray(overlay_arr, "RGBA")
        img = Image.alpha_composite(img, overlay)
    return img


def _label_font():
    from PIL import ImageFont

    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        return ImageFont.load_default()


def _box_xyxy(bbox) -> Optional[tuple]:
    """(x1, y1, x2, y2) as floats, or None unless the box is exactly four numbers with x1 <= x2
    and y1 <= y2 (the overlay skips such a box; the table still lists it)."""
    if not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2) if x1 <= x2 and y1 <= y2 else None


def _draw_bboxes(img_rgb, objects: List[Dict], view_idx: int) -> None:
    """Draw each object's box for this view on the RGB image, labelled "[i] label (score)"."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img_rgb)
    for obj_i, obj_d in enumerate(objects):
        pv_bboxes = obj_d.get("per_view_bboxes", {})
        box = _box_xyxy(pv_bboxes.get(str(view_idx)))
        if box is None:
            continue
        color = _OVERLAY_COLORS[obj_i % len(_OVERLAY_COLORS)]
        x1, y1, x2, y2 = box
        for offset in range(3):  # 3 px thick border
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )
        label = f"[{obj_i}] {obj_d.get('label', '?')}"
        pv_scores = obj_d.get("per_view_scores", {})
        sc = pv_scores.get(str(view_idx))
        if sc is not None:
            label += f" ({sc:.2f})"
        font = _label_font()
        tw = draw.textlength(label, font=font)
        th = 18
        ly = max(0, y1 - th - 2)
        draw.rectangle([x1, ly, x1 + tw + 4, ly + th], fill=color)
        draw.text(
            (x1 + 2, ly + 1), label, fill=(255, 255, 255), font=font
        )


def _jpeg_img_tag(img_rgb, view_idx: int) -> str:
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<img src="data:image/jpeg;base64,{b64}" '
        f'alt="View {view_idx}" title="View {view_idx}" '
        f'style="max-height:400px">'
    )


def _overlay_images(item_id: str, image_dir: str, objects: List[Dict]) -> List[str]:
    """Every input view with the objects' masks and boxes drawn on it.

    Empty when the images are not found or Pillow is not installed.
    """
    image_paths = resolve_image_paths_for_sample(item_id, image_dir)
    if not image_paths or not _pil_available():
        return []
    from PIL import Image

    parts = ["<h3>Per-View Bbox + Mask Overlays</h3>", '<div class="images-row">']
    for view_idx, ip in enumerate(image_paths):
        img = Image.open(str(ip)).convert("RGBA")
        img = _draw_mask_overlays(img, objects, view_idx)  # masks first, under the boxes
        img_rgb = img.convert("RGB")
        _draw_bboxes(img_rgb, objects, view_idx)
        parts.append(_jpeg_img_tag(img_rgb, view_idx))
    parts.append("</div>")
    return parts


def section_detected_objects(ctx: ReportContext) -> List[str]:
    """9b. Detected Objects (bbox/mask overlays)."""
    scene_data = ctx.scene_data
    if not (scene_data and scene_data.get("objects")):
        return []
    objects = scene_data["objects"]
    parts = ['<h2>9b. Detected Objects</h2><div class="card">']
    parts.append(f"<p>{len(objects)} object(s) detected</p>")
    parts.extend(_detected_objects_table(objects, ctx.grounding_results))
    parts.extend(_overlay_images(ctx.item_id, ctx.image_dir, objects))
    parts.append("</div>")
    return parts


def section_planned_vs_detected(ctx: ReportContext) -> List[str]:
    """9c. Planned vs Detected Objects."""
    planner_data = ctx.planner_data
    grounding_results = ctx.grounding_results
    parts: List[str] = []
    # ── 9c. Planned vs Detected Objects Comparison ──
    if planner_data or grounding_results:
        parts.append('<h2>9c. Planned vs Detected Objects</h2><div class="card">')
        parts.append('<p style="color:var(--muted);font-size:0.85rem">Comparison of planner groundings vs actual detection results</p>')
        
        # Build comparison table
        comp_rows = []
        all_phrases = set()
        
        # Collect planned phrases
        planned_groundings = planner_data.get("object_groundings", []) if planner_data else []
        for g in planned_groundings:
            phrase = g.get("phrase", "?")
            all_phrases.add(phrase)
        
        # Collect grounding result phrases
        for gr in grounding_results:
            phrase = gr.get("phrase", "?")
            all_phrases.add(phrase)
        
        for phrase in sorted(all_phrases):
            # Find planned grounding
            planned = None
            for g in planned_groundings:
                if g.get("phrase") == phrase:
                    planned = g
                    break
            
            # Find actual grounding result
            actual = None
            for gr in grounding_results:
                if gr.get("phrase") == phrase:
                    actual = gr
                    break
            
            planned_desc = planned.get("description", "—") if planned else "—"
            planned_cam = planned.get("cam_id", "—") if planned else "—"
            planned_kind = "REGION" if (planned and planned.get("is_region")) else "OBJECT"
            
            if actual:
                actual_verified = actual.get("verified", False)
                verified_badge = '<span class="badge" style="background:#16a34a">YES</span>' if actual_verified else '<span class="badge" style="background:#ef4444">NO</span>'
                actual_obj_ids = actual.get("obj_indices", [])
                actual_score = actual.get("score", 0)
                actual_attempts = actual.get("attempts", 0)
                actual_status = f"{verified_badge} objs={actual_obj_ids} score={_score_html(actual_score)} attempts={actual_attempts}"
            else:
                actual_status = '<span style="color:var(--muted)">No detection attempt</span>'
            
            comp_rows.append(
                f"<tr><td><code>{esc(phrase)}</code></td>"
                f"<td>{esc(planned_desc)}</td>"
                f"<td>cam {planned_cam}</td>"
                f'<td><span class="badge" style="background:{"#a855f7" if planned_kind == "REGION" else "#6b7280"}">{planned_kind}</span></td>'
                f"<td>{actual_status}</td></tr>"
            )
        
        if comp_rows:
            parts.append(
                '<table style="width:100%;font-size:0.88rem">'
                "<thead><tr><th>Phrase</th><th>Planned Description</th><th>Cam</th><th>Kind</th><th>Detection Result</th></tr></thead>"
                f"<tbody>{''.join(comp_rows)}</tbody></table>"
            )
        else:
            parts.append('<p style="color:var(--muted)">No planned groundings available</p>')
        
        parts.append("</div>")
    return parts


def _scene_source_notes(ctx: ReportContext) -> List[str]:
    """The scene file shown, flagged when it was auto-detected from a sibling run."""
    scene_path = ctx.scene_path
    parts = [
        f'<p style="color:var(--muted);font-size:0.85rem">Scene: {esc(scene_path or "?")}</p>'
    ]
    own_scenes = Path(ctx.result_json).parent / "scenes"
    if not ctx.scene_dir and scene_path and Path(scene_path).parent != own_scenes:
        parts.append(
            '<p><span class="badge badge-wrong">SCENE FROM ANOTHER RUN</span> '
            "this run has no scene JSON for the sample; sections 9b, 10 and the "
            "scene facts come from the sibling run above.</p>"
        )
    return parts


def _fmt_vec(v, ndp=3):
    if v is None:
        return "—"
    try:
        return "[" + ", ".join(f"{float(x):+.{ndp}f}" for x in v) + "]"
    except (TypeError, ValueError):
        return esc(str(v))


def _fmt_rotmat(R):
    """A 3x3 rotation as a small dark HTML table; anything else as escaped text."""
    if R is None:
        return "—"
    try:
        arr = np.asarray(R, dtype=float)
        if arr.shape != (3, 3):
            return esc(str(R))
        rows_html = "".join(
            "<tr>" + "".join(
                f'<td style="padding:2px 8px;font-family:monospace;text-align:right">{arr[i,j]:+.3f}</td>'
                for j in range(3)
            ) + "</tr>"
            for i in range(3)
        )
        return (
            '<table style="margin:0;border-collapse:collapse;'
            'background:#1e1e2e;color:#cdd6f4;border-radius:4px">'
            f"{rows_html}</table>"
        )
    except Exception:
        return esc(str(R))


def _azimuth_cell(obj_d: Dict) -> str:
    """World azimuth (first Euler angle) wrapped to [-180, 180), with the raw value."""
    euler = obj_d.get("euler_world_deg") or [None, None, None]
    try:
        az_w = float(euler[0])
        az_w_disp = ((az_w + 180.0) % 360.0) - 180.0
        return f"{az_w_disp:+.1f}&nbsp;(raw&nbsp;{az_w:.1f})"
    except (TypeError, ValueError):
        return "—"


def _fmt_degrees(x) -> str:
    return f"{float(x):+.0f}" if isinstance(x, (int, float)) else "?"


def _raw_orientation_entry(r: Dict) -> str:
    """One raw Orient-Anything detection: view, (azimuth, polar, roll) and confidence."""
    az = r.get("azimuth")
    pol = r.get("polar")
    roll = r.get("roll")
    conf = r.get("confidence")
    v = r.get("view")
    conf_str = (
        f"&nbsp;c={float(conf):.2f}"
        if isinstance(conf, (int, float))
        else ""
    )
    return (
        f"<span style='font-family:monospace'>v{v}:&nbsp;"
        f"({_fmt_degrees(az)},&nbsp;{_fmt_degrees(pol)},&nbsp;{_fmt_degrees(roll)}){conf_str}</span>"
    )


def _raw_orientation_cell(obj_d: Dict, raw_az_by_label: Dict[str, List[Dict]]) -> str:
    """The raw per-view Orient-Anything outputs recorded for the object's keyword (or label)."""
    src_kw = (obj_d.get("metadata") or {}).get("source_keyword") or obj_d.get("label", "")
    raw_list = raw_az_by_label.get(src_kw, []) or raw_az_by_label.get(obj_d.get("label", ""), [])
    if not raw_list:
        return "<span style='color:var(--muted)'>—</span>"
    return "<br>".join([_raw_orientation_entry(r) for r in raw_list])


def _rotation_cell(rot) -> str:
    if rot is None:
        return "—"
    return (
        f"<details><summary style='cursor:pointer;color:var(--muted)'>show</summary>"
        f"{_fmt_rotmat(rot)}</details>"
    )


def _scene_object_row(i: int, obj_d: Dict, raw_az_by_label: Dict[str, List[Dict]]) -> str:
    c = [round(x, 3) for x in obj_d["center_world"]]
    front = obj_d.get("front_world")
    up = obj_d.get("up_world")
    right = obj_d.get("right_world")
    dims = obj_d.get("dims")
    rot = obj_d.get("rotation_world")
    az_cell = _azimuth_cell(obj_d)
    raw_cell = _raw_orientation_cell(obj_d, raw_az_by_label)
    rot_cell = _rotation_cell(rot)
    return (
        f"<tr>"
        f"<td>{i}</td>"
        f"<td>{esc(obj_d.get('label', ''))}</td>"
        f"<td><code>{_fmt_vec(c, 3)}</code></td>"
        f"<td><code>{_fmt_vec(front, 3)}</code></td>"
        f"<td><code>{_fmt_vec(up, 3)}</code></td>"
        f"<td><code>{_fmt_vec(right, 3)}</code></td>"
        f"<td><code>{_fmt_vec(dims, 3)}</code></td>"
        f"<td><code>{az_cell}</code></td>"
        f"<td>{raw_cell}</td>"
        f"<td>{obj_d.get('views', [])}</td>"
        f"<td>{rot_cell}</td>"
        f"</tr>"
    )


def _scene_object_table(scene_data: Dict, raw_az_by_label: Dict[str, List[Dict]]) -> List[str]:
    parts = [
        "<table>"
        "<tr>"
        "<th>Idx</th><th>Label</th>"
        "<th>Center (world)</th>"
        "<th>Front</th><th>Up</th><th>Right</th>"
        "<th>Dims (w,h,d)</th>"
        "<th>Azimuth&nbsp;(deg)</th>"
        "<th>OA raw (az,&nbsp;polar,&nbsp;roll)</th>"
        "<th>Views</th>"
        "<th>Rotation</th>"
        "</tr>"
    ]
    for i, obj_d in enumerate(scene_data.get("objects", [])):
        parts.append(_scene_object_row(i, obj_d, raw_az_by_label))
    parts.append("</table>")
    return parts


def _scene_camera_table(scene_data: Dict) -> List[str]:
    """Camera centers (inverted extrinsics) and image sizes."""
    parts = ["<table><tr><th>Idx</th><th>Position (world)</th><th>Image Size</th></tr>"]
    for i, cam_d in enumerate(scene_data.get("cameras", [])):
        E = np.array(cam_d["extrinsics"])
        E4 = np.eye(4)
        E4[:3, :4] = E if E.shape == (3, 4) else E[:3, :4]
        pos = np.linalg.inv(E4)[:3, 3]
        pos = [round(float(x), 2) for x in pos]
        parts.append(
            f"<tr><td>{i}</td><td>{pos}</td><td>{cam_d.get('image_size', '?')}</td></tr>"
        )
    parts.append("</table>")
    return parts


def _scene_plot(ctx: ReportContext) -> List[str]:
    """The interactive Plotly 3D view: objects, cameras and the room center."""
    room_center = ctx.scene_obj.room_center().tolist() if ctx.scene_obj else None
    traces_json = build_plotly_traces_json(ctx.scene_data, room_center)
    div_id = f"scene3d_{ctx.item_id}"
    return [
        f'<div id="{div_id}" class="plotly-container"></div>',
        script_tag(template_block("report.js", "scene3d").substitute(
            traces_json=traces_json, div_id=div_id,
        )),
    ]


def section_scene_3d(ctx: ReportContext) -> List[str]:
    """10. 3D Scene."""
    parts = ['<h2>10. 3D Scene</h2><div class="card">']
    if ctx.scene_data:
        parts.extend(_scene_source_notes(ctx))
        parts.extend(_scene_object_table(ctx.scene_data, ctx.raw_az_by_label))
        parts.extend(_scene_camera_table(ctx.scene_data))
        parts.extend(_scene_plot(ctx))
    else:
        parts.append(
            f'<p style="color:var(--muted)">Scene JSON not found for id={esc(ctx.item_id)}</p>'
        )
    parts.append("</div>")
    return parts


def section_timing(ctx: ReportContext) -> List[str]:
    """11. Timing Breakdown."""
    sample = ctx.sample
    parts: List[str] = []
    # ── 11. Timing ──
    parts.append('<h2>11. Timing Breakdown</h2><div class="card">')
    timings = [
        ("Planner", sample.get("planner_time_s")),
        ("Scene build", sample.get("scene_build_time_s")),
        ("Execution", sample.get("exec_time_s")),
    ]
    total_t = sum(t for _, t in timings if t is not None)
    max_t = max((t for _, t in timings if t is not None), default=1)
    parts.append("<table><tr><th>Stage</th><th>Time</th><th></th></tr>")
    bar_colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b"]
    for (name, t), bc in zip(timings, bar_colors):
        if t is not None:
            pct = t / max_t * 100 if max_t > 0 else 0
            parts.append(
                f"<tr><td>{name}</td><td>{t:.2f}s</td>"
                f'<td><span class="timing-bar" style="width:{pct:.0f}%;background:{bc}">&nbsp;</span></td></tr>'
            )
        else:
            parts.append(
                f'<tr><td>{name}</td><td style="color:var(--muted)">N/A</td><td></td></tr>'
            )
    parts.append(f"<tr><th>Total</th><th>{total_t:.2f}s</th><th></th></tr>")
    parts.append("</table></div>")
    return parts


def section_footer(ctx: ReportContext) -> List[str]:
    """Closing tags."""
    return [template_block("report.html", "footer").substitute()]


SECTIONS = [
    section_header,
    section_question,
    section_input_images,
    section_pose_constraints,
    section_planner,
    section_disambiguation,
    section_assembled_prompt,
    section_program_cache,
    section_generated_code,
    section_retries,
    section_execution,
    section_score_cache,
    section_detected_objects,
    section_planned_vs_detected,
    section_scene_3d,
    section_timing,
    section_footer,
]
