from __future__ import annotations

import html
import json
import subprocess
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .boundary_split import split_combined_audio_to_lines
from .config import DubbingConfig
from .generation_report import replace_metadata_row, write_generation_report
from .tts_indextts2 import align_raw_line_audio
from .utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


def _status_needs_review(row: dict[str, Any]) -> bool:
    status = str(row.get("generation_status") or "")
    flags = set(row.get("quality_flags", []) or [])
    return (
        status.startswith("flagged_after_")
        or status in {"wrong_pause_manual_review", "failed_pause_detection", "failed_generation"}
        or bool(
            flags.intersection(
                {
                    "wrong_pause_manual_review",
                    "duration_speedup_gt_1_5",
                    "duration_short_lt_0_5",
                    "final_speedup_gt_1_15",
                    "abnormal_pause_detected",
                    "pause_repaired",
                }
            )
        )
    )


def build_pause_review_manifest(output_dir: str | Path, write: bool = True) -> dict[str, Any]:
    output = Path(output_dir)
    work_dir = output / "work"
    generation_units_path = work_dir / "generation_units.json"
    if generation_units_path.exists():
        units = read_json(generation_units_path)
        flagged_units = [unit for unit in units if _status_needs_review(unit)]
        report_path = output / "generation_report.json"
        report = read_json(report_path) if report_path.exists() else {}
        manifest = {
            "output_dir": str(output),
            "emotion_json": report.get("emotion_json", ""),
            "mode": "generation_units",
            "unit_count": len(units),
            "flagged_units": flagged_units,
            "units": units,
            "flagged_line_ids": [
                int(line_id)
                for unit in flagged_units
                for line_id in (unit.get("source_line_indices") or unit.get("line_ids") or [])
            ],
        }
        if write:
            write_json(work_dir / "pause_review_manifest.json", manifest)
        return manifest

    line_rows = read_jsonl(work_dir / "line_generation_metadata.jsonl")
    group_rows = read_jsonl(work_dir / "group_generation_metadata.jsonl")
    report_path = output / "generation_report.json"
    report = read_json(report_path) if report_path.exists() else {}
    emotion_json = report.get("emotion_json", "")

    flagged_line_ids = [int(row["line_id"]) for row in line_rows if row.get("line_id") is not None and _status_needs_review(row)]
    flagged_groups = []
    for group in group_rows:
        line_ids = [int(line_id) for line_id in group.get("line_ids", [])]
        if any(line_id in flagged_line_ids for line_id in line_ids) or group.get("quality_flags"):
            flagged_groups.append(group)

    manifest = {
        "output_dir": str(output),
        "emotion_json": emotion_json,
        "line_count": len(line_rows),
        "group_count": len(group_rows),
        "flagged_line_ids": flagged_line_ids,
        "flagged_groups": flagged_groups,
        "lines": line_rows,
    }
    if write:
        write_json(work_dir / "pause_review_manifest.json", manifest)
    return manifest


def _find_group(output_dir: Path, group_id: str) -> dict[str, Any]:
    groups_path = output_dir / "work" / "sentence_groups.json"
    groups = read_json(groups_path)
    for group in groups:
        if str(group.get("group_id")) == str(group_id):
            return group
    raise KeyError(f"Group {group_id!r} not found in {groups_path}")


def _find_group_metadata(output_dir: Path, group_id: str) -> dict[str, Any]:
    for group in read_jsonl(output_dir / "work" / "group_generation_metadata.jsonl"):
        if str(group.get("group_id")) == str(group_id):
            return group
    raise KeyError(f"Group metadata {group_id!r} not found")


def apply_manual_cuts(output_dir: str | Path, group_id: str, cut_points_sec: list[float], config: DubbingConfig | None = None) -> dict[str, Any]:
    config = config or DubbingConfig()
    output = Path(output_dir)
    work_dir = ensure_dir(output / "work")
    raw_dir = ensure_dir(output / "lines" / "raw")
    aligned_dir = ensure_dir(output / "lines" / "aligned")
    public_dir = ensure_dir(output / "lines")
    group = _find_group(output, group_id)
    group_meta = _find_group_metadata(output, group_id)
    selected_audio = group_meta.get("selected_audio") or group_meta.get("repaired_audio") or group_meta.get("uncut_audio")
    if not selected_audio:
        raise RuntimeError(f"Group {group_id} has no selected audio to cut")

    split = split_combined_audio_to_lines(
        selected_audio,
        group.get("lines", []),
        raw_dir,
        manual_cut_points_sec=cut_points_sec,
        min_pause_ms=int(config.sentence_grouping.boundary_min_silence_ms),
        silence_db_offset=float(config.pause_detection.silence_db_offset),
        min_piece_ms=int(config.pause_repair.min_piece_ms),
    )
    rows = read_jsonl(work_dir / "line_generation_metadata.jsonl")
    updated_rows = rows
    changed_rows: list[dict[str, Any]] = []
    for line in group.get("lines", []):
        line_id = int(line["id"])
        raw_path = Path(split.piece_paths[line_id])
        aligned_path = aligned_dir / f"line_{line_id:04d}.wav"
        public_path = public_dir / f"line_{line_id:04d}.wav"
        raw_duration, alignment = align_raw_line_audio(
            raw_path,
            aligned_path,
            public_path,
            float(line.get("target_duration", 0.0)),
            config,
            action_prefix="manual_cut",
        )
        existing = next((row for row in updated_rows if int(row.get("line_id", -1)) == line_id), {})
        changed = {
            **existing,
            "line_id": line_id,
            "raw_output": str(raw_path),
            "aligned_output": str(aligned_path),
            "public_output": str(public_path),
            "raw_duration": raw_duration,
            "final_duration": alignment["final_duration"],
            "alignment_action": alignment["alignment_action"],
            "generation_status": "manual_cut_applied",
            "manual_cut_points_sec": cut_points_sec,
            "split_boundary_cuts": [cut.to_dict() for cut in split.cuts],
            "quality_flags": sorted(set((existing.get("quality_flags", []) or []) + ["manual_cut_applied"])),
        }
        updated_rows = replace_metadata_row(updated_rows, changed)
        changed_rows.append(changed)

    write_jsonl(work_dir / "line_generation_metadata.jsonl", updated_rows)
    write_generation_report(output, updated_rows, emotion_json=build_pause_review_manifest(output, write=False).get("emotion_json", ""))
    manifest = build_pause_review_manifest(output, write=True)
    return {"split": split.to_dict(), "changed_rows": changed_rows, "manifest": manifest}


def _html_page(output_dir: Path) -> bytes:
    title = f"Pause Review - {output_dir.name}"
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f5; color: #222; }}
    h1 {{ font-size: 22px; }}
    .group {{ border: 1px solid #ccc; background: white; padding: 14px; margin: 14px 0; border-radius: 6px; }}
    canvas {{ width: 100%; height: 120px; border: 1px solid #ddd; background: #fafafa; }}
    code {{ background: #eee; padding: 1px 4px; }}
    button {{ margin: 4px 6px 4px 0; }}
    input {{ width: 320px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div id="app">Loading...</div>
<script>
async function drawWave(canvas, url) {{
  const ctx = canvas.getContext('2d');
  const res = await fetch(url);
  const buf = await res.arrayBuffer();
  const audio = await new AudioContext().decodeAudioData(buf);
  const data = audio.getChannelData(0);
  const w = canvas.width = canvas.clientWidth * devicePixelRatio;
  const h = canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#2f6f82';
  ctx.beginPath();
  const step = Math.max(1, Math.floor(data.length / w));
  for (let x = 0; x < w; x++) {{
    let min = 1, max = -1;
    for (let i = x * step; i < Math.min(data.length, (x + 1) * step); i++) {{
      min = Math.min(min, data[i]); max = Math.max(max, data[i]);
    }}
    ctx.moveTo(x, (1 + min) * h / 2);
    ctx.lineTo(x, (1 + max) * h / 2);
  }}
  ctx.stroke();
}}
function audioUrl(path) {{ return '/audio?path=' + encodeURIComponent(path); }}
function pauseLabels(item) {{
  const attempts = item.pause_attempts || item.attempts || [];
  const labels = [];
  for (const attempt of attempts) {{
    for (const pause of (attempt.pauses || [])) {{
      if (pause.allowed) continue;
      labels.push(`${{pause.start_sec ?? pause.pause_start ?? 0}}-${{pause.end_sec ?? pause.pause_end ?? 0}}s ${{pause.reason || ''}}`);
    }}
  }}
  return labels;
}}
async function load() {{
  const manifest = await (await fetch('/manifest')).json();
  const root = document.getElementById('app');
  root.innerHTML = '';
  if (manifest.mode === 'generation_units') {{
    const units = manifest.flagged_units || [];
    if (!units.length) root.innerHTML = '<p>No flagged units.</p>';
    for (const unit of units) {{
      const div = document.createElement('div');
      div.className = 'group';
      const finalAudio = unit.aligned_audio_path || unit.aligned_output || unit.public_output || '';
      const rawAudio = unit.raw_audio_path || unit.raw_output || '';
      const lines = unit.source_line_indices || unit.line_ids || [];
      const labels = pauseLabels(unit);
      div.innerHTML = `<h2>${{unit.unit_id}} lines ${{lines.join(', ')}}</h2>
        <p>${{unit.text || ''}}</p>
        <p>Span: <code>${{unit.start}}-${{unit.end}}</code> target <code>${{unit.span_target_duration}}</code> scale <code>${{unit.duration_scale || ''}}</code></p>
        <p>Flags: <code>${{(unit.quality_flags || []).join(', ')}}</code></p>
        <p>Raw</p><audio controls src="${{audioUrl(rawAudio)}}"></audio>
        <p>Final</p><audio controls src="${{audioUrl(finalAudio)}}"></audio>
        <canvas></canvas>
        <p>Pauses: <code>${{labels.join(' | ') || 'none'}}</code></p>
        <p><button data-regen>Regenerate unit</button></p>
        <pre data-result></pre>`;
      root.appendChild(div);
      drawWave(div.querySelector('canvas'), audioUrl(finalAudio || rawAudio)).catch(err => div.querySelector('pre').textContent = err);
      div.querySelector('[data-regen]').onclick = async () => {{
        const response = await fetch('/regenerate', {{method:'POST', body: JSON.stringify({{line_ids: lines}})}});
        div.querySelector('[data-result]').textContent = await response.text();
      }};
    }}
    return;
  }}
  if (!manifest.flagged_groups.length) root.innerHTML = '<p>No flagged groups.</p>';
  for (const group of manifest.flagged_groups) {{
    const div = document.createElement('div');
    div.className = 'group';
    const selected = group.selected_audio || group.repaired_audio || group.uncut_audio;
    const cuts = ((group.split || {{}}).cuts || []).map(c => c.cut_sec).join(', ');
    div.innerHTML = `<h2>${{group.group_id}} lines ${{group.line_ids.join(', ')}}</h2>
      <p>${{group.text || ''}}</p>
      <p>Flags: <code>${{(group.quality_flags || []).join(', ')}}</code></p>
      <audio controls src="${{audioUrl(selected)}}"></audio>
      <canvas></canvas>
      <p>Cut points seconds: <input value="${{cuts}}" data-cuts><button data-apply>Apply cuts</button>
      <button data-regen>Regenerate group</button></p>
      <pre data-result></pre>`;
    root.appendChild(div);
    drawWave(div.querySelector('canvas'), audioUrl(selected)).catch(err => div.querySelector('pre').textContent = err);
    div.querySelector('[data-apply]').onclick = async () => {{
      const values = div.querySelector('[data-cuts]').value.split(',').map(v => parseFloat(v.trim())).filter(v => !Number.isNaN(v));
      const response = await fetch('/apply-cut', {{method:'POST', body: JSON.stringify({{group_id: group.group_id, cut_points_sec: values}})}});
      div.querySelector('[data-result]').textContent = await response.text();
    }};
    div.querySelector('[data-regen]').onclick = async () => {{
      const response = await fetch('/regenerate', {{method:'POST', body: JSON.stringify({{line_ids: group.line_ids}})}});
      div.querySelector('[data-result]').textContent = await response.text();
    }};
  }}
}}
load().catch(err => document.getElementById('app').textContent = err);
</script>
</body>
</html>"""
    return body.encode("utf-8")


def serve_pause_review(output_dir: str | Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    output = Path(output_dir).resolve()
    build_pause_review_manifest(output, write=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, data: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                self._send(200, _html_page(output), "text/html; charset=utf-8")
                return
            if parsed.path == "/manifest":
                self._send(200, json.dumps(build_pause_review_manifest(output, write=True), ensure_ascii=False).encode("utf-8"))
                return
            if parsed.path == "/audio":
                query = urllib.parse.parse_qs(parsed.query)
                path = Path(query.get("path", [""])[0])
                if not path.exists():
                    self._send(404, b"missing audio", "text/plain")
                    return
                self._send(200, path.read_bytes(), "audio/wav")
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            try:
                if self.path == "/apply-cut":
                    result = apply_manual_cuts(output, str(payload["group_id"]), [float(v) for v in payload.get("cut_points_sec", [])])
                    self._send(200, json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"))
                    return
                if self.path == "/regenerate":
                    line_ids = ",".join(str(int(v)) for v in payload.get("line_ids", []))
                    manifest = build_pause_review_manifest(output, write=False)
                    cmd = [
                        sys.executable,
                        "-m",
                        "stable_dubbing.main",
                        "regenerate-lines",
                        "--line-ids",
                        line_ids,
                        "--output-dir",
                        str(output),
                        "--emotion-json",
                        str(manifest.get("emotion_json", "")),
                    ]
                    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
                    build_pause_review_manifest(output, write=True)
                    self._send(200, json.dumps({"command": cmd, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}, indent=2).encode("utf-8"))
                    return
                self._send(404, b"not found", "text/plain")
            except Exception as exc:
                self._send(500, json.dumps({"error": str(exc)}, indent=2).encode("utf-8"))

    server = ThreadingHTTPServer((host, int(port)), Handler)
    print(f"Pause review server: http://{host}:{int(port)}")
    print(f"Output directory: {output}")
    server.serve_forever()
