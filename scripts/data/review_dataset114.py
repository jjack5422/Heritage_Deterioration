"""Review dataset114 tiles and quarantine incorrect image/mask pairs safely."""

from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import io
import json
import os
import secrets
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, render_template_string, request, send_file
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "dataset114"
STATES = {
    "unreviewed": "未檢查",
    "keep": "保留",
    "reject": "已隔離",
    "unsure": "待確認",
}
BACKGROUND = (20, 25, 32)


def atomic_write(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 text file without losing its prior copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_audit(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def display_color(class_id: int) -> tuple[int, int, int]:
    """Return stable, distinct colors while reserving dark gray for background."""
    fixed = {1: (255, 103, 92), 2: (55, 196, 221), 4: (245, 190, 62)}
    if class_id in fixed:
        return fixed[class_id]
    hue = (class_id * 0.61803398875) % 1.0
    return tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.68, 0.95))


class Dataset114:
    """Manifest-backed dataset whose rejected pairs live in a reversible quarantine."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.metadata = self.root / "metadata"
        self.manifest_path = self.metadata / "manifest.csv"
        self.classes_path = self.metadata / "classes.json"
        self.quarantine = self.root / "rejected_tiles"
        self.rejected_manifest_path = self.quarantine / "manifest.csv"
        if not self.manifest_path.is_file() or not self.classes_path.is_file():
            raise ValueError("dataset114 requires metadata/manifest.csv and metadata/classes.json")

        contract = json.loads(self.classes_path.read_text(encoding="utf-8"))
        categories = contract.get("categories")
        if not isinstance(categories, list):
            raise ValueError("classes.json categories must be a list")
        self.classes = {0: {"id": 0, "code": "BG", "name": "Background", "color": BACKGROUND}}
        for category in categories:
            class_id = category.get("id")
            if type(class_id) is not int or not isinstance(category.get("code"), str) or not isinstance(category.get("name"), str):
                raise ValueError("Invalid category in classes.json")
            self.classes[class_id] = {**category, "color": display_color(class_id)}

        self.fieldnames, active_rows = self._read_manifest(self.manifest_path, required=True)
        rejected_fields, rejected_rows = self._read_manifest(self.rejected_manifest_path, required=False)
        if rejected_fields and rejected_fields != self.fieldnames:
            raise ValueError("Rejected manifest columns differ from metadata/manifest.csv")
        self.active_rows = {row["image"]: row for row in active_rows}
        self.rejected_rows = {row["image"]: row for row in rejected_rows}
        overlap = set(self.active_rows) & set(self.rejected_rows)
        if overlap:
            raise ValueError(f"Rows occur in active and rejected manifests: {sorted(overlap)[:3]}")

        all_rows = active_rows + rejected_rows
        if not all_rows:
            raise ValueError("No manifest rows found")
        duplicate_images = len({row["image"] for row in all_rows}) != len(all_rows)
        if duplicate_images:
            raise ValueError("Duplicate image paths in manifests")
        self.items = []
        for row in sorted(all_rows, key=lambda value: value["image"]):
            rejected = row["image"] in self.rejected_rows
            image, mask = self._paths(row, rejected)
            self._validate_path(image)
            self._validate_path(mask)
            if not image.is_file() or not mask.is_file():
                state = "rejected" if rejected else "active"
                raise ValueError(f"Missing {state} image/mask pair for {row['image']}")
            class_ids = self._parse_class_ids(row.get("classes_present", ""))
            unknown = class_ids - self.classes.keys()
            if unknown:
                raise ValueError(f"Unknown classes {sorted(unknown)} in {row['image']}")
            self.items.append({
                "name": Path(row["image"]).name,
                "key": row["image"],
                "row": row,
                "image": image,
                "mask": mask,
                "rejected": rejected,
                "class_ids": sorted(class_ids),
            })
        self.by_key = {item["key"]: item for item in self.items}
        label_identity = [(class_id, value["code"]) for class_id, value in sorted(self.classes.items())]
        digest = hashlib.sha256(json.dumps(label_identity, separators=(",", ":")).encode("utf-8"))
        for row in sorted(all_rows, key=lambda value: value["image"]):
            digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        self.fingerprint = digest.hexdigest()

    def _read_manifest(self, path: Path, required: bool) -> tuple[list[str], list[dict[str, str]]]:
        if not path.exists():
            if required:
                raise ValueError(f"Missing manifest: {path}")
            return [], []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or [])
            if not {"image", "mask", "classes_present"}.issubset(fields):
                raise ValueError(f"Manifest lacks required columns: {path}")
            rows = [dict(row) for row in reader]
        return fields, rows

    def _parse_class_ids(self, value: str) -> set[int]:
        if not value:
            return set()
        try:
            return {int(part) for part in value.split(";") if part}
        except ValueError as exc:
            raise ValueError(f"Invalid classes_present value: {value!r}") from exc

    def _source_path(self, value: str) -> Path:
        path = (self.root / value).resolve()
        self._validate_path(path)
        return path

    def _validate_path(self, path: Path) -> None:
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"Manifest path escapes dataset root: {path}")

    def _paths(self, row: dict[str, str], rejected: bool) -> tuple[Path, Path]:
        if rejected:
            return self.quarantine / "images" / Path(row["image"]).name, self.quarantine / "masks" / Path(row["mask"]).name
        return self._source_path(row["image"]), self._source_path(row["mask"])

    def manifest_text(self, rows: list[dict[str, str]]) -> str:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=self.fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return "\ufeff" + stream.getvalue()

    def _rewrite_manifests(self) -> None:
        def order(row: dict[str, str]) -> tuple[str, int, int, str]:
            return (
                row.get("task_id", ""),
                int(row.get("row") or -1),
                int(row.get("column") or -1),
                row["image"],
            )

        active = sorted(self.active_rows.values(), key=order)
        rejected = sorted(self.rejected_rows.values(), key=order)
        atomic_write(self.manifest_path, self.manifest_text(active))
        atomic_write(self.rejected_manifest_path, self.manifest_text(rejected))

    def _transition(self, key: str, reject: bool) -> None:
        item = self.by_key[key]
        if item["rejected"] == reject:
            return
        source_image, source_mask = item["image"], item["mask"]
        target_image, target_mask = self._paths(item["row"], reject)
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_mask.parent.mkdir(parents=True, exist_ok=True)
        if target_image.exists() or target_mask.exists():
            raise OSError(f"Refusing to overwrite quarantine pair for {item['name']}")
        old_active = dict(self.active_rows)
        old_rejected = dict(self.rejected_rows)
        moved_image = moved_mask = False
        try:
            source_image.replace(target_image)
            moved_image = True
            source_mask.replace(target_mask)
            moved_mask = True
            if reject:
                self.rejected_rows[key] = self.active_rows.pop(key)
            else:
                self.active_rows[key] = self.rejected_rows.pop(key)
            self._rewrite_manifests()
        except Exception:
            self.active_rows = old_active
            self.rejected_rows = old_rejected
            try:
                self._rewrite_manifests()
            finally:
                if moved_mask and target_mask.exists():
                    target_mask.replace(source_mask)
                if moved_image and target_image.exists():
                    target_image.replace(source_image)
            raise
        item["rejected"] = reject
        item["image"], item["mask"] = target_image, target_mask

    def reject(self, key: str) -> None:
        self._transition(key, True)

    def restore(self, key: str) -> None:
        self._transition(key, False)

    def load_pair(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        item = self.items[index]
        with Image.open(item["image"]) as image:
            original = np.array(image.convert("RGB"))
        with Image.open(item["mask"]) as mask:
            values = np.array(mask)
        if values.ndim != 2:
            raise ValueError(f"Expected a single-channel class-ID mask: {item['name']}")
        if values.shape != original.shape[:2]:
            raise ValueError(f"Image/mask dimensions differ: {item['name']}")
        known_ids = set(int(value) for value in np.unique(values))
        unknown = known_ids - self.classes.keys()
        if unknown:
            raise ValueError(f"Unknown mask IDs {sorted(unknown)} in {item['name']}")
        colored = np.empty_like(original)
        colored[:] = BACKGROUND
        foreground = values != 0
        for class_id in known_ids:
            colored[values == class_id] = self.classes[class_id]["color"]
        return original, colored, foreground


class ReviewStore:
    def __init__(self, dataset: Dataset114):
        self.dataset = dataset
        self.path = dataset.metadata / "manual_review.json"
        self.audit_path = dataset.metadata / "manual_review.jsonl"
        self.lock = threading.RLock()
        self.records: dict[str, dict] = {}
        self.undo_stack: list[dict] = []
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != dataset.fingerprint:
                raise ValueError("Saved review state belongs to different dataset content")
            self.records = payload.get("records", {})
            self.undo_stack = payload.get("undo_stack", [])
            if set(self.records) - dataset.by_key.keys():
                raise ValueError("Saved review state contains unknown tiles")
        for key, item in dataset.by_key.items():
            recorded = self.records.get(key, {}).get("status", "unreviewed")
            if item["rejected"] and recorded != "reject":
                self.records[key] = {"status": "reject", "note": "", "updated_at": ""}
            elif not item["rejected"] and recorded == "reject":
                raise ValueError(f"Review state marks an active tile as rejected: {key}")

    def _payload(self, records: dict, undo_stack: list[dict]) -> str:
        return json.dumps({
            "schema_version": 1,
            "dataset": str(self.dataset.root),
            "fingerprint": self.dataset.fingerprint,
            "records": records,
            "undo_stack": undo_stack,
        }, ensure_ascii=False, indent=2) + "\n"

    def _commit(self, records: dict, undo_stack: list[dict]) -> None:
        atomic_write(self.path, self._payload(records, undo_stack))
        self.records = records
        self.undo_stack = undo_stack

    def save(self, index: int, status: str, note: str) -> str | None:
        item = self.dataset.items[index]
        key = item["key"]
        with self.lock:
            old_record = self.records.get(key)
            old_rejected = item["rejected"]
            records = dict(self.records)
            undo_stack = list(self.undo_stack)
            now = datetime.now(timezone.utc).isoformat()
            if status == "reject" and not old_rejected:
                undo_stack.append({"key": key, "record": old_record})
                self.dataset.reject(key)
            elif status != "reject" and old_rejected:
                self.dataset.restore(key)
                undo_stack = [entry for entry in undo_stack if entry.get("key") != key]
            records[key] = {"status": status, "note": note, "updated_at": now}
            try:
                self._commit(records, undo_stack)
            except Exception:
                if item["rejected"] != old_rejected:
                    self.dataset._transition(key, old_rejected)
                raise
            try:
                append_audit(self.audit_path, {
                    "timestamp": now,
                    "action": "reject" if status == "reject" and not old_rejected else "restore" if status != "reject" and old_rejected else "review",
                    "key": key,
                    "status": status,
                    "note": note,
                })
            except OSError as exc:
                return f"決定已儲存，但稽核紀錄寫入失敗：{exc}"
        return None

    def undo_last_rejection(self) -> tuple[str, str | None]:
        with self.lock:
            stack = list(self.undo_stack)
            while stack and not self.dataset.by_key.get(stack[-1].get("key"), {}).get("rejected"):
                stack.pop()
            if not stack:
                raise ValueError("目前沒有可還原的隔離操作")
            entry = stack.pop()
            key = entry["key"]
            item = self.dataset.by_key[key]
            records = dict(self.records)
            prior = entry.get("record")
            self.dataset.restore(key)
            if prior is None:
                records.pop(key, None)
                restored_status = "unreviewed"
            else:
                records[key] = prior
                restored_status = prior["status"]
            try:
                self._commit(records, stack)
            except Exception:
                self.dataset.reject(key)
                raise
            warning = None
            try:
                append_audit(self.audit_path, {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": "undo_reject",
                    "key": key,
                    "status": restored_status,
                })
            except OSError as exc:
                warning = f"檔案已還原，但稽核紀錄寫入失敗：{exc}"
            return item["name"], warning

    def can_undo(self) -> bool:
        return any(self.dataset.by_key.get(entry.get("key"), {}).get("rejected") for entry in self.undo_stack)


PAGE = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset114 人工篩選台</title><style>
:root{--ink:#17212b;--muted:#56616d;--canvas:#e9edf0;--paper:#fff;--line:#c6cdd2;--blue:#164e63;--blue-hi:#0e7490;--keep:#176447;--danger:#b4232f;--danger-hi:#951d27;--amber:#8a5a08;--focus:#006bd6;--shadow:0 8px 24px rgb(23 33 43/.09)}
*{box-sizing:border-box}html{color-scheme:light}body{margin:0;background:var(--canvas);color:var(--ink);font:16px/1.5 "Work Sans","Noto Sans TC",system-ui,sans-serif}
button,input,select,textarea{font:inherit}button,select,input,.download{min-height:44px;border:1px solid var(--line);border-radius:4px;background:var(--paper);color:var(--ink);padding:8px 12px}button,.download,select{cursor:pointer;touch-action:manipulation}button:hover,.download:hover{background:#edf3f5}button:active,.download:active{filter:brightness(.92)}button:disabled{cursor:not-allowed;opacity:.5}
:focus-visible{outline:3px solid var(--focus);outline-offset:2px}a{color:var(--blue)}header{background:var(--paper);border-bottom:1px solid var(--line)}.header-inner,main{max-width:1560px;margin:auto;padding:16px 24px}.eyebrow{color:var(--blue);font:700 12px/1.2 ui-monospace,SFMono-Regular,monospace;letter-spacing:.12em;text-transform:uppercase}.title-row{display:flex;align-items:end;justify-content:space-between;gap:16px;flex-wrap:wrap}h1{font:650 clamp(25px,3vw,36px)/1.1 "Aptos Display","Noto Sans TC",system-ui,sans-serif;margin:6px 0 2px;letter-spacing:-.025em}.summary{color:var(--muted);font-size:14px}.progress{height:4px;background:#dce2e5}.progress>span{display:block;height:100%;background:var(--blue-hi);transition:width .18s ease}
.toolbar,.panel,.note-card{background:var(--paper);border:1px solid var(--line);box-shadow:var(--shadow)}.toolbar{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;padding:12px;margin-bottom:12px}.cluster{display:flex;gap:8px;align-items:center;flex-wrap:wrap}label{display:inline-flex;align-items:center;gap:7px;color:var(--muted);font-size:14px}.download{text-decoration:none;display:inline-flex;align-items:center}.review-head{display:flex;align-items:start;justify-content:space-between;gap:16px;margin:14px 0 8px}.filename{font:600 15px/1.4 ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere;margin:0}.position{color:var(--muted);white-space:nowrap;font-size:14px}.actions{display:grid;grid-template-columns:repeat(6,minmax(115px,1fr));gap:8px;margin-bottom:12px}.actions button{font-weight:650}.keep{background:var(--keep);border-color:var(--keep);color:#fff}.reject{background:var(--danger);border-color:var(--danger);color:#fff}.unsure{background:var(--amber);border-color:var(--amber);color:#fff}.keep:hover,.reject:hover,.unsure:hover{filter:brightness(1.1)}.undo{border-color:var(--danger);color:var(--danger)}
.compare{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{min-width:0;padding:12px}.panel-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}.panel h2{font-size:14px;margin:0}.viewport{height:min(58vh,650px);overflow:auto;background:#111820}.stage{position:relative;width:100%;margin:auto}.stage img{display:block;width:100%;height:auto;image-rendering:auto}.stage .mask{position:absolute;inset:0;opacity:.48}.legend{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}.chip{display:inline-flex;align-items:center;gap:5px;padding:3px 7px;border:1px solid var(--line);background:var(--paper);font:12px/1.4 ui-monospace,SFMono-Regular,monospace}.swatch{width:11px;height:11px;border:1px solid #5b6670}.note-card{padding:12px;margin-top:12px}.note-card textarea{display:block;width:100%;margin-top:6px;border:1px solid var(--line);padding:8px;resize:vertical}.message{min-height:24px;color:var(--blue);font-weight:600;font-size:14px;margin:8px 0}.message.error{color:var(--danger)}.empty{text-align:center;background:var(--paper);border:1px dashed var(--line);padding:56px 16px;color:var(--muted)}details{margin:14px 0;color:var(--muted);font-size:14px}input[type=range]{padding:0;width:120px}input[type=number]{width:82px}input[type=checkbox]{min-height:0;width:18px;height:18px}
@media(max-width:900px){.header-inner,main{padding:12px}.actions{grid-template-columns:repeat(3,1fr)}.compare{grid-template-columns:1fr}.viewport{height:min(72vw,560px)}}
@media(max-width:520px){.actions{grid-template-columns:1fr 1fr}.position{white-space:normal}.toolbar{align-items:stretch}.cluster{align-items:stretch}.cluster>label{width:100%;justify-content:space-between}.cluster select,.cluster input{flex:1;min-width:0}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
</style></head><body>
<header><div class="header-inner"><div class="eyebrow">MGLST · TILE QUALITY CONTROL</div><div class="title-row"><div><h1>Dataset114 人工篩選台</h1><div class="summary">辨識明顯漏標與誤標；排除項目移入可復原隔離區。</div></div><strong id="counts" role="status"></strong></div></div><div class="progress"><span id="progress"></span></div></header>
<main><div class="toolbar"><div class="cluster"><label>狀態<select id="filter"><option value="all">全部</option><option value="unreviewed">未檢查</option><option value="keep">保留</option><option value="reject">已隔離</option><option value="unsure">待確認</option></select></label><label>標註類別<select id="classFilter"><option value="all">全部類別</option>{% for item in classes %}<option value="{{item.id}}">{{item.code}} · {{item.name}}</option>{% endfor %}</select></label><label>搜尋<input id="search" type="search" placeholder="檔名或來源" size="16"></label></div><div class="cluster"><label>跳至第<input id="jump" type="number" min="1" value="1">張</label><button id="go">前往</button><button id="undo" class="undo">還原上一筆隔離</button></div></div>
<div id="message" class="message" role="status" aria-live="polite"></div><div id="review"><div class="review-head"><h2 id="filename" class="filename"></h2><span id="position" class="position"></span></div><div class="actions"><button id="previous">← 上一張</button><button class="keep" data-status="keep">保留 [1]</button><button class="reject" data-status="reject">移至隔離區 [2]</button><button class="unsure" data-status="unsure">待確認 [3]</button><button data-status="unreviewed">重設 [0]</button><button id="next">下一張 →</button></div>
<div class="compare"><section class="panel"><div class="panel-title"><h2>原始 tile</h2><span id="activeBadge" class="chip"></span></div><div class="viewport" id="left"><div class="stage"><img id="original" alt="目前 tile 原圖"></div></div></section><section class="panel"><div class="panel-title"><h2>標註檢視 <span id="viewTitle">／疊圖</span></h2><div class="cluster"><label>模式<select id="mode"><option value="overlay">疊圖</option><option value="mask">純 mask</option></select></label><label>透明度<input id="alpha" type="range" min="0" max="100" value="48"><span id="alphaValue">48%</span></label><label>放大<select id="zoom"><option value="1">適合視窗</option><option value="2">2×</option><option value="4">4×</option></select></label></div></div><div class="viewport" id="right"><div class="stage"><img id="base" alt="標註下方原圖"><img id="mask" class="mask" alt="目前 tile 的彩色標註"></div></div></section></div><div id="legend" class="legend"></div>
<div class="note-card"><label for="note">備註</label><textarea id="note" rows="2" maxlength="4000" placeholder="例如：裂縫漏標、整片誤標、邊界無法判斷"></textarea><div class="cluster" style="margin-top:8px"><button id="saveNote">儲存備註</button><label><input id="advance" type="checkbox" checked>判定後自動下一張</label></div></div></div><div id="empty" class="empty" hidden>目前條件沒有 tile；請調整狀態、類別或搜尋。</div>
<details><summary>資料安全與快捷鍵</summary><p>按 1 保留、2 移至隔離區、3 待確認、0 重設；方向鍵前後移動。隔離會成對移動 image 與 mask，並同步更新主 manifest；不會永久刪除。可用「還原上一筆隔離」或在已隔離 tile 上改選其他狀態來復原。所有決定立即保存，僅限單人、本機單一分頁使用。</p></details></main>
<script>
const token={{token|tojson}}, states={{states|tojson}}, classInfo={{class_info|tojson}}, $=id=>document.getElementById(id);let items=[],visible=[],current=null,busy=false,ready=false,imageRatio=1;
function message(text,error=false){$('message').textContent=text;$('message').classList.toggle('error',error)}
function lock(value){busy=value;document.querySelectorAll('button,select,input').forEach(control=>control.disabled=value);if(!value)$('undo').disabled=!items.some(item=>item.undoable)}
async function api(url,options){const response=await fetch(url,options);const type=response.headers.get('content-type')||'';const data=type.includes('json')?await response.json():{};if(!response.ok)throw Error(data.error||`HTTP ${response.status}`);return data}
function refresh(){const q=$('search').value.toLowerCase(),status=$('filter').value,classId=$('classFilter').value;visible=items.filter(item=>(status==='all'||item.status===status)&&(classId==='all'||item.class_ids.includes(Number(classId)))&&(item.name.toLowerCase().includes(q)||item.key.toLowerCase().includes(q)));const counts={unreviewed:0,keep:0,reject:0,unsure:0};items.forEach(item=>counts[item.status]++);$('counts').textContent=`${items.length} 張｜保留 ${counts.keep}｜隔離 ${counts.reject}｜待確認 ${counts.unsure}｜未檢查 ${counts.unreviewed}`;$('progress').style.width=`${100*(items.length-counts.unreviewed)/items.length}%`;$('undo').disabled=!items.some(item=>item.undoable)}
function renderLegend(item){$('legend').replaceChildren(...item.class_ids.map(id=>{const info=classInfo[id],chip=document.createElement('span'),swatch=document.createElement('i');chip.className='chip';swatch.className='swatch';swatch.style.background=`rgb(${info.color.join(',')})`;chip.append(swatch,`${info.code} · ${info.name}`);return chip}))}
async function show(index){ready=false;current=index;const item=items[index];$('review').hidden=!item;$('empty').hidden=!!item;if(!item)return;$('filename').textContent=item.key;$('position').textContent=`第 ${index+1} / ${items.length} 張 · ${states[item.status]} · 篩選結果 ${visible.length} 張`;$('jump').value=index+1;$('note').value=item.note;$('activeBadge').textContent=item.rejected?'已在隔離區':'有效資料集';renderLegend(item);for(const id of ['left','right'])$(id).scrollTo(0,0);for(const id of ['original','base','mask'])$(id).removeAttribute('src');const stamp=Date.now(),image=new Image(),mask=new Image();image.src=`/image/${index}/original?v=${stamp}`;mask.src=`/image/${index}/mask?v=${stamp}`;await Promise.all([image.decode(),mask.decode()]);imageRatio=image.naturalWidth/image.naturalHeight;$('original').src=image.src;$('base').src=image.src;$('mask').src=mask.src;applyView();ready=true}
function applyView(){if(current===null)return;const pure=$('mode').value==='mask';$('mask').src=`/image/${current}/${pure?'mask':'overlay'}?v=${Date.now()}`;$('mask').style.opacity=pure?1:Number($('alpha').value)/100;$('base').style.visibility=pure?'hidden':'visible';$('viewTitle').textContent=pure?'／純 mask':'／疊圖';$('alphaValue').textContent=$('alpha').value+'%';document.querySelectorAll('.viewport').forEach(view=>{const width=Math.min(view.clientWidth,view.clientHeight*imageRatio);view.querySelector('.stage').style.width=width*Number($('zoom').value)+'px'})}
async function save(status){const item=items[current],data=await api('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify({index:current,status,note:$('note').value})});Object.assign(item,data.item);items.forEach(value=>value.undoable=data.undo_key===value.key);refresh();message(data.warning||`已儲存：${item.name} → ${states[status]}`,!!data.warning)}
async function saveDirty(){if(current!==null&&$('note').value!==items[current].note)await save(items[current].status)}
async function perform(action){if(busy)return;lock(true);try{await action()}catch(error){message(`操作失敗：${error.message}。檔案未永久刪除，請重試。`,true)}finally{lock(false)}}
async function move(delta){await saveDirty();const position=visible.findIndex(item=>item.index===current),target=visible[position+delta];if(target)await show(target.index);else message(delta>0?'已到最後一張。':'已到第一張。')}
document.querySelectorAll('[data-status]').forEach(button=>button.onclick=()=>perform(async()=>{if(!ready)throw Error('影像尚未載入完成');const old=visible.map(item=>item.index),position=old.indexOf(current);await save(button.dataset.status);const shouldAdvance=$('advance').checked||!visible.some(item=>item.index===current);if(shouldAdvance){const next=old.slice(position+1).find(index=>visible.some(item=>item.index===index));await show(next??(visible.some(item=>item.index===current)?current:(visible[0]?.index??null)))}else await show(current)}));
$('previous').onclick=()=>perform(()=>move(-1));$('next').onclick=()=>perform(()=>move(1));$('saveNote').onclick=()=>perform(()=>save(items[current].status));for(const id of ['filter','classFilter','search'])$(id).onchange=()=>perform(async()=>{await saveDirty();refresh();await show(visible[0]?.index??null)});$('go').onclick=()=>perform(async()=>{const index=Number($('jump').value)-1;if(!Number.isInteger(index)||index<0||index>=items.length)throw Error('請輸入有效編號');await saveDirty();$('filter').value='all';$('classFilter').value='all';$('search').value='';refresh();await show(index)});
$('undo').onclick=()=>perform(async()=>{const data=await api('/api/undo',{method:'POST',headers:{'X-Review-Token':token}});items=await api('/api/items');refresh();await show(items.find(item=>item.key===data.key)?.index??current);message(data.warning||`已還原：${data.name}`,!!data.warning)});['mode','alpha','zoom'].forEach(id=>$(id).oninput=applyView);window.addEventListener('resize',applyView);let scrolling=false;for(const [a,b] of [['left','right'],['right','left']])$(a).onscroll=()=>{if(scrolling)return;scrolling=true;$(b).scrollTop=$(a).scrollTop;$(b).scrollLeft=$(a).scrollLeft;requestAnimationFrame(()=>scrolling=false)};
document.addEventListener('keydown',event=>{if(event.ctrlKey||event.metaKey||event.altKey||event.shiftKey||event.repeat||['INPUT','TEXTAREA','SELECT','BUTTON','A'].includes(event.target.tagName)||busy||current===null)return;const keys={'1':'keep','2':'reject','3':'unsure','0':'unreviewed'};if(keys[event.key]){event.preventDefault();document.querySelector(`[data-status="${keys[event.key]}"]`).click()}else if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();perform(()=>move(event.key==='ArrowLeft'?-1:1))}});window.addEventListener('beforeunload',event=>{if(busy||(current!==null&&$('note').value!==items[current].note)){event.preventDefault();event.returnValue=''}});
perform(async()=>{items=await api('/api/items');refresh();$('jump').max=items.length;await show(items.find(item=>item.status==='unreviewed')?.index??0);message('資料配對已驗證，可以開始篩選。')});
</script></body></html>"""


def create_app(dataset: Dataset114, store: ReviewStore) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
    token = secrets.token_urlsafe(32)

    def item_payload(index: int) -> dict:
        item = dataset.items[index]
        record = store.records.get(item["key"], {})
        undo_key = store.undo_stack[-1]["key"] if store.can_undo() else None
        return {
            "index": index,
            "name": item["name"],
            "key": item["key"],
            "status": record.get("status", "unreviewed"),
            "note": record.get("note", ""),
            "rejected": item["rejected"],
            "class_ids": item["class_ids"],
            "undoable": item["key"] == undo_key,
        }

    @app.after_request
    def secure_response(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'"
        return response

    @app.get("/")
    def home():
        classes = [dataset.classes[class_id] for class_id in sorted({value for item in dataset.items for value in item["class_ids"]})]
        return render_template_string(PAGE, token=token, states=STATES, classes=classes, class_info=dataset.classes)

    @app.get("/api/items")
    def items():
        with store.lock:
            return jsonify([item_payload(index) for index in range(len(dataset.items))])

    @app.post("/api/review")
    def review():
        if not secrets.compare_digest(request.headers.get("X-Review-Token", ""), token):
            abort(403)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Expected a JSON object"), 400
        index, status, note = payload.get("index"), payload.get("status"), payload.get("note", "")
        if type(index) is not int or not 0 <= index < len(dataset.items):
            return jsonify(error="Invalid tile index"), 400
        if status not in STATES or not isinstance(note, str) or len(note) > 4000:
            return jsonify(error="Invalid status or note"), 400
        try:
            warning = store.save(index, status, note)
        except (OSError, ValueError) as exc:
            return jsonify(error=str(exc)), 500
        undo_key = store.undo_stack[-1]["key"] if store.can_undo() else None
        return jsonify(ok=True, warning=warning, item=item_payload(index), undo_key=undo_key)

    @app.post("/api/undo")
    def undo():
        if not secrets.compare_digest(request.headers.get("X-Review-Token", ""), token):
            abort(403)
        try:
            name, warning = store.undo_last_rejection()
        except ValueError as exc:
            return jsonify(error=str(exc)), 409
        key = next(item["key"] for item in dataset.items if item["name"] == name)
        return jsonify(ok=True, key=key, name=name, warning=warning)

    @app.get("/image/<int:index>/<kind>")
    def image(index: int, kind: str):
        if not 0 <= index < len(dataset.items) or kind not in {"original", "mask", "overlay"}:
            abort(404)
        try:
            original, colored, foreground = dataset.load_pair(index)
        except (OSError, ValueError) as exc:
            return jsonify(error=str(exc)), 422
        if kind == "original":
            result = Image.fromarray(original)
        elif kind == "mask":
            result = Image.fromarray(colored)
        else:
            result = Image.fromarray(np.dstack((colored, foreground.astype(np.uint8) * 255)))
        stream = io.BytesIO()
        result.save(stream, format="PNG")
        stream.seek(0)
        return send_file(stream, mimetype="image/png")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="人工檢查 dataset114 tiles，將明顯漏標／誤標配對移至可復原隔離區。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="dataset114 根目錄")
    parser.add_argument("--host", default="127.0.0.1", help="預設只允許本機連線")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--check", action="store_true", help="完整驗證 manifest、image、mask、尺寸與類別後退出")
    args = parser.parse_args()
    dataset = Dataset114(args.dataset)
    if args.check:
        for index in range(len(dataset.items)):
            dataset.load_pair(index)
        print(f"OK: {len(dataset.items)} image/mask pairs; manifests, dimensions, and mask IDs verified.")
        return

    import fcntl

    lock_path = dataset.metadata / ".manual_review.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("dataset114 is already open in another review process")
        store = ReviewStore(dataset)
        print(f"Loaded {len(dataset.items)} pairs. Review: http://{args.host}:{args.port}", flush=True)
        print(f"Quarantine: {dataset.quarantine}", flush=True)
        create_app(dataset, store).run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
