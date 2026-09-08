"""Browse original images / existing labels and persist manual selections.

Run with crackseg_env/bin/python scripts/data/review_dataset.py.
Requires Flask, Pillow and NumPy. No images or masks are modified.
"""

from __future__ import annotations

import argparse
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
DEFAULT_DATASET = ROOT / "古蹟裂縫/01_CVAT原始匯出/clean_v2_multiclass_512_cvat"
DEFAULT_OUTPUT = ROOT / "outputs/dataset_review"
STATES = {"unreviewed": "未檢查", "keep": "保留", "reject": "排除", "unsure": "待確認"}
COLORS = {
    "background": (0, 0, 0), "crack": (255, 24, 3),
    "loss": (9, 249, 213), "shrinkage": (149, 0, 222),
    "craquelure": (102, 255, 102), "flaking": (236, 236, 0),
    "stain": (93, 149, 13), "ignore": (160, 160, 160),
}
LABELS = {
    "background": "背景", "crack": "裂縫", "loss": "缺失",
    "shrinkage": "皺縮", "craquelure": "龜裂", "flaking": "起甲",
    "stain": "汙漬（候選）", "ignore": "忽略區域",
}


def atomic_write(path: Path, content: str) -> None:
    """Replace one output atomically; a failed write preserves its prior copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class ReviewDataset:
    def __init__(self, dataset: Path):
        self.root = dataset.resolve()
        if (self.root / "JPEGImages").is_dir():
            image_dir, mask_dir = self.root / "JPEGImages", self.root / "SegmentationClass"
            self.rgb_masks = True
            labelmap = self.root / "labelmap.txt"
            self.colors = {}
            for line in labelmap.read_text(encoding="utf-8-sig").splitlines():
                if line.strip() and not line.startswith("#"):
                    name, rgb, *_ = line.split(":")
                    self.colors[name] = tuple(int(v) for v in rgb.split(","))
            self.class_ids = {}
            contract = labelmap.read_bytes()
        else:
            image_dir, mask_dir = self.root / "images", self.root / "masks"
            self.rgb_masks = False
            manifest_path = self.root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.class_ids = manifest["label_contract"]["class_ids"].copy()
            self.class_ids["ignore"] = manifest["label_contract"].get("ignore_value", 255)
            unknown = set(self.class_ids) - COLORS.keys()
            if unknown:
                raise ValueError(f"No display colors defined for: {sorted(unknown)}")
            self.colors = {name: COLORS[name] for name in self.class_ids}
            contract = manifest_path.read_bytes()
        if "background" not in self.colors:
            raise ValueError("Label contract must define background")
        if not mask_dir.is_dir():
            raise ValueError(f"Missing mask directory: {mask_dir}")
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
        if not images:
            raise ValueError(f"No images found: {image_dir}")
        if len({p.stem for p in images}) != len(images):
            raise ValueError("Duplicate image stems; cannot pair masks unambiguously")
        self.items = []
        digest = hashlib.sha256(contract)
        for image in images:
            mask = mask_dir / f"{image.stem}.png"
            if not mask.is_file():
                raise ValueError(f"Missing matching mask: {mask}")
            self.items.append({"name": image.name, "image": str(image), "mask": str(mask)})
            digest.update(image.name.encode("utf-8"))
            for path in (image, mask):
                digest.update(hashlib.sha256(path.read_bytes()).digest())
        self.fingerprint = digest.hexdigest()

    def load_pair(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        item = self.items[index]
        with Image.open(item["image"]) as image:
            original = np.array(image.convert("RGB"))
        with Image.open(item["mask"]) as mask:
            values = np.array(mask.convert("RGB") if self.rgb_masks else mask)
        if values.shape[:2] != original.shape[:2]:
            raise ValueError(f"Image/mask dimensions differ: {item['name']}")
        if not self.rgb_masks and values.ndim != 2:
            raise ValueError(f"Expected a single-channel class-ID mask: {item['name']}")
        colored = np.zeros_like(original)
        known = np.zeros(values.shape[:2], dtype=bool)
        foreground = np.zeros_like(known)
        for name, color in self.colors.items():
            pixels = np.all(values == color, axis=-1) if self.rgb_masks else values == self.class_ids[name]
            colored[pixels] = color
            known |= pixels
            if name != "background":
                foreground |= pixels
        if not known.all():
            raise ValueError(f"Unknown mask colors/IDs in {item['name']}; refusing to hide them")
        return original, colored, foreground


class ReviewStore:
    def __init__(self, dataset: ReviewDataset, output: Path):
        self.dataset = dataset
        self.output = output.resolve()
        if self.output == dataset.root or dataset.root in self.output.parents:
            raise ValueError("Review output must be outside the source dataset")
        self.path = self.output / "review.json"
        self.lock = threading.Lock()
        self.records = {}
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("fingerprint") != dataset.fingerprint or saved.get("dataset") != str(dataset.root):
                raise ValueError("Saved progress belongs to different/changed data. Use a different --output directory.")
            self.records = saved["records"]
            names = {item["name"] for item in dataset.items}
            if not isinstance(self.records, dict) or set(self.records) - names:
                raise ValueError("Invalid saved review records")
            for record in self.records.values():
                if record.get("status") not in STATES or not isinstance(record.get("note"), str):
                    raise ValueError("Invalid saved status or note")

    def csv_text(self, records: dict, selected_only: bool = False) -> str:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=["name", "status", "status_zh", "note", "image", "mask", "updated_at"])
        writer.writeheader()
        for item in self.dataset.items:
            record = records.get(item["name"], {})
            status = record.get("status", "unreviewed")
            if selected_only and status != "keep":
                continue
            row = {**item, "status": status, "status_zh": STATES[status], "note": record.get("note", ""), "updated_at": record.get("updated_at", "")}
            # Keep spreadsheets from interpreting user notes or filenames as formulas.
            writer.writerow({k: "'" + v if v.startswith(("=", "+", "-", "@", "\t", "\r")) else v for k, v in row.items()})
        return "\ufeff" + stream.getvalue()

    def save(self, index: int, status: str, note: str) -> str | None:
        with self.lock:
            records = dict(self.records)
            records[self.dataset.items[index]["name"]] = {
                "status": status, "note": note, "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            payload = {"schema_version": 1, "dataset": str(self.dataset.root), "fingerprint": self.dataset.fingerprint, "records": records}
            atomic_write(self.path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            self.records = records
            try:
                atomic_write(self.output / "review.csv", self.csv_text(records))
                atomic_write(self.output / "selected.csv", self.csv_text(records, True))
            except OSError as exc:
                return f"進度已儲存，但 CSV 匯出失敗：{exc}。請用下載按鈕取得最新名單。"
        return None


PAGE = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>影像與舊標註篩選</title><style>
:root{--bg:#f3f5f7;--panel:#fff;--text:#17212e;--muted:#526072;--line:#b8c3ce;--accent:#185b96;--keep:#17613b;--reject:#a52835;--unsure:#805600}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,"Noto Sans TC",sans-serif}
main{max-width:1500px;margin:auto;padding:24px}h1{font-size:25px;margin:0}h2{font-size:16px;margin:0}p{margin:6px 0 16px;color:var(--muted)}
.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.spread{justify-content:space-between}.toolbar,.card{padding:16px;background:var(--panel);border:1px solid var(--line);border-radius:10px;margin-top:16px}
button,select,input,textarea,.download{font:inherit;border:1px solid var(--line);border-radius:6px;padding:9px 12px;background:var(--panel);color:var(--text);min-height:44px}
button,.download,select{cursor:pointer}button:hover,.download:hover{background:#e8edf2}button:disabled{opacity:.45;cursor:wait}a{color:var(--accent)}
:focus-visible{outline:3px solid var(--accent);outline-offset:3px}input[type=range]{padding:0;width:130px}input[type=number]{width:90px}input[type=checkbox]{min-height:0;width:18px;height:18px}
label{display:inline-flex;align-items:center;gap:8px}.compare{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{min-width:0}.viewport{height:min(45vh,512px);overflow:auto;background:#121922;margin-top:12px}
.stage{position:relative;width:100%;margin:auto}.stage img{display:block;width:100%;height:auto}.stage .mask{position:absolute;inset:0;width:100%;height:auto;opacity:.45}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:14px;margin-top:14px}.swatch{display:inline-block;width:14px;height:14px;border:1px solid #56616e;vertical-align:middle;margin-right:5px}
.keep,.keep:hover{background:var(--keep);color:white}.reject,.reject:hover{background:var(--reject);color:white}.unsure,.unsure:hover{background:var(--unsure);color:white}.keep:hover,.reject:hover,.unsure:hover{filter:brightness(1.12)}
.filename{overflow-wrap:anywhere;font-size:17px}#message{min-height:26px;margin:12px 0;color:var(--accent)}#message.error{color:var(--reject)}textarea{width:100%;resize:vertical}.download{text-decoration:none;display:inline-block}
progress{width:100%;height:8px;accent-color:var(--accent)}details{margin-top:16px;color:var(--muted);overflow-wrap:anywhere}.empty{padding:40px;text-align:center}.muted{color:var(--muted)}
@media(max-width:750px){main{padding:12px}.compare{grid-template-columns:1fr}.viewport{height:85vw;max-height:560px}.toolbar{padding:12px}}
</style></head><body><main>
<div class="row spread"><div><h1>影像與舊標註篩選</h1><p>逐張檢查，留下值得使用的影像。</p></div><span id="counts" role="status"></span></div>
<progress id="progress" value="0" max="1" aria-label="已檢查進度"></progress>
<div class="toolbar row spread"><div class="row">
<label>篩選<select id="filter"><option value="all">全部</option><option value="unreviewed">未檢查</option><option value="keep">保留</option><option value="reject">排除</option><option value="unsure">待確認</option></select></label>
<label>檔名搜尋<input id="search" type="search" placeholder="來源或檔名" size="16"></label>
<label>跳至第<input id="jump" type="number" min="1" value="1">張</label><button id="go">前往</button>
</div><div class="row"><a class="download" href="/export/selected">下載保留名單</a><a class="download" href="/export/review">下載全部紀錄</a></div></div>
<div id="message" role="status" aria-live="polite"></div>
<div id="review"><div class="row spread"><h2 id="filename" class="filename"></h2><span id="position"></span></div>
<div class="toolbar row"><button id="previous">← 上一張</button><button class="keep" data-status="keep">保留 [1]</button><button class="reject" data-status="reject">排除 [2]</button><button class="unsure" data-status="unsure">待確認 [3]</button><button data-status="unreviewed">重設 [0]</button><button id="next">下一張 →</button></div>
<div class="compare"><section class="card"><h2>原圖</h2><div class="viewport" id="left"><div class="stage"><img id="original" alt="目前影像的原圖"></div></div></section>
<section class="card"><h2>舊標註 <span class="muted" id="viewTitle">／疊圖</span></h2><div class="viewport" id="right"><div class="stage"><img id="base" alt="舊標註下方的原圖"><img id="mask" class="mask" alt="目前影像的舊標註"></div></div></section></div>
<div class="legend">{% for name, color in colors.items() %}<span><i class="swatch" style="background:rgb({{color|join(',')}})"></i>{{labels.get(name,name)}}</span>{% endfor %}</div>
<div class="toolbar row"><label>標註顯示<select id="mode"><option value="overlay">半透明疊圖</option><option value="mask">純遮罩</option></select></label>
<label>透明度<input id="alpha" type="range" min="0" max="100" value="45"><span id="alphaValue">45%</span></label>
<label>放大<select id="zoom"><option value="1">完整影像</option><option value="2">2 倍</option><option value="4">4 倍</option></select></label>
<label><input id="advance" type="checkbox" checked>選擇後自動下一張</label></div>
<div class="card"><label for="note">備註（例如：邊界不準、漏標、影像清楚）</label><textarea id="note" rows="2" maxlength="4000"></textarea>
<button id="saveNote" style="margin-top:12px">儲存備註</button></div></div>
<div id="empty" class="empty" hidden>沒有符合條件的影像，請調整篩選或搜尋。</div>
<details><summary>資料來源與操作說明</summary><p>資料集：{{dataset}}<br>儲存位置：{{output}}<br>此處「舊標註」指上述資料夾的現有標註。選擇只會更新篩選紀錄，原圖與遮罩皆保留。<br>按 1 / 2 / 3 選擇；0 重設；左右鍵換圖。放大後兩側捲動同步。備註在換圖或選擇時儲存。<br>每次選擇自動寫入 review.json、review.csv、selected.csv；重新啟動自動載入。僅供單人本機使用。</p></details>
</main><script>
const token={{token|tojson}}, $=id=>document.getElementById(id), labels={{states|tojson}};
let items=[], visible=[], current=null, busy=false, ready=false, imageRatio=1;
function message(text,error=false){$('message').textContent=text;$('message').classList.toggle('error',error)}
function lock(value){busy=value;document.querySelectorAll('button').forEach(b=>b.disabled=value);['filter','search','jump'].forEach(id=>$(id).disabled=value)}
async function api(url,options){const response=await fetch(url,options);const data=await response.json();if(!response.ok)throw Error(data.error||'讀取失敗');return data}
function refreshList(){const query=$('search').value.toLowerCase(), filter=$('filter').value;visible=items.filter(x=>(filter==='all'||x.status===filter)&&x.name.toLowerCase().includes(query));const counts={unreviewed:0,keep:0,reject:0,unsure:0};items.forEach(x=>counts[x.status]++);$('counts').textContent=`${items.length} 張 · 保留 ${counts.keep} · 排除 ${counts.reject} · 待確認 ${counts.unsure} · 未檢查 ${counts.unreviewed}`;$('progress').max=items.length;$('progress').value=items.length-counts.unreviewed}
async function show(index){
  ready=false;current=index;const item=items[index];$('review').hidden=!item;$('empty').hidden=!!item;if(!item)return;
  $('filename').textContent=item.name;$('position').textContent=`第 ${index+1} / ${items.length} 張 · ${labels[item.status]} · 篩選 ${visible.length} 張`;$('jump').value=index+1;$('note').value=item.note;
  $('left').scrollTo(0,0);$('right').scrollTo(0,0);$('original').removeAttribute('src');$('base').removeAttribute('src');$('mask').removeAttribute('src');
  const image=new Image(), mask=new Image();image.src=`/image/${index}/original`;mask.src=`/image/${index}/mask`;
  await Promise.all([image.decode(),mask.decode()]);imageRatio=image.naturalWidth/image.naturalHeight;$('original').src=image.src;$('base').src=image.src;$('mask').src=mask.src;applyView();ready=true;
}
function applyView(){if(current===null)return;const pure=$('mode').value==='mask';$('mask').src=`/image/${current}/${pure?'mask':'overlay'}`;$('mask').style.opacity=pure?1:Number($('alpha').value)/100;$('base').style.visibility=pure?'hidden':'visible';$('viewTitle').textContent=pure?'／純遮罩':'／疊圖';$('alphaValue').textContent=$('alpha').value+'%';document.querySelectorAll('.viewport').forEach(view=>{const width=Math.min(view.clientWidth,view.clientHeight*imageRatio);view.querySelector('.stage').style.width=(width*Number($('zoom').value))+'px'})}
async function save(status){const item=items[current];const note=$('note').value;const data=await api('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify({index:current,status,note})});item.status=status;item.note=note;refreshList();message(data.warning||`已儲存：${item.name} → ${labels[status]}`,!!data.warning)}
async function saveDirty(){if(current!==null&&$('note').value!==items[current].note)await save(items[current].status)}
async function perform(action){if(busy)return;lock(true);try{await action()}catch(error){message(`操作失敗：${error.message}。請重試；未儲存內容會留在畫面。`,true)}finally{lock(false)}}
async function move(delta){await saveDirty();const position=visible.findIndex(x=>x.index===current), target=visible[position+delta];if(target)await show(target.index);else message(delta>0?'已到最後一張。':'已到第一張。')}
document.querySelectorAll('[data-status]').forEach(button=>button.onclick=()=>perform(async()=>{
  if(!ready)throw Error('請等待影像載入完成，或重新載入頁面');const old=visible.map(x=>x.index), position=old.indexOf(current);await save(button.dataset.status);
  const shouldAdvance=$('advance').checked || !visible.some(x=>x.index===current);
  if(shouldAdvance){const next=old.slice(position+1).find(i=>visible.some(x=>x.index===i));await show(next??(visible.some(x=>x.index===current)?current:(visible[0]?.index??null)));}else await show(current);
}));
$('previous').onclick=()=>perform(()=>move(-1));$('next').onclick=()=>perform(()=>move(1));$('saveNote').onclick=()=>perform(()=>save(items[current].status));
for(const id of ['filter','search'])$(id).onchange=()=>perform(async()=>{await saveDirty();refreshList();await show(visible[0]?.index??null)});
$('go').onclick=()=>perform(async()=>{const index=Number($('jump').value)-1;if(!Number.isInteger(index)||index<0||index>=items.length)throw Error('請輸入有效的影像編號');await saveDirty();$('filter').value='all';$('search').value='';refreshList();await show(index)});
['mode','alpha','zoom'].forEach(id=>$(id).oninput=applyView);
window.addEventListener('resize',applyView);
let scrolling=false;for(const [a,b] of [['left','right'],['right','left']])$(a).onscroll=()=>{if(scrolling)return;scrolling=true;$(b).scrollTop=$(a).scrollTop;$(b).scrollLeft=$(a).scrollLeft;requestAnimationFrame(()=>scrolling=false)};
document.addEventListener('keydown',event=>{if(event.ctrlKey||event.metaKey||event.altKey||event.shiftKey||event.repeat||['INPUT','TEXTAREA','SELECT','BUTTON','A'].includes(event.target.tagName)||busy||current===null)return;const keys={'1':'keep','2':'reject','3':'unsure','0':'unreviewed'};if(keys[event.key]){event.preventDefault();document.querySelector(`[data-status="${keys[event.key]}"]`).click()}else if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();perform(()=>move(event.key==='ArrowLeft'?-1:1))}});
window.addEventListener('beforeunload',event=>{if(busy||(current!==null&&$('note').value!==items[current].note)){event.preventDefault();event.returnValue=''}});
perform(async()=>{items=await api('/api/items');refreshList();$('jump').max=items.length;await show(items.find(x=>x.status==='unreviewed')?.index??0);message('已載入進度，可開始選擇。')});
</script></body></html>"""


def create_app(dataset: ReviewDataset, store: ReviewStore) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
    token = secrets.token_urlsafe(32)

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/")
    def home():
        return render_template_string(PAGE, colors=dataset.colors, labels=LABELS, states=STATES,
                                      dataset=dataset.root, output=store.output, token=token)

    @app.get("/api/items")
    def items():
        with store.lock:
            return jsonify([{"index": i, "name": item["name"], "status": store.records.get(item["name"], {}).get("status", "unreviewed"),
                             "note": store.records.get(item["name"], {}).get("note", "")} for i, item in enumerate(dataset.items)])

    @app.post("/api/review")
    def review():
        if not secrets.compare_digest(request.headers.get("X-Review-Token", ""), token):
            abort(403)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="Expected a JSON object"), 400
        index, status, note = data.get("index"), data.get("status"), data.get("note", "")
        if type(index) is not int or not 0 <= index < len(dataset.items):
            return jsonify(error="Invalid image index"), 400
        if not isinstance(status, str) or status not in STATES or not isinstance(note, str) or len(note) > 4000:
            return jsonify(error="Invalid status or note"), 400
        try:
            warning = store.save(index, status, note)
        except OSError as exc:
            return jsonify(error=f"儲存失敗：{exc}"), 500
        return jsonify(ok=True, warning=warning)

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

    @app.get("/export/<kind>")
    def export(kind: str):
        if kind not in {"selected", "review"}:
            abort(404)
        with store.lock:
            content = store.csv_text(store.records, kind == "selected")
        return send_file(io.BytesIO(content.encode("utf-8")), mimetype="text/csv; charset=utf-8",
                         as_attachment=True, download_name=f"{kind}.csv")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="逐張查看原圖／舊標註，篩選並儲存名單。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="CVAT JPEGImages/SegmentationClass 或 images/masks + manifest.json 的根目錄")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="篩選紀錄存放目錄")
    parser.add_argument("--host", default="127.0.0.1", help="預設僅本機；遠端可透過 SSH port forwarding 使用")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--check", action="store_true", help="檢查每張影像／遮罩配對、尺寸、類別後退出，不啟動網站")
    args = parser.parse_args()
    dataset = ReviewDataset(args.dataset)
    if args.check:
        for index in range(len(dataset.items)):
            dataset.load_pair(index)
        print(f"OK: {len(dataset.items)} image/mask pairs; dimensions and label values verified.")
        return
    # A single writer prevents two tool instances from overwriting each other's choices.
    import fcntl

    store = ReviewStore(dataset, args.output)
    store.output.mkdir(parents=True, exist_ok=True)
    with (store.output / ".review.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This output directory is already open in another review process.")
        # Re-read after acquiring the process lock.
        store = ReviewStore(dataset, args.output)
        print(f"Loaded {len(dataset.items)} pairs. Review: http://{args.host}:{args.port}", flush=True)
        print(f"Selections: {store.output}", flush=True)
        create_app(dataset, store).run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
