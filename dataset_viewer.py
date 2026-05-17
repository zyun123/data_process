#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MUGE Dataset Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d8dee8;
      --text: #18202b;
      --muted: #667085;
      --accent: #1769aa;
      --accent-soft: #e8f2fb;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      gap: 16px;
    }
    header h1 {
      margin: 0;
      font-size: 17px;
      font-weight: 650;
      white-space: nowrap;
    }
    header code {
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    main {
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: calc(100vh - 56px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
    }
    .content {
      padding: 20px;
      min-width: 0;
    }
    label {
      display: block;
      margin: 14px 0 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    select, input, button {
      width: 100%;
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    input { padding: 0 10px; }
    button {
      margin-top: 10px;
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font-weight: 650;
    }
    button.secondary {
      background: #fff;
      color: var(--accent);
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .status {
      margin-top: 14px;
      min-height: 20px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .status.error { color: var(--danger); }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric, .record, .image-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    .metric {
      padding: 12px;
    }
    .metric strong {
      display: block;
      font-size: 22px;
      line-height: 1.1;
    }
    .metric span {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    .record {
      padding: 14px;
      margin-bottom: 16px;
    }
    .record h2 {
      margin: 0 0 10px;
      font-size: 15px;
    }
    .text {
      line-height: 1.7;
      white-space: pre-wrap;
      word-break: break-word;
    }
    textarea.editor {
      width: 100%;
      min-height: 88px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      line-height: 1.6;
      color: var(--text);
      background: #fff;
    }
    .record-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .record-actions button {
      margin: 0;
      width: auto;
      height: 30px;
      padding: 0 10px;
      font-size: 12px;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .chip {
      border: 1px solid #b8d7ef;
      background: var(--accent-soft);
      color: #164b76;
      border-radius: 999px;
      padding: 4px 9px;
      cursor: pointer;
      font-size: 12px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 14px;
    }
    .image-card {
      overflow: hidden;
    }
    .image-card img {
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #eef1f5;
    }
    .image-meta {
      padding: 10px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .image-meta button {
      margin: 0;
      width: auto;
      height: 28px;
      padding: 0 9px;
      font-size: 12px;
    }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .summary { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>MUGE Dataset Viewer</h1>
    <code id="datasetPath"></code>
  </header>
  <main>
    <aside>
      <label for="split">Split</label>
      <select id="split">
        <option value="train">train</option>
        <option value="valid">valid</option>
      </select>

      <label for="textId">按 text_id 查询</label>
      <input id="textId" inputmode="numeric" placeholder="例如 0">
      <button id="findText">显示文本和图片</button>

      <label for="imageId">按 image_id 查询</label>
      <input id="imageId" inputmode="numeric" placeholder="例如 1505">
      <button id="findImage">显示图片</button>

      <button id="randomText" class="secondary">随机 text</button>
      <button id="randomImage" class="secondary">随机 image</button>
      <div id="status" class="status"></div>
    </aside>
    <section class="content">
      <div id="summary" class="summary"></div>
      <div id="record"></div>
      <div id="images" class="grid"></div>
    </section>
  </main>
  <script>
    const state = { summary: null };
    const el = (id) => document.getElementById(id);

    function setStatus(text, error=false) {
      const s = el('status');
      s.textContent = text || '';
      s.className = error ? 'status error' : 'status';
    }

    async function api(path, params={}) {
      const url = new URL(path, window.location.origin);
      for (const [key, value] of Object.entries(params)) url.searchParams.set(key, value);
      const res = await fetch(url);
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || res.statusText);
      return data;
    }

    async function apiPost(path, body={}) {
      const res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || res.statusText);
      return data;
    }

    function split() { return el('split').value; }

    function renderSummary(data) {
      state.summary = data;
      el('datasetPath').textContent = data.dataset_dir;
      const parts = ['train', 'valid'].map((name) => {
        const item = data.splits[name] || { texts: 0, images: 0 };
        return `
          <div class="metric">
            <strong>${item.texts}</strong>
            <span>${name} texts</span>
          </div>
          <div class="metric">
            <strong>${item.images}</strong>
            <span>${name} images</span>
          </div>`;
      }).join('');
      el('summary').innerHTML = parts;
    }

    function renderRecord(item, title) {
      if (!item) {
        el('record').innerHTML = '';
        return;
      }
      const chips = (item.image_ids || []).map((id) =>
        `<button class="chip" data-image-id="${id}">image_id ${id}</button>`
      ).join('');
      el('record').innerHTML = `
        <div class="record">
          <h2>${title}</h2>
          ${editableTextHtml(item)}
          <div class="chips">${chips}</div>
        </div>`;
      bindEditButtons();
      document.querySelectorAll('[data-image-id]').forEach((btn) => {
        btn.addEventListener('click', () => lookupImage(btn.dataset.imageId));
      });
    }

    function renderImages(images, refs=[]) {
      el('images').innerHTML = images.map((img) => `
        <div class="image-card">
          <img alt="image_id ${img.image_id}" src="${img.src}">
          <div class="image-meta">
            <span>image_id ${img.image_id}</span>
            <button data-jump-image="${img.image_id}" class="secondary">查看</button>
          </div>
        </div>
      `).join('');
      document.querySelectorAll('[data-jump-image]').forEach((btn) => {
        btn.addEventListener('click', () => lookupImage(btn.dataset.jumpImage));
      });
      if (refs.length) {
        const refHtml = refs.map((r) => `
          <div class="record">
            <h2>text_id ${r.text_id}</h2>
            ${editableTextHtml(r)}
            <div class="chips">
              <button class="chip" data-text-id="${r.text_id}">查看这条 text 的全部图片</button>
            </div>
          </div>
        `).join('');
        el('record').insertAdjacentHTML('beforeend', `
          <div class="record">
            <h2>引用这个 image 的文本</h2>
          </div>`);
        el('record').insertAdjacentHTML('beforeend', refHtml);
        bindEditButtons();
        document.querySelectorAll('[data-text-id]').forEach((btn) => {
          btn.addEventListener('click', () => lookupText(btn.dataset.textId));
        });
      }
    }

    function editableTextHtml(item) {
      return `
        <textarea class="editor" data-edit-text-id="${item.text_id}">${escapeHtml(item.text || '')}</textarea>
        <div class="record-actions">
          <button data-save-text-id="${item.text_id}">保存修改</button>
          <button class="secondary" data-reset-text-id="${item.text_id}">撤销未保存</button>
        </div>`;
    }

    function bindEditButtons() {
      document.querySelectorAll('[data-save-text-id]').forEach((btn) => {
        btn.addEventListener('click', () => saveText(btn.dataset.saveTextId));
      });
      document.querySelectorAll('[data-reset-text-id]').forEach((btn) => {
        btn.addEventListener('click', () => lookupText(btn.dataset.resetTextId));
      });
    }

    async function saveText(textId) {
      const editor = document.querySelector(`[data-edit-text-id="${textId}"]`);
      if (!editor) return;
      try {
        setStatus('保存中...');
        const data = await apiPost('/api/text', {
          split: split(),
          text_id: Number(textId),
          text: editor.value,
        });
        setStatus(`已保存 text_id ${data.text.text_id}，备份文件：${data.backup}`);
        await lookupText(textId);
      } catch (err) {
        setStatus(err.message, true);
      }
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[c]));
    }

    async function lookupText(id) {
      try {
        setStatus('加载中...');
        const data = await api('/api/text', { split: split(), text_id: id });
        el('textId').value = id;
        renderRecord(data.text, `text_id ${data.text.text_id}`);
        renderImages(data.images);
        setStatus(`找到 ${data.images.length} 张图片`);
      } catch (err) {
        setStatus(err.message, true);
      }
    }

    async function lookupImage(id) {
      try {
        setStatus('加载中...');
        const data = await api('/api/image', { split: split(), image_id: id });
        el('imageId').value = id;
        renderRecord(null);
        renderImages([data.image], data.texts);
        setStatus(`找到 image_id ${id}，引用文本 ${data.texts.length} 条`);
      } catch (err) {
        setStatus(err.message, true);
      }
    }

    function randomFrom(min, max) {
      return Math.floor(Math.random() * (max - min + 1)) + min;
    }

    async function loadSummary() {
      try {
        renderSummary(await api('/api/summary'));
      } catch (err) {
        setStatus(err.message, true);
      }
    }

    el('findText').addEventListener('click', () => lookupText(el('textId').value.trim()));
    el('findImage').addEventListener('click', () => lookupImage(el('imageId').value.trim()));
    el('textId').addEventListener('keydown', (e) => { if (e.key === 'Enter') lookupText(el('textId').value.trim()); });
    el('imageId').addEventListener('keydown', (e) => { if (e.key === 'Enter') lookupImage(el('imageId').value.trim()); });
    el('randomText').addEventListener('click', () => {
      const s = state.summary?.splits?.[split()];
      if (s?.text_ids?.length) lookupText(s.text_ids[randomFrom(0, s.text_ids.length - 1)]);
    });
    el('randomImage').addEventListener('click', () => {
      const s = state.summary?.splits?.[split()];
      if (s?.image_ids?.length) lookupImage(s.image_ids[randomFrom(0, s.image_ids.length - 1)]);
    });
    el('split').addEventListener('change', () => {
      el('record').innerHTML = '';
      el('images').innerHTML = '';
      setStatus('');
    });
    loadSummary();
  </script>
</body>
</html>
"""


class DatasetIndex:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.lock = threading.Lock()
        self.splits = {}
        for split in ("train", "valid"):
            texts_path = dataset_dir / f"{split}_texts.jsonl"
            imgs_path = dataset_dir / f"{split}_imgs.tsv"
            if texts_path.is_file() and imgs_path.is_file():
                self.splits[split] = self._load_split(texts_path, imgs_path)
        if not self.splits:
            raise FileNotFoundError(f"No train/valid MUGE files found in {dataset_dir}")

    def _load_split(self, texts_path: Path, imgs_path: Path) -> dict:
        images = {}
        with imgs_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid TSV row in {imgs_path}:{line_no}")
                images[int(parts[0])] = parts[1]

        texts = {}
        rows = []
        image_to_texts = {}
        with texts_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text_id = int(item["text_id"])
                item["text_id"] = text_id
                item["image_ids"] = [int(image_id) for image_id in item.get("image_ids", [])]
                rows.append(item)
                texts[text_id] = item
                for image_id in item["image_ids"]:
                    image_to_texts.setdefault(image_id, []).append(item)

        return {
            "texts_path": texts_path,
            "texts": texts,
            "text_rows": rows,
            "images": images,
            "image_to_texts": image_to_texts,
        }

    def summary(self) -> dict:
        splits = {}
        for name, data in self.splits.items():
            splits[name] = {
                "texts": len(data["texts"]),
                "images": len(data["images"]),
                "text_ids": sorted(data["texts"].keys()),
                "image_ids": sorted(data["images"].keys()),
            }
        return {"dataset_dir": str(self.dataset_dir), "splits": splits}

    def image_src(self, split: str, image_id: int) -> str:
        data = self._split(split)
        b64 = data["images"].get(image_id)
        if b64 is None:
            raise KeyError(f"image_id {image_id} not found in {split}_imgs.tsv")
        mime = guess_image_mime(b64)
        return f"data:{mime};base64,{b64}"

    def get_text(self, split: str, text_id: int) -> dict:
        data = self._split(split)
        item = data["texts"].get(text_id)
        if item is None:
            raise KeyError(f"text_id {text_id} not found in {split}_texts.jsonl")
        images = [{"image_id": image_id, "src": self.image_src(split, image_id)} for image_id in item["image_ids"]]
        return {"text": item, "images": images}

    def get_image(self, split: str, image_id: int) -> dict:
        data = self._split(split)
        src = self.image_src(split, image_id)
        refs = data["image_to_texts"].get(image_id, [])
        return {
            "image": {"image_id": image_id, "src": src},
            "texts": [{"text_id": item["text_id"], "text": item.get("text", "")} for item in refs],
        }

    def update_text(self, split: str, text_id: int, text: str) -> dict:
        text = str(text).strip()
        if not text:
            raise ValueError("text cannot be empty")

        with self.lock:
            data = self._split(split)
            item = data["texts"].get(text_id)
            if item is None:
                raise KeyError(f"text_id {text_id} not found in {split}_texts.jsonl")

            item["text"] = text
            texts_path = data["texts_path"]
            backup_path = texts_path.with_name(texts_path.name + ".bak")
            if not backup_path.exists():
                shutil.copy2(texts_path, backup_path)

            tmp_path = texts_path.with_name(texts_path.name + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                for row in data["text_rows"]:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmp_path.replace(texts_path)

        return {"text": item, "backup": str(backup_path)}

    def _split(self, split: str) -> dict:
        if split not in self.splits:
            raise KeyError(f"split {split!r} not available")
        return self.splits[split]


def guess_image_mime(b64: str) -> str:
    try:
        header = base64.b64decode(b64[:64], validate=False)
    except Exception:
        return "image/jpeg"
    if header.startswith(b"\x89PNG"):
        return "image/png"
    if header.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if header.startswith(b"RIFF") and b"WEBP" in header[:16]:
        return "image/webp"
    if header.startswith(b"BM"):
        return "image/bmp"
    return mimetypes.types_map.get(".jpg", "image/jpeg")


def parse_int(params: dict, key: str) -> int:
    values = params.get(key)
    if not values or values[0] == "":
        raise ValueError(f"Missing {key}")
    return int(values[0])


class Handler(BaseHTTPRequestHandler):
    index: DatasetIndex

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_html(HTML)
            elif parsed.path == "/api/summary":
                self._send_json(self.index.summary())
            elif parsed.path == "/api/text":
                params = parse_qs(parsed.query)
                split = params.get("split", ["train"])[0]
                self._send_json(self.index.get_text(split, parse_int(params, "text_id")))
            elif parsed.path == "/api/image":
                params = parse_qs(parsed.query)
                split = params.get("split", ["train"])[0]
                self._send_json(self.index.get_image(split, parse_int(params, "image_id")))
            else:
                self._send_json({"error": "not found"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path != "/api/text":
                self._send_json({"error": "not found"}, status=404)
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            split = payload.get("split", "train")
            text_id = int(payload["text_id"])
            text = payload.get("text", "")
            self._send_json(self.index.update_text(split, text_id, text))
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, fmt, *args):
        return

    def _send_html(self, body: str):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, body: dict, status: int = 200):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse MUGE train/valid JSONL and TSV images in a local web UI.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/2026-05-17/cvat_merge_pos_datasets_expanded"),
        help="Directory containing train_texts.jsonl/train_imgs.tsv and valid files.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    Handler.index = DatasetIndex(dataset_dir)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Dataset: {dataset_dir}")
    print(f"Viewer: {url}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
