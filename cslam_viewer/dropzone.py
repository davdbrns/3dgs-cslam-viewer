"""Full-window drag and drop for the viser page.

viser has no drop support, so we inject a small script into its page and receive the dropped
files on a separate localhost HTTP server. Browsers never expose local paths, which is why the
files are uploaded (copied into a temp folder) rather than opened in place.
"""

import atexit
import re
import shutil
import tempfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import viser
from viser import _messages

_JS = r"""
(() => {
  if (window.__cslamDrop) return;
  window.__cslamDrop = true;
  const base = `${location.protocol}//${location.hostname}:__PORT__`;
  const accepted = /\.(ply|npz|json)$/i;

  const overlay = document.createElement("div");
  overlay.style.cssText = "position:fixed;inset:12px;z-index:100000;display:none;" +
    "align-items:center;justify-content:center;text-align:center;border-radius:12px;" +
    "border:3px dashed rgba(255,255,255,0.9);background:rgba(15,20,30,0.6);" +
    "color:#fff;font:600 20px system-ui,sans-serif;pointer-events:none;white-space:pre-line";
  document.body.appendChild(overlay);
  const show = (text) => { overlay.textContent = text; overlay.style.display = "flex"; };
  const hide = () => { overlay.style.display = "none"; };

  let depth = 0, busy = false;
  const hasFiles = (e) => e.dataTransfer && Array.from(e.dataTransfer.types).includes("Files");
  window.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); depth++;
    if (!busy) show("Drop maps or run folders\n(.ply, trajectories.npz, loops.npz, manifest.json)");
  }, true);
  window.addEventListener("dragleave", (e) => {
    if (!hasFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (!depth && !busy) hide();
  }, true);
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); }, true);

  const readEntries = (reader) => new Promise((resolve, reject) => {
    const out = [];
    const next = () => reader.readEntries((batch) => {
      if (!batch.length) resolve(out); else { out.push(...batch); next(); }
    }, reject);
    next();
  });
  async function walk(entry, prefix, out) {
    if (entry.isFile) {
      if (accepted.test(entry.name)) out.push([prefix + entry.name, await new Promise((r, j) => entry.file(r, j))]);
    } else if (entry.isDirectory) {
      for (const child of await readEntries(entry.createReader())) await walk(child, prefix + entry.name + "/", out);
    }
  }

  window.addEventListener("drop", async (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); e.stopPropagation(); depth = 0;
    if (busy) return;
    // Entries must be taken synchronously; the DataTransfer is cleared after the handler yields.
    const entries = Array.from(e.dataTransfer.items).map((i) => i.webkitGetAsEntry()).filter(Boolean);
    const plainFiles = Array.from(e.dataTransfer.files);
    busy = true;
    try {
      const files = [];
      for (const entry of entries) await walk(entry, "", files);
      // Some drag sources provide no file-system entries; plain files still work, folders do not.
      if (!entries.length) for (const f of plainFiles) if (accepted.test(f.name)) files.push([f.name, f]);
      if (!files.length) throw new Error("No .ply / .npz / .json files found");
      const batch = Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
      const total = files.reduce((s, [, f]) => s + f.size, 0);
      let sent = 0;
      for (const [i, [path, file]] of files.entries()) {
        show(`Uploading ${i + 1}/${files.length}: ${path}\n${(sent / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`);
        const res = await fetch(`${base}/upload?batch=${batch}&path=${encodeURIComponent(path)}`, { method: "PUT", body: file });
        if (!res.ok) throw new Error(await res.text());
        sent += file.size;
      }
      const res = await fetch(`${base}/done?batch=${batch}`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      hide();
    } catch (err) {
      show("Upload failed:\n" + err.message);
      setTimeout(hide, 4000);
    } finally {
      busy = false;
    }
  }, true);
})();
"""

_BATCH_RE = re.compile(r"^[a-z0-9]{4,32}$")


def install(server: viser.ViserServer, host: str, port: int, on_drop: Callable[[Path], None]) -> None:
    """Starts the upload server and injects the drop script. `on_drop` receives the folder holding
    one drop's files, with their relative folder structure preserved."""
    upload_root = Path(tempfile.mkdtemp(prefix="cslam-viewer-"))
    atexit.register(shutil.rmtree, upload_root, ignore_errors=True)

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, text: str = "") -> None:
            body = text.encode()
            self.send_response(code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "PUT, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _batch_dir(self, query: dict) -> Path | None:
            batch = query.get("batch", [""])[0]
            return upload_root / batch if _BATCH_RE.match(batch) else None

        def do_OPTIONS(self) -> None:
            self._reply(204)

        def do_PUT(self) -> None:
            url = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(url.query)
            batch_dir = self._batch_dir(query)
            rel = Path(query.get("path", [""])[0])
            if url.path != "/upload" or batch_dir is None or not rel.parts or rel.is_absolute() or ".." in rel.parts:
                return self._reply(400, "bad upload request")
            dest = batch_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            remaining = int(self.headers.get("Content-Length", 0))
            with open(dest, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 8 << 20))
                    if not chunk:
                        return self._reply(400, "upload interrupted")
                    f.write(chunk)
                    remaining -= len(chunk)
            self._reply(200)

        def do_POST(self) -> None:
            url = urllib.parse.urlparse(self.path)
            batch_dir = self._batch_dir(urllib.parse.parse_qs(url.query))
            if url.path != "/done" or batch_dir is None or not batch_dir.is_dir():
                return self._reply(400, "unknown upload batch")
            threading.Thread(target=on_drop, args=(batch_dir,), daemon=True).start()
            self._reply(200)

        def log_message(self, *args) -> None:
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # Same mechanism viser uses to ship plotly.js; it is replayed to clients that connect later.
    server.gui._websock_interface.queue_message(
        _messages.RunJavascriptMessage(source=_JS.replace("__PORT__", str(port)))
    )
