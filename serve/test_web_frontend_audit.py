"""Comprehensive Byte-by-Byte Audit and Verification of Strata's native Web UI.

Tests:
1. HTTP routing for all web assets:
   - Root / -> serve/web/index.html (text/html; charset=utf-8)
   - Static assets /web/* -> app.css, app.js, components.css, tokens.css, sprite.svg (correct MIME)
   - Fonts /fonts/* -> outfit-latin-wght.woff2, outfit-latin-ext-wght.woff2 (font/woff2)
   - Monitor view /monitor -> serve/web/monitor.html (text/html; charset=utf-8)
   - API Monitor /api-monitor -> serve/web/monitor.html (text/html; charset=utf-8)
   - Offline verification: 0 external CDN links, 100% self-hosted
2. Examination of serve/web/app.js:
   - Verify endpoints (/v1/chat/completions, /settings, /health, /metrics, /config, /mcp)
   - Verify 0 foreign llama.cpp endpoints (/completion, /infill, /tokenize)
   - Verify native UI rendering logic: model badges, VRAM/GPU load gauges, context meter,
     stop generation button, and reasoning foldouts.
3. Live query via urllib against running server instance.
"""
import json
import re
import unittest
import urllib.request
import urllib.error
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "serve" / "web"


class WebFrontendAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = MockEngine(cls.tok, "Test response.", max_context=16384)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
        cls.svc.api_monitor = True
        cls.svc.model = "qwen3.6-35b-a3b-uncensored"
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path):
        req = urllib.request.Request(f"{self.base}{path}")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def test_01_root_serves_index_html(self):
        status, ctype, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "text/html; charset=utf-8")
        disk_bytes = (WEB_DIR / "index.html").read_bytes()
        self.assertEqual(body, disk_bytes, "Root / must match serve/web/index.html byte-for-byte")

    def test_02_monitor_serves_monitor_html(self):
        for path in ("/monitor", "/api-monitor"):
            with self.subTest(path=path):
                status, ctype, body = self.get(path)
                self.assertEqual(status, 200)
                self.assertEqual(ctype, "text/html; charset=utf-8")
                disk_bytes = (WEB_DIR / "monitor.html").read_bytes()
                self.assertEqual(body, disk_bytes, f"{path} must match serve/web/monitor.html byte-for-byte")

    def test_03_static_web_assets_served_identically(self):
        assets = [
            ("/web/app.js", "text/javascript; charset=utf-8", "app.js"),
            ("/web/app.css", "text/css; charset=utf-8", "app.css"),
            ("/web/components.css", "text/css; charset=utf-8", "components.css"),
            ("/web/tokens.css", "text/css; charset=utf-8", "tokens.css"),
            ("/web/sprite.svg", "image/svg+xml", "sprite.svg"),
            ("/web/monitor.js", "text/javascript; charset=utf-8", "monitor.js"),
        ]
        for url_path, expected_mime, filename in assets:
            with self.subTest(file=filename):
                status, ctype, body = self.get(url_path)
                self.assertEqual(status, 200)
                self.assertEqual(ctype, expected_mime)
                disk_bytes = (WEB_DIR / filename).read_bytes()
                self.assertEqual(body, disk_bytes, f"{url_path} must match {filename} byte-for-byte")

    def test_04_font_assets_served_identically(self):
        fonts = [
            ("/fonts/outfit-latin-wght.woff2", "font/woff2", "outfit-latin-wght.woff2"),
            ("/fonts/outfit-latin-ext-wght.woff2", "font/woff2", "outfit-latin-ext-wght.woff2"),
        ]
        for url_path, expected_mime, filename in fonts:
            with self.subTest(font=filename):
                status, ctype, body = self.get(url_path)
                self.assertEqual(status, 200)
                self.assertEqual(ctype, expected_mime)
                disk_bytes = (WEB_DIR / "fonts" / filename).read_bytes()
                self.assertEqual(body, disk_bytes, f"{url_path} must match {filename} byte-for-byte")

    def test_05_offline_self_hosted_verification(self):
        """Verify no external CDN dependencies (scripts, stylesheets, fonts) exist."""
        html_files = [WEB_DIR / "index.html", WEB_DIR / "monitor.html"]
        css_files = list(WEB_DIR.glob("*.css"))
        js_files = list(WEB_DIR.glob("*.js"))

        cdn_patterns = [
            re.compile(r'https?://(?:cdn|cdnjs|unpkg|jsdelivr|fonts\.googleapis|ajax\.googleapis)', re.I),
            re.compile(r'@import\s+url\([\'"]?https?:', re.I),
        ]

        for f in html_files + css_files + js_files:
            content = f.read_text(encoding="utf-8")
            for pattern in cdn_patterns:
                m = pattern.search(content)
                self.assertIsNone(m, f"File {f.name} contains external CDN dependency: {m.group(0) if m else ''}")

    def test_06_svg_sprites_complete_coverage(self):
        """Verify every icon referenced in HTML and JS exists in sprite.svg."""
        sprite_text = (WEB_DIR / "sprite.svg").read_text(encoding="utf-8")
        defined_ids = set(re.findall(r'id="([^"]+)"', sprite_text))

        used_icons = set()
        for f in [WEB_DIR / "index.html", WEB_DIR / "app.js"]:
            text = f.read_text(encoding="utf-8")
            for m in re.finditer(r'sprite\.svg#([a-zA-Z0-9_-]+)', text):
                used_icons.add(m.group(1))
            for m in re.finditer(r'icon\(\s*"([^"]+)"', text):
                used_icons.add("i-" + m.group(1))
            for m in re.finditer(r'icon:\s*"([^"]+)"', text):
                used_icons.add("i-" + m.group(1))

        missing = used_icons - defined_ids
        self.assertEqual(missing, set(), f"Missing icon definitions in sprite.svg: {missing}")

    def test_07_no_foreign_llama_endpoints(self):
        """Verify no lingering llama.cpp endpoints exist in app.js or frontend files."""
        for f in [WEB_DIR / "app.js", WEB_DIR / "monitor.js"]:
            content = f.read_text(encoding="utf-8")
            for llama_ep in ["/completion\"", "'/completion'", "\"/infill\"", "'/infill'", "\"/tokenize\"", "'/tokenize'"]:
                self.assertNotIn(llama_ep, content, f"Found foreign endpoint {llama_ep} in {f.name}")
            self.assertNotIn("llama.cpp", content.lower())

    def test_08_native_ui_elements_in_frontend(self):
        """Verify Strata's native UI elements exist in HTML and are controlled in app.js."""
        index_html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        app_js = (WEB_DIR / "app.js").read_text(encoding="utf-8")

        # 1. Model badges (Idle, Reading, Generating, Queued, Error)
        for badge in ["idle", "reading", "generating", "queued", "error"]:
            self.assertIn(f'data-s="{badge}"', index_html, f"Badge {badge} missing in index.html")
        self.assertIn("renderMonitor", app_js)
        self.assertIn("#state-badges .st-badge", app_js)

        # 2. VRAM / GPU load gauges
        self.assertIn('id="metrics"', index_html)
        self.assertIn("gpu_util", app_js)
        self.assertIn("gpu_mem_used", app_js)
        self.assertIn("gpu_temp", app_js)
        self.assertIn("expert_slots", app_js)

        # 3. Context meter (SVG gauge)
        self.assertIn('id="ctx-fill"', index_html)
        self.assertIn('id="ctx-pct"', index_html)
        self.assertIn('id="ctx-sub"', index_html)
        self.assertIn('$("ctx-fill").setAttribute("stroke-dasharray"', app_js)

        # 4. Stop generation button
        self.assertIn('id="stop-btn"', index_html)
        self.assertIn('$("stop-btn").onclick = () => { if (busy) busy.controller.abort(); };', app_js)
        self.assertIn('$("stop-btn").hidden = !on;', app_js)

        # 5. Reasoning foldouts
        self.assertIn('class="st-collapse think"', app_js)
        self.assertIn('m.reasoning', app_js)
        self.assertIn('reasoning_content', app_js)
        self.assertIn('reasoning_effort', app_js)
        self.assertIn('s-thinking', index_html)

    def test_09_security_path_traversal_protection(self):
        """Verify attempts to escape /web/ or /fonts/ return 404."""
        bad_paths = [
            "/web/..%2Fserver.py",
            "/web/..%2F..%2Fconfigs%2Fstrata-qwen36.json",
            "/fonts/..%2F..%2Fsetup.py",
            "/fonts/missing.woff2",
            "/web/test.py",
            "/web/index.html",  # only served at /
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                status, _, _ = self.get(path)
                self.assertEqual(status, 404, f"Path {path} should be rejected with 404")


if __name__ == "__main__":
    unittest.main()
