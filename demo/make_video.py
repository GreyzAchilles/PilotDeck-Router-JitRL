"""Produce the ~2-minute demo video (demo/demo-video.mp4).

Pipeline (Windows, stdlib + ffmpeg + headless Edge):
  1. Render narrative cards (HTML -> PNG) with headless Edge;
  2. Screenshot the REAL demo UI (offline mode) with headless Edge,
     including the #live tab deep link;
  3. Rasterize the technical diagrams (SVG -> PNG) with headless Edge;
  4. Assemble all frames with durations + fades via ffmpeg concat.

Run:
    python demo/make_video.py            # assumes demo server NOT required:
                                        # it starts its own in-process server

Requirements on PATH: ffmpeg; Edge at the standard location.
Output: demo/demo-video.mp4 (H.264 yuv420p, 1600x900), plus frames under
demo/.video/ (gitignored build artifacts).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.server import DemoApp, DemoHTTPServer  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / ".video"
W, H = 1600, 900
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
FFMPEG = shutil.which("ffmpeg") or r"D:\DemoMake\AppDownload\ffmpeg-2026-04-26-git-4867d251ad-essentials_build\ffmpeg-2026-04-26-git-4867d251ad-essentials_build\bin\ffmpeg.exe"

FONT_CSS = """
body{margin:0;font-family:'Segoe UI','Microsoft YaHei',sans-serif;
background:linear-gradient(135deg,#0f172a 0%,#1e293b 60%,#0f172a 100%);
color:#f8fafc;display:flex;align-items:center;justify-content:center;height:100vh}
.card{max-width:1200px;padding:60px 80px}
.kicker{color:#34d399;font-size:26px;letter-spacing:6px;margin-bottom:24px}
h1{font-size:64px;margin:0 0 28px;line-height:1.25}
h2{font-size:44px;margin:0 0 24px;color:#e2e8f0}
p{font-size:30px;line-height:1.7;color:#cbd5e1;margin:0}
.big{color:#34d399;font-weight:700}
.bad{color:#f87171;font-weight:700}
table{border-collapse:collapse;font-size:28px;margin-top:20px}
td,th{border:1px solid #475569;padding:10px 22px;text-align:left}
th{color:#94a3b8}
.url{margin-top:36px;color:#64748b;font-size:24px}
"""

CARDS: dict[str, str] = {
    "card_title": """
    <div class="card"><div class="kicker">PILOTDECK ROUTER × JITRL</div>
    <h1>让路由器跨轮学习</h1>
    <p>填补 TokenSaver Judge 的<span class="big">跨轮经验真空</span>：
    冻结本地 Judge · 非参数记忆 · 闭式 logit 更新</p>
    <p class="url">z′(tier) = z(tier) + β·Â(tier)</p></div>""",
    "card_c0": """
    <div class="card"><div class="kicker">C0 · 有害 FLIPS</div>
    <h2>小邻域探索 bonus 的代价</h2>
    <table><tr><th>任务</th><th>翻转</th><th>成本</th><th>质量</th></tr>
    <tr><td>T06 code_gen</td><td>simple → complex</td><td class="bad">+$0.004045</td><td>1 → 1</td></tr>
    <tr><td>T19 refactor</td><td>simple → complex</td><td class="bad">+$0.004194</td><td>1 → 1</td></tr></table>
    <p>两次翻转合计 <span class="bad">+$0.008239</span>，质量增益为 0 —— C0 比 B 贵 3.78% 的主因</p></div>""",
    "card_c1": """
    <div class="card"><div class="kicker">C1 · 记忆级联</div>
    <h2>局部门控为何仍失败</h2>
    <p>min_neighbors=3 门控<span class="big">抑制了</span> T06/T19 的翻转</p>
    <p>但被门控的决策仍按 base tier 写回记忆 → 邻域优势改变</p>
    <table><tr><th>T20</th><th>结果</th></tr>
    <tr><td>simple → complex（新 flip）</td><td class="bad">+$0.004190，质量 1 → 1</td></tr></table>
    <p>问题被<span class="bad">移动</span>（T06/T19 → T20），没有被解决 · P1–P5 = 3/5</p></div>""",
    "card_t8": """
    <div class="card"><div class="kicker">T8 消融 · LOGIT 调制 vs PROMPT 注入</div>
    <h2>方向性偏向 prompt injection——但要说清楚为什么</h2>
    <table><tr><th></th><th>T8-LM</th><th>T8-PM</th></tr>
    <tr><td>质量</td><td>2.5417</td><td>2.5417</td></tr>
    <tr><td>执行成本</td><td>$0.032653</td><td class="big">$0.027993</td></tr>
    <tr><td>记忆改变决策</td><td>1（T19 有害 flip）</td><td class="bad">0 / 6 次注入</td></tr></table>
    <p>PM 更便宜是因为它<span class="bad">什么都没改变</span>；成本差 ~90% 来自 LM 的 T19。
    当前 1B Judge 封顶了两种机制的上限</p></div>""",
    "card_privacy": """
    <div class="card"><div class="kicker">现场可靠性</div>
    <h2>隐私与降级</h2>
    <p><span class="big">默认离线</span>——实验面板零网络可用</p>
    <p>会话记忆仅驻内存 · 观众输入不落盘 · <span class="big">API key 永不出后端</span></p>
    <p>在线失败 → 结构化降级横幅，永不白屏</p></div>""",
    "card_end": """
    <div class="card"><div class="kicker">诚实负结果也是结果</div>
    <h2>端到端可行 · 开销近零 · 成本目标未达</h2>
    <p>C0 D10 = 3/4 · C1 P1–P5 = 3/5 · T8 弱 Judge 封顶——全部有机理分析</p>
    <p>代码 · 冻结数据 · 预注册文档 全部开源</p>
    <p class="url">github.com/GreyzAchilles/PilotDeck-Router-JitRL</p></div>""",
}


def card_html(body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{FONT_CSS}</style></head><body>{body}</body></html>")


def edge_shot(src: str, out: Path, width: int = W, height: int = H,
              extra: list[str] | None = None) -> None:
    """Headless Edge screenshot of a URL or file path."""
    if src.startswith("http"):
        target = src
    else:
        target = Path(src).resolve().as_uri()
    cmd = [str(EDGE), "--headless=new", "--disable-gpu",
           f"--window-size={width},{height}", "--force-device-scale-factor=1",
           "--hide-scrollbars", f"--screenshot={out.resolve()}",
           "--virtual-time-budget=9000", target]
    if extra:
        cmd[1:1] = extra
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    if not out.exists():
        raise RuntimeError(f"edge did not write {out}")


def main() -> int:
    if not EDGE.exists():
        print("[video] Edge not found"); return 1
    if not Path(FFMPEG).exists():
        print("[video] ffmpeg not found"); return 1
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "cards").mkdir(exist_ok=True)

    # ---- 1. narrative cards
    for name, body in CARDS.items():
        html = OUT_DIR / "cards" / f"{name}.html"
        html.write_text(card_html(body), encoding="utf-8")
        edge_shot(str(html), OUT_DIR / f"{name}.png")
        print(f"[video] card {name}")

    # ---- 2. real demo UI (in-process offline server)
    app = DemoApp(online=False)
    httpd = DemoHTTPServer(("127.0.0.1", 0), app)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(40):
        try:
            urllib.request.urlopen(base + "/api/health", timeout=1); break
        except Exception:
            time.sleep(0.25)
    edge_shot(base + "/", OUT_DIR / "ui_offline.png")                    # tab1 top
    edge_shot(base + "/#live", OUT_DIR / "ui_live_offline.png")          # tab2 offline state
    edge_shot(base + "/", OUT_DIR / "ui_offline_tall.png",
              width=W, height=2200)                                      # more sections
    httpd.shutdown()
    print("[video] demo UI shots")

    # ---- 3. diagrams (SVG -> PNG via Edge)
    for svg, png in (("architecture.svg", "diag_arch.png"),
                     ("arms.svg", "diag_arms.png")):
        src = REPO / "diagrams" / svg
        wrapped = OUT_DIR / f"wrap_{png}.html"
        wrapped.write_text(
            "<!doctype html><meta charset='utf-8'>"
            "<style>body{margin:0;display:flex;align-items:center;"
            "justify-content:center;height:100vh;background:#fff}"
            "img{max-width:96%;max-height:92%}</style>"
            f"<img src='{src.resolve().as_uri()}'>", encoding="utf-8")
        edge_shot(str(wrapped), OUT_DIR / png)
        print(f"[video] diagram {svg}")

    # ---- 4. assemble (segments: file, seconds)
    segments = [
        ("card_title.png", 8),
        ("diag_arch.png", 12),
        ("ui_offline.png", 10),
        ("ui_offline_tall.png", 8),
        ("card_c0.png", 12),
        ("card_c1.png", 12),
        ("card_t8.png", 12),
        ("ui_live_offline.png", 10),
        ("diag_arms.png", 10),
        ("card_privacy.png", 10),
        ("card_end.png", 8),
    ]
    concat = OUT_DIR / "list.txt"
    lines = []
    for fname, secs in segments:
        src = (OUT_DIR / fname).resolve()
        if not src.exists():
            print(f"[video] MISSING {src}"); return 1
        # normalize every frame to WxH (pad, no distortion)
        norm = (OUT_DIR / f"n_{fname}").with_suffix(".png")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
                        "-vf", f"scale={W}:{H}:force_original_aspect_ratio="
                        f"decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:white",
                        "-frames:v", "1", str(norm)], check=True)
        lines.append(f"file '{norm.as_posix()}'\nduration {secs}")
    # concat demuxer: repeat last entry
    lines.append(f"file '{(OUT_DIR / ('n_' + segments[-1][0])).with_suffix('.png').as_posix()}'")
    concat.write_text("\n".join(lines), encoding="utf-8")

    out_mp4 = Path(__file__).resolve().parent / "demo-video.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-vf", f"fps=30,format=yuv420p",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-movflags", "+faststart", str(out_mp4)], check=True)
    dur = subprocess.run([FFMPEG, "-i", str(out_mp4)], capture_output=True,
                         text=True).stderr
    print(f"[video] wrote {out_mp4}")
    for ln in dur.splitlines():
        if "Duration" in ln:
            print(f"[video] {ln.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
