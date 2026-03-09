"""
Visualise the SilverGait LangGraph workflow.

Outputs:
  - graph.png   (PNG via mermaid.ink — requires internet)
  - graph.mmd   (raw Mermaid source — open in https://mermaid.live)
  - ASCII print to terminal (always works, no dependencies)

Usage:
  python visualize.py
"""

from pathlib import Path
from backend.graph.workflow import build_graph

OUTPUT_DIR = Path(__file__).parent

app = build_graph().compile()

# ── 1. ASCII (always works) ───────────────────────────────────────────────────
print("\n=== SilverGait Agent Graph (ASCII) ===\n")
app.get_graph().print_ascii()

# ── 2. Mermaid source ────────────────────────────────────────────────────────
mermaid_src = app.get_graph().draw_mermaid()
mmd_path = OUTPUT_DIR / "graph.mmd"
mmd_path.write_text(mermaid_src)
print(f"\nMermaid source saved → {mmd_path}")
print("Paste it at https://mermaid.live to render interactively.\n")

# ── 3. PNG via mermaid.ink (requires internet) ────────────────────────────────
try:
    png_bytes = app.get_graph().draw_mermaid_png()
    png_path = OUTPUT_DIR / "graph.png"
    png_path.write_bytes(png_bytes)
    print(f"PNG saved → {png_path}")
except Exception as e:
    print(f"PNG generation skipped ({e})")
    print("Install playwright for local rendering: pip install playwright && playwright install")
