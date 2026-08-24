"""Assemble the standalone draft board: engine JS + UI + inlined data."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    data = (HERE / "board_data.json").read_text()
    shell = (HERE / "web" / "shell.html").read_text()
    engine = (HERE / "web" / "engine.js").read_text()
    ui = (HERE / "web" / "ui.js").read_text()
    css = (HERE / "web" / "style.css").read_text()

    html = (
        shell.replace("/*STYLE*/", css)
        .replace("//ENGINE", engine)
        .replace("//UI", ui)
        .replace('"DATA_PLACEHOLDER"', data)
    )
    out = HERE / "draft_board.html"
    out.write_text(html)
    print(f"wrote {out.name} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
