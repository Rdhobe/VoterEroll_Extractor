"""MD-only trial: Agra WI-1 roll, page 3 -> BLOCKED Markdown. No Excel.

Kept OUTSIDE scripts/ on purpose: quick experiments live here,
the real pipeline lives in scripts/.

Run from the project root (.venv active):
    python test_chrome_md.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from chrome_script import (extract_pdf_to_md, parse_md_page, setup_logger,  # noqa: E402
                           write_blocked_md)

PDF = (r"D:\Election_Download\Uttar_Pradesh\Agra\AC_86_Etmadpur"
       r"\2026-EROLLGEN-S24-86-SIR-FinalRoll-Revision1-HIN-1-WI.pdf")
OUT_MD = r"D:\Election_Output\test_page3.md"
PAGE = 3


def main():
    logger = setup_logger()
    paged, n_chars, dt = extract_pdf_to_md(
        PDF, pages_spec=str(PAGE), skip_pages="", dpi=200, logger=logger)
    pno, body = paged[0]
    cards = parse_md_page(pno, body)
    c0 = cards[0] if cards else {}
    write_blocked_md([(pno, c0.get("Assembly") or "", c0.get("Section") or "",
                       c0.get("Part") or "", cards)],
                     OUT_MD, Path(PDF).stem, 200, str(PAGE), dt)
    text = Path(OUT_MD).read_text(encoding="utf-8")
    first = text.split("--- BLOCK", 1)
    print("=" * 60)
    print(f"PAGE {PAGE}: {len(cards)} blocks ({n_chars} raw chars)")
    print("=" * 60)
    print("--- BLOCK" + (first[1].split("------------------------------")[0]
                         if len(first) > 1 else "(no blocks)"))
    print("=" * 60)
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    sys.exit(main())
