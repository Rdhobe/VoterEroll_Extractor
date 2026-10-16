#!/usr/bin/env python3
"""
Convert Chrome-Screen-AI OCR output of an Indian electoral roll (Hindi, 3 cards
per row) into an Excel sheet.

    python roll_to_excel.py chrome_out.md voters.xlsx

Why this is more than a simple regex loop
-----------------------------------------
The OCR dumps each page as separate, shuffled streams:
  * serial numbers      (331, 332, ...)
  * EPIC / voter IDs    (ZSY1439728, UP/70/355/0000290 ...)
  * card bodies         (नाम / पिता का नाम / मकान संख्या / आयु-लिंग)
so a card's number, ID and details are NOT next to each other.  Field-level
regexes are reliable; re-linking the three parts is a best-effort heuristic and
every inferred link is flagged in the "Check" column for manual review.
"""
import re
import sys
from collections import defaultdict

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

# ----------------------------------------------------------------- regexes
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

RE_PAGE = re.compile(r"^--- PAGE (\d+) ---\s*$", re.M)
RE_MD_NOISE = re.compile(r"^\s*#+\s*", re.M)               # '### ' heading marks
RE_SERIAL = re.compile(r"^\s*(\d{1,4})\s*$", re.M)          # a line holding only a number
RE_EPIC = re.compile(r"^\s*([A-Z]{2,4}\d{6,8}|UP/\d+/\d+/\d+)\s*$", re.M)

# card body fields (the 'ः' visarga is often OCR'd instead of ':')
COLON = r"\s*[:ः]+\s*"
RE_NAME = re.compile(rf"^\s*नाम{COLON}(.+?)\s*$", re.M)
RE_REL = re.compile(rf"^\s*(पिता|पति|माता|अन्य)\s*(?:का\s*)+नाम{COLON}(.+?)\s*$", re.M)
RE_HOUSE = re.compile(rf"मकान\s*संख्या{COLON}(\S+)")
RE_AGE_SEX = re.compile(rf"आयु{COLON}(\d+)\s*लिंग{COLON}(पुरुष|महिला|अन्य|\S+)")

# page header fields
RE_ASSEMBLY = re.compile(rf"विधानसभा\s*निर्वाचन\s*क्षेत्र[ःः:]*\s*(.+)")
RE_SECTION = re.compile(rf"अनुभाग\s*संख्या\s*और\s*नाम{COLON}(.+)")
RE_PART = re.compile(rf"भाग\s*संख्या{COLON}(\d+)")


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" ):(")


def parse_page(page_no: int, txt: str) -> list[dict]:
    serials = RE_SERIAL.findall(txt)
    names = [clean(x) for x in RE_NAME.findall(txt)]
    rels = [(a, clean(b)) for a, b in RE_REL.findall(txt)]
    houses = [clean(x) for x in RE_HOUSE.findall(txt)]
    agesex = RE_AGE_SEX.findall(txt)

    n = len(names)
    if not (len(rels) == len(houses) == len(agesex) == n):
        print(f"  ! page {page_no}: field counts differ "
              f"(name={n}, relation={len(rels)}, house={len(houses)}, age/sex={len(agesex)}) "
              f"- rows beyond the shortest list are dropped", file=sys.stderr)
        n = min(n, len(rels), len(houses), len(agesex))

    # ---- 1. card details (fields appear in the same relative order) -------
    cards = []
    for i in range(n):
        cards.append(dict(Name=names[i], Relation_Type=rels[i][0], Relation_Name=rels[i][1],
                          House=houses[i], Age=int(agesex[i][0]), Gender=agesex[i][1],
                          _file_pos=i))

    # ---- 2. serials: sorted, they are the consecutive block on the page ----
    serials = sorted(int(s) for s in serials)
    if len(serials) != n:
        print(f"  ! page {page_no}: {len(serials)} serials vs {n} cards", file=sys.stderr)

    # House numbers never decrease with serial number, but the OCR shuffles
    # card order -> stable-sort the cards by house number, then hand out serials.
    def house_key(c):
        m = re.match(r"\d+", c["House"])
        return int(m.group()) if m else 10**9
    ordered = sorted(cards, key=house_key)
    group_size = defaultdict(int)
    for c in cards:
        group_size[house_key(c)] += 1
    for pos, (c, s) in enumerate(zip(ordered, serials)):
        c["Serial"] = s
        if c["_file_pos"] != pos:
            c["_serial_note"] = "serial inferred"
        elif group_size[house_key(c)] > 1:
            # block of serials for this house is right, order inside it is OCR order
            c["_serial_note"] = "verify order within house"
        else:
            c["_serial_note"] = ""

    # ---- 3. EPIC IDs: pair a serial with the token that directly follows it
    stream = [(m.start(), "S", m.group(1)) for m in RE_SERIAL.finditer(txt)] + \
             [(m.start(), "E", m.group(1)) for m in RE_EPIC.finditer(txt)]
    stream.sort()
    epic_of, used = {}, set()
    for (_, k1, v1), (_, k2, v2) in zip(stream, stream[1:]):
        if k1 == "S" and k2 == "E":
            epic_of[int(v1)] = v2
            used.add(v2)
    leftovers = [v for _, k, v in stream if k == "E" and v not in used]
    for s in serials:                                # fill gaps in stream order
        if s not in epic_of and leftovers:
            epic_of[s] = leftovers.pop(0)
            epic_of[("guess", s)] = True
    for c in cards:
        c["EPIC"] = epic_of.get(c["Serial"], "")
        c["_epic_note"] = "EPIC inferred" if epic_of.get(("guess", c["Serial"])) else ""

    # ---- 4. page header info ---------------------------------------------
    asm = RE_ASSEMBLY.search(txt); sec = RE_SECTION.search(txt); prt = RE_PART.search(txt)
    for c in cards:
        c["Page"] = page_no
        c["Assembly"] = clean(asm.group(1)) if asm else ""
        c["Section"] = clean(sec.group(1)) if sec else ""
        c["Part"] = prt.group(1) if prt else ""
        c["Check"] = "; ".join(x for x in (c.pop("_serial_note"), c.pop("_epic_note")) if x)
        c.pop("_file_pos")
    return cards


def main(src: str, dst: str) -> None:
    text = open(src, encoding="utf-8").read().translate(DEV_DIGITS)
    text = RE_MD_NOISE.sub("", text)
    parts = RE_PAGE.split(text)           # ['', '14', body, '15', body]
    rows = []
    for i in range(1, len(parts), 2):
        rows += parse_page(int(parts[i]), parts[i + 1])

    cols = ["Page", "Part", "Section", "Serial", "EPIC", "Name", "Relation_Type",
            "Relation_Name", "House", "Age", "Gender", "Check", "Assembly"]
    df = pd.DataFrame(rows, columns=cols).sort_values("Serial")
    df.columns = ["Page", "Part No", "Section", "Serial No", "EPIC ID", "Name",
                  "Relation", "Relative Name", "House No", "Age", "Gender", "Check", "Assembly"]
    df.to_excel(dst, index=False, sheet_name="Voters")

    # ---- light formatting -------------------------------------------------
    wb = load_workbook(dst); ws = wb["Voters"]
    hdr_fill = PatternFill("solid", start_color="1F4E78")
    for c in ws[1]:
        c.font = Font(name="Arial", bold=True, color="FFFFFF"); c.fill = hdr_fill
        c.alignment = Alignment(horizontal="center")
    flag = PatternFill("solid", start_color="FFF2CC")
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name="Arial", size=10)
        if row[11].value:                                  # 'Check' column
            for c in row:
                c.fill = flag
    for i, col in enumerate(ws.columns, 1):
        w = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[get_column_letter(i)].width = min(max(w + 2, 8), 40)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(dst)
    flagged = int((df["Check"] != "").sum())
    print(f"Wrote {len(df)} rows to {dst}  ({flagged} flagged in 'Check' column)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python roll_to_excel.py input.md output.xlsx")
    main(sys.argv[1], sys.argv[2])
