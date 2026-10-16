"""Voter-roll ENSEMBLE pipeline v2: Tesseract + Chrome Screen AI on one PDF.

V1 (scripts/main.py = Tesseract, scripts/chrome_script.py = Chrome) is FROZEN -
this file is fully self-contained (v1 logic snapshotted in, then extended), so
experiments here can never break the working v1 commands. If v2 fails, v1 stays.

Flow for a single PDF:
  1. Tesseract side  - render pages in RAM -> OpenCV grid detect (IoU-NMS) ->
                       box OCR (hin+eng + eng-whitelist header pass) -> parse.
  2. Chrome side     - Screen AI page OCR -> per-page Markdown -> chunked parse.
  3. MERGE           - match voters across engines (name -> ID -> serial),
                       adjudicate every field, flag every disagreement.

Matching (same PDF page on both sides):
  pass 1: normalized Name (duplicates disambiguated by serial/id/age)
  pass 2: strict Voter_ID (global - IDs are unique per roll)
  pass 3: (page, serial)
  rest: tes-only / chrome-only rows (coverage preserved both ways)

Adjudication (a blank never beats a value; conflicts always flagged):
  Serial/Name/Relation/Age/Gender/House : Tesseract pick on conflict
  EPIC AAA1234567 conflict              : Tesseract pick (box-crop, closer read)
  UP/.. conflict                        : longer (more complete) pick
  Chrome EPIC failing strict validation : dropped + noted (never linked on it)

Usage:
    python scripts/ensemble_v2.py --input "D:\\...\\roll.pdf" --output "D:\\Election_Output"
"""
import argparse
import contextlib
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import pymupdf as _pdf_lib
except ImportError:
    import fitz as _pdf_lib

import cv2
import numpy as np
import pandas as pd
import pytesseract

# ---------------------------------------------------------------- logger
LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"


def setup_logger(log_file=None, verbose=False):
    logger = logging.getLogger("ensemble")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(LOG_FMT))
    logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(LOG_FMT))
        logger.addHandler(fh)
    return logger


def ensure_tesseract(logger):
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        os.path.join(pf, "Tesseract-OCR", "tesseract.exe"),
        os.path.join(local, "Programs", "Tesseract-OCR", "tesseract.exe"),
        os.path.join(local, "Tesseract-OCR", "tesseract.exe"),
    ]
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        pass
    for c in candidates:
        if os.path.exists(c):
            pytesseract.pytesseract.tesseract_cmd = c
            try:
                v = pytesseract.get_tesseract_version()
                logger.info("Tesseract: %s (%s)", v, c)
                return True
            except Exception as e:
                logger.warning("Tesseract at %s failed: %s", c, e)
    if shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
        return True
    logger.error("Tesseract not found.")
    return False


# ============================================================ TESSERACT SIDE
DEV2ARAB = str.maketrans("०१२३४५६७८९", "0123456789")
EPIC_RE = re.compile(r"\b([A-Z]{3}[0-9]{7})\b")
UPROLL_RE = re.compile(r"\bUP/\d+/\d+/\d+\b")
EPIC_STRICT = re.compile(r"^[A-Z]{3}[0-9]{7}$")
UP_STRICT = re.compile(r"^UP/\d+/\d+/\d+$")

_LEAD_FIX = {"2": "Z", "5": "S", "1": "I", "0": "O", "8": "B", "6": "G"}
_DIGIT_FIX = {"I": "1", "L": "1", "O": "0", "S": "5", "B": "8", "Z": "2", "A": "4"}


def norm_digits(s):
    return s.translate(DEV2ARAB) if s else s


def correct_epic(token):
    if not token:
        return "", False
    t = token.strip().upper().replace(" ", "").strip("|!l:.,")
    m0 = UPROLL_RE.search(t)
    if m0:
        return m0.group(0), False
    m = EPIC_RE.search(t)
    if m:
        return m.group(1), False
    if len(t) >= 10:
        head, tail = t[:3], t[3:]
        fixed_head = "".join(_LEAD_FIX.get(ch, ch) for ch in head)
        fixed_tail = "".join(_DIGIT_FIX.get(ch, ch) if ch.isalpha() else ch for ch in tail)
        cand = re.sub(r"[^A-Z0-9]", "", fixed_head + fixed_tail)[:10]
        ms = EPIC_RE.search(cand)
        if ms:
            return ms.group(1), True
    return "", False


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)


def nms_boxes(boxes, iou_thresh=0.4):
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        if all(box_iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def detect_boxes(img, logger=None):
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w_img // 40), 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h_img // 40)))
    horiz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hor_kernel)
    vert = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, ver_kernel)
    grid = cv2.addWeighted(horiz, 1, vert, 1, 0)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, close_kernel)
    grid = cv2.dilate(grid, close_kernel, iterations=1)
    contours, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    expected = (w_img / 3) * (h_img / 10)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if (expected * 0.3) < area < (expected * 1.5):
            aspect = w / float(h) if h else 0
            if 1.5 < aspect < 5.0:
                boxes.append((x, y, w, h))
    before = len(boxes)
    boxes = nms_boxes(boxes)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows, cur = [], []
    for b in boxes:
        if cur and abs(b[1] - cur[0][1]) > 50:
            rows.append(sorted(cur, key=lambda z: z[0]))
            cur = []
        cur.append(b)
    if cur:
        rows.append(sorted(cur, key=lambda z: z[0]))
    ordered = [b for r in rows for b in r]
    if logger and before != len(ordered):
        logger.debug("NMS dedup: %d -> %d boxes", before, len(ordered))
    return ordered


def preprocess_roi(roi):
    roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    h, w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(gray, (int(w * 0.72), int(h * 0.25)), (w, h), (255, 255, 255), -1)
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def ocr_block(roi, config_full, config_epic):
    proc = preprocess_roi(roi)
    try:
        full = pytesseract.image_to_string(proc, config=config_full)
    except Exception:
        full = ""
    lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
    full = "\n".join(lines)
    epic_fix = None
    try:
        h = proc.shape[0]
        strip = proc[0:max(20, int(h * 0.32)), :]
        s = pytesseract.image_to_string(strip, config=config_epic)
        s = norm_digits(s.upper())
        m = UPROLL_RE.search(s) or EPIC_RE.search(s)
        if m:
            epic_fix = m.group(0)
    except Exception:
        pass
    return full, epic_fix


def parse_block(block_text, filename, epic_hint=None):
    text = (block_text or "").strip()
    if not text:
        return None
    entry = {"Serial_No": "", "Voter_ID": "", "Part_Section": "",
             "Name": "", "Relation_Type": "", "Relative_Name": "",
             "House_No": "", "Age": "", "Gender": "",
             "Source_File": filename, "Raw_Text_Dump": text}
    t = norm_digits(text)
    m = re.search(r"^\s*([\d.,]+).*?([A-Z0-9\u0966-\u096F/]{8,})\s+([\d/]+)", t, re.MULTILINE)
    if m:
        entry["Serial_No"] = m.group(1).replace(",", "").replace(".", "")
        entry["Voter_ID"], _ = correct_epic(m.group(2))
        entry["Part_Section"] = m.group(3)
    else:
        mh = re.search(r"([0-9]{2,4})\s*[|!l:\u0965.]*\s*((?:UP/[0-9/]+|[A-Z0-9]{2,}/?[A-Z0-9/]{5,}))", t)
        if mh:
            entry["Serial_No"] = mh.group(1).replace(",", "").replace(".", "")
            entry["Voter_ID"], _ = correct_epic(mh.group(2))
        else:
            me = re.search(r"\b((?:UP/[0-9/]{5,}|[A-Z]{3}[0-9]{7}|[A-Z0-9]{8,}))\b", t)
            if me:
                entry["Voter_ID"], _ = correct_epic(me.group(1))
    cur = entry["Voter_ID"] or ""
    if epic_hint and not (EPIC_RE.search(cur) or UPROLL_RE.search(cur)):
        entry["Voter_ID"] = epic_hint
    mn = re.search(r"(?:\u092e\u0924\u0926\u093e\u0930\u093e\u091a\u0947|\u092a\u0942\u0930\u094d\u0923|Waa|Hse)(?:.*?[:\-']\s*|\s+)(?:Ol\s+)?(?:['']\s*)?(.*)", t, re.IGNORECASE)
    if mn:
        entry["Name"] = dejunk_name(mn.group(1).split("\n")[0].strip())
    if not entry["Name"]:
        hn = re.search(r"\u0928\u093e\u092e\s*[:\-']\s*(.*)", t)
        if hn:
            entry["Name"] = dejunk_name(hn.group(1).strip())
    mr = re.search(r"(\u0935\u0921\u093f\u0932\u093e\u0902\u091a\u0947|\u092a\u0924\u0940\u091a\u0947|\u0906\u0908\u091a\u0947|\u0907\u0924\u0930)\s*\u0928\u093e\u0935(?:\u093e)?\s*[:\-']\s*(.*)", t)
    if mr:
        entry["Relation_Type"] = mr.group(1)
        entry["Relative_Name"] = dejunk_name(mr.group(2).strip())
    if not entry["Relative_Name"]:
        hr = re.search(r"(\u092a\u093f\u0924\u093e|\u092a\u0924\u093f|\u092e\u093e\u0924\u093e|\u092a\u0924\u094d\u0928\u0940|\u0905\u0928\u094d\u092f)\s*(?:\u0915\u093e|\u0915\u0940|\u0915\u0947)?\s*\u0928\u093e\u092e\s*[:\-']\s*(.*)", t)
        if hr:
            entry["Relation_Type"] = hr.group(1).strip()
            entry["Relative_Name"] = dejunk_name(hr.group(2).strip())
    mh2 = re.search(r"(?:\u0918\u0930 \u0915\u094d\u0930\u092e\u093e\u0902\u0915|UX PATH).*?[:\-']\s*(.*)", t, re.IGNORECASE)
    if mh2:
        entry["House_No"] = mh2.group(1).strip()
    if not entry["House_No"]:
        hh = re.search(r"(?:\u092e\u0915\u093e\u0928|Wart|Fart|makan|den)\s*(?:\u0938\u0902\u0916\u094d\u092f\u093e|sankhya|number)?\s*[:\-']?\s*(.*)", t, re.IGNORECASE)
        if hh:
            entry["House_No"] = hh.group(1).strip()
    # age and sex parsed INDEPENDENTLY: a misread लिंग must not eat the age
    ma = (re.search(r"\u0935\u092f\s*[:\-']?\s*(\d+)", t)
          or re.search(r"\u0906\u092f\u0941\s*[:\-']?\s*(\d+)", t)
          or re.search(r"\u0909\u092e\u094d\u0930\s*[:\-']?\s*(\d+)", t))
    if ma:
        entry["Age"] = ma.group(1)
    ms = re.search(r"\u0932\u093f\u0902\u0917\s*[:\-']\s*([^\n|]*)", t)
    if ms:
        rs = ms.group(1).strip()
        if "\u092a\u0941" in rs or "M" in rs or "\u092a\u0941\u0930\u0941\u0937" in rs or "\u092a\u0941\u0930\u0942\u0937" in rs:
            entry["Gender"] = "Male"
        elif any(k in rs for k in ("\u0938\u094d\u0924\u094d\u0930\u0940", "\u092e\u0939\u093f\u0932\u093e", "F", "AY")) or "\u0938\u094d\u0930\u0940" in rs:
            entry["Gender"] = "Female"
    return entry


# =============================================================== CHROME SIDE
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
RE_SERIAL = re.compile(r"^\s*(\d{1,4})\s*$", re.M)
RE_EPIC = re.compile(r"^\s*([A-Z]{2,4}\d{6,8}|UP/\d+/\d+/\d+)\s*$", re.M)
COLON = r"\s*[:ः]+\s*"
RE_NAME = re.compile(rf"^\s*नाम{COLON}(.+?)\s*$", re.M)
RE_REL = re.compile(rf"^\s*(पिता|पति|माता|अन्य)\s*(?:का\s*)+नाम{COLON}(.+?)\s*$", re.M)
RE_HOUSE = re.compile(rf"मकान\s*संख्या{COLON}(\S+)")
RE_AGE_SEX = re.compile(rf"आयु{COLON}(\d+)\s*लिंग{COLON}(पुरुष|महिला|अन्य|\S+)")
RE_AGE_ALONE = re.compile(rf"आयु{COLON}(\d+)")
RE_SEX_ALONE = re.compile(rf"लिंग{COLON}(\S+)")
RE_ASSEMBLY = re.compile(rf"विधानसभा\s*निर्वाचन\s*क्षेत्र[ःः:]*\s*(.+)")
RE_SECTION = re.compile(rf"अनुभाग\s*संख्या\s*और\s*नाम{COLON}(.+)")
RE_PART = re.compile(rf"भाग\s*संख्या{COLON}(\d+)")


@contextlib.contextmanager
def quiet_stderr(quiet=True):
    if not quiet:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old, 2)
        os.close(devnull)
        os.close(old)


def _silence_fitz_shim():
    try:
        import pymupdf
        if hasattr(pymupdf, "set_messages"):
            pymupdf.set_messages(stream=open(os.devnull, "w"))
    except ImportError:
        pass
    if "fitz" in sys.modules:
        return
    try:
        import fitz  # noqa: F401
    except ImportError:
        pass


def clean_md(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" ):(")


def parse_md_page(page_no: int, txt: str) -> list:
    txt = txt.translate(DEV_DIGITS)
    txt = re.sub(r"^\s*#+\s*", "", txt, flags=re.M)
    serials = RE_SERIAL.findall(txt)
    name_hits = list(RE_NAME.finditer(txt))
    if not name_hits:
        if txt.strip():
            print(f"  ! page {page_no}: no Name lines found", file=sys.stderr)
        return []
    cards = []
    for k, m in enumerate(name_hits):
        chunk = txt[m.start():name_hits[k + 1].start() if k + 1 < len(name_hits) else len(txt)]
        name = dejunk_name(clean_md(m.group(1)))
        rm = RE_REL.search(chunk)
        hm = RE_HOUSE.search(chunk)
        am = RE_AGE_SEX.search(chunk)
        aa = am or RE_AGE_ALONE.search(chunk)
        sm = am or RE_SEX_ALONE.search(chunk)
        notes = []
        rel_type, rel_name = (rm.group(1), dejunk_name(re.sub(r"^नाम\s*[:ः]\s*", "", clean_md(rm.group(2))))) if rm else ("", "")
        if rm and RE_REL.search(chunk, rm.end()):
            notes.append("2nd relation in block ignored")
        house = clean_md(hm.group(1)) if hm else ""
        if not hm:
            notes.append("house missing")
        if aa:
            age = int(aa.group(1))
        else:
            age = ""
            notes.append("age missing")
        if sm:
            gender = sm.group(2) if sm.re is RE_AGE_SEX else sm.group(1)
        else:
            gender = ""
            notes.append("sex missing")
        if rel_name and rel_name == name:
            rel_name = ""
            notes.append("relation==name dropped")
        if rel_type == "पति" and gender == "पुरुष":
            notes.append("relation/gender conflict")
        cards.append(dict(Name=name, Relation_Type=rel_type, Relation_Name=rel_name,
                          House=house, Age=age, Gender=gender, _notes=notes,
                          _file_pos=k))
    n = len(cards)
    serials = sorted(set(int(s) for s in serials))
    clusters, cur = [], []
    for s in serials:
        if cur and s - cur[-1] > 5:
            clusters.append(cur)
            cur = []
        cur.append(s)
    if cur:
        clusters.append(cur)
    best = max(clusters, key=len) if clusters else []
    stray = sorted(set(serials) - set(best))
    if stray:
        print(f"  ! page {page_no}: ignoring stray serials {stray}", file=sys.stderr)
    serials = best
    if len(serials) != n:
        print(f"  ! page {page_no}: {len(serials)} serials vs {n} cards", file=sys.stderr)

    def house_key(c):
        m = re.match(r"\d+", str(c["House"]))
        return int(m.group()) if m else 10 ** 9
    ordered = sorted(cards, key=house_key)
    group_size = defaultdict(int)
    for c in cards:
        group_size[house_key(c)] += 1
    for pos, c in enumerate(ordered):
        if pos < len(serials):
            c["Serial"] = serials[pos]
            if c["_file_pos"] != pos:
                c["_serial_note"] = "serial inferred"
            elif group_size[house_key(c)] > 1:
                c["_serial_note"] = "verify order within house"
            else:
                c["_serial_note"] = ""
        else:
            c["Serial"] = None
            c["_serial_note"] = "serial missing"

    stream = [(m.start(), "S", m.group(1)) for m in RE_SERIAL.finditer(txt)] + \
             [(m.start(), "E", m.group(1)) for m in RE_EPIC.finditer(txt)]
    stream.sort()
    epic_of, used = {}, set()
    for (_, k1, v1), (_, k2, v2) in zip(stream, stream[1:]):
        if k1 == "S" and k2 == "E":
            epic_of[int(v1)] = v2
            used.add(v2)
    leftovers = [v for _, k, v in stream if k == "E" and v not in used]
    for s in serials:
        if s not in epic_of and leftovers:
            epic_of[s] = leftovers.pop(0)
            epic_of[("guess", s)] = True
    for c in cards:
        s = c.get("Serial")
        c["EPIC"] = epic_of.get(s, "") if s is not None else ""
        c["_epic_note"] = "EPIC inferred" if (s is not None and epic_of.get(("guess", s))) else ""

    asm = RE_ASSEMBLY.search(txt); sec = RE_SECTION.search(txt); prt = RE_PART.search(txt)
    for c in cards:
        c["Page"] = page_no
        c["Assembly"] = clean_md(asm.group(1)) if asm else ""
        c["Section"] = clean_md(sec.group(1)) if sec else ""
        c["Part"] = prt.group(1) if prt else ""
        c["Check"] = "; ".join(x for x in [c.pop("_serial_note"), c.pop("_epic_note")] + c.pop("_notes") if x)
        c.pop("_file_pos")
    return cards


# =================================================================== MERGER
CHR_GENDER = {"पुरुष": "Male", "महिला": "Female"}


def norm_name(s):
    """Canonical form for matching: nukta/anusvara/ZW folded (फ़==फ)."""
    s = re.sub(r"[\s।|:\-.,]", "", str(s)).strip() if s else ""
    s = s.replace("़", "").replace("ँ", "ं")
    return "".join(ch for ch in s if ch not in ("‌", "‍"))


# trailing OCR-fragment tokens: never part of a real name (crop-edge junk
# like "संतसरन जाना oe", "मकान ... = उपलब्ध" spillover, stray matras)
JUNK_TAIL = {"जाना", "oe", "OE", "o", "O", "हा", "हे", "-", "=", "—", "–",
             "उपलब्ध", "फोटो", "े", "।", ":"}


def dejunk_name(s):
    toks = str(s).split()
    while toks and toks[-1].strip(".,:;~—-()\"'") in JUNK_TAIL:
        toks.pop()
    return " ".join(toks).strip()


def page_of_tes(source_file):
    m = re.search(r"_p(\d+)$", str(source_file or ""))
    return int(m.group(1)) if m else 0


def strict_id(v):
    v = (v or "").strip()
    if EPIC_STRICT.match(v) or UP_STRICT.match(v):
        return v
    return ""


def pick_value(tval, cval, prefer="tes", note_t="", note_c="", note_both=""):
    """A blank never beats a value. Returns (value, note_or_empty)."""
    t = ("" if tval is None else str(tval)).strip()
    c = ("" if cval is None else str(cval)).strip()
    if t and not c:
        return t, (note_t or "")
    if c and not t:
        return c, (note_c or "")
    if not t and not c:
        return "", ""
    if t == c:
        return t, ""
    return (t if prefer == "tes" else c), note_both


def adjudicate_pair(t, c):
    """Merge one Tesseract + one Chrome card. Returns (merged_row, conflict_list)."""
    hard = []   # value disagreements
    soft = []   # single-source fills
    row = {}

    row["Source_Page"] = t["_page"]
    s, n = pick_value(t.get("Serial_No"), c.get("Serial"),
                      note_c="serial chrome-only", note_t="serial tes-only",
                      note_both=f"serial differs T:{t.get('Serial_No')} C:{c.get('Serial')}")
    row["Serial_No"] = s
    if n:
        (soft if "differs" not in n else hard).append(n)

    tv, cv_raw = (t.get("Voter_ID") or "").strip(), (c.get("EPIC") or "").strip()
    cv = strict_id(cv_raw)
    cv_note = "" if cv else (f"invalid chrome ID dropped: {cv_raw}" if cv_raw else "")
    if tv and cv:
        if tv == cv:
            row["Voter_ID"] = tv
        else:
            is_up = tv.startswith("UP/") or cv.startswith("UP/")
            if is_up:
                row["Voter_ID"] = tv if len(tv) >= len(cv) else cv
            else:
                row["Voter_ID"] = tv  # EPIC conflict: box-crop read wins
            hard.append(f"ID conflict T:{tv} C:{cv_raw}")
    elif tv:
        row["Voter_ID"] = tv
        soft.append("ID tes-only")
    elif cv:
        row["Voter_ID"] = cv
        soft.append("ID chrome-only")
    else:
        row["Voter_ID"] = ""
    if cv_note:
        soft.append(cv_note)

    nm, n = pick_value(t.get("Name"), c.get("Name"),
                       note_both=f"name differs T:{t.get('Name')} C:{c.get('Name')}")
    row["Name"] = nm
    if n:
        (soft if "differs" not in n else hard).append(n)

    for col, tc, cc in (("Relation_Type", "Relation_Type", "Relation_Type"),
                        ("Relative_Name", "Relative_Name", "Relation_Name"),
                        ("House_No", "House_No", "House")):
        # Relative_Name prefers chrome: measured 79-vs-34 wins on disagreements
        v, nn = pick_value(t.get(tc), c.get(cc),
                           prefer=("chrome" if col == "Relative_Name" else "tes"),
                           note_t=f"{col} tes-only", note_c=f"{col} chrome-only",
                           note_both=f"{col} differs T:{t.get(tc)} C:{c.get(cc)}")
        row[col] = v
        if nn:
            (soft if "differs" not in nn else hard).append(nn)

    av, nn = pick_value(t.get("Age"), c.get("Age"),
                        note_both=f"age differs T:{t.get('Age')} C:{c.get('Age')}")
    row["Age"] = av
    if nn:
        (soft if "differs" not in nn else hard).append(nn)

    cg = CHR_GENDER.get((c.get("Gender") or "").strip(), (c.get("Gender") or "").strip())
    gv, nn = pick_value(t.get("Gender"), cg,
                        note_both=f"gender conflict T:{t.get('Gender')} C:{c.get('Gender')}")
    row["Gender"] = gv
    if nn:
        (soft if "conflict" not in nn else hard).append(nn)

    row["Part_Section"] = (t.get("Part_Section") or "").strip()
    return row, hard, soft


def singleton_row(side, card, page):
    """A card found by only one engine - kept with full provenance."""
    if side == "tes":
        return {
            "Source_Page": page, "Serial_No": (card.get("Serial_No") or "").strip(),
            "Voter_ID": (card.get("Voter_ID") or "").strip(),
            "Name": (card.get("Name") or "").strip(),
            "Relation_Type": (card.get("Relation_Type") or "").strip(),
            "Relative_Name": (card.get("Relative_Name") or "").strip(),
            "Age": (card.get("Age") or "").strip(),
            "Gender": (card.get("Gender") or "").strip(),
            "House_No": (card.get("House_No") or "").strip(),
            "Part_Section": (card.get("Part_Section") or "").strip(),
            "Match": "tes-only", "Conflicts": "",
            "Engine_Note": "",
        }, []
    return {
        "Source_Page": page, "Serial_No": "" if card.get("Serial") is None else str(card.get("Serial")),
        "Voter_ID": strict_id(card.get("EPIC")),
        "Name": (card.get("Name") or "").strip(),
        "Relation_Type": (card.get("Relation_Type") or "").strip(),
        "Relative_Name": (card.get("Relation_Name") or "").strip(),
        "Age": "" if card.get("Age") == "" else str(card.get("Age")),
        "Gender": CHR_GENDER.get((card.get("Gender") or "").strip(), (card.get("Gender") or "").strip()),
        "House_No": (card.get("House") or "").strip(),
        "Part_Section": "",
        "Match": "chrome-only", "Conflicts": "",
        "Engine_Note": (card.get("Check") or "").strip(),
    }, []


def merge_cards(tes_cards, chr_cards):
    """Name -> ID -> serial matching, then per-field adjudication."""
    for tc in tes_cards:
        tc["_page"] = page_of_tes(tc.get("Source_File"))
        tc["_used"] = False
    for cc in chr_cards:
        cc["_used"] = False

    tes_by_key = defaultdict(list)
    for i, tc in enumerate(tes_cards):
        if norm_name(tc.get("Name")):
            tes_by_key[(tc["_page"], norm_name(tc.get("Name")))].append(i)
    chr_order = sorted(range(len(chr_cards)), key=lambda i: (chr_cards[i].get("Page", 0), i))

    merged = []

    def use_pair(ti, ci, match, extra_note=""):
        tc, cc = tes_cards[ti], chr_cards[ci]
        tc["_used"] = True
        cc["_used"] = True
        row, hard, soft = adjudicate_pair(tc, cc)
        row["Match"] = match
        notes = hard + soft
        if extra_note:
            notes.append(extra_note)
        if cc.get("Check"):
            notes.append("chrome: " + cc["Check"])
        row["Conflicts"] = "; ".join(n for n in notes if n)
        row["Engine_Note"] = (cc.get("Check") or "").strip()
        row["_hard"] = len(hard)
        merged.append(row)

    # pass 1: normalized name on the same page
    for ci in chr_order:
        cc = chr_cards[ci]
        key = (cc.get("Page", 0), norm_name(cc.get("Name")))
        cands = [ti for ti in tes_by_key.get(key, []) if not tes_cards[ti]["_used"]]
        if not cands:
            continue
        if len(cands) == 1:
            use_pair(cands[0], ci, "name")
            continue
        best, best_score = cands[0], -1
        for ti in cands:
            tc = tes_cards[ti]
            score = (str(tc.get("Serial_No") or "") == str(cc.get("Serial") or "")) + \
                    ((tc.get("Voter_ID") or "") == strict_id(cc.get("EPIC")) and bool(tc.get("Voter_ID"))) + \
                    (str(tc.get("Age") or "") == str(cc.get("Age") or "") and str(tc.get("Age") or "") != "")
            if score > best_score:
                best, best_score = ti, score
        use_pair(best, ci, "name",
                 "" if best_score > 0 else "ambiguous name match, first-listed")

    # pass 1b: fuzzy names (OCR spelling variants: रवि/रवी, anusvara drops).
    # Same page only, ratio >= 0.88, and never when both sides carry a strict
    # ID that disagrees (IDs rule - different IDs mean different voters).
    import difflib
    for ci in chr_order:
        cc = chr_cards[ci]
        if cc["_used"]:
            continue
        ck = norm_name(cc.get("Name"))
        if not ck:
            continue
        c_id = strict_id(cc.get("EPIC"))
        best_ti, best_r = None, 0.0
        for ti, tc in enumerate(tes_cards):
            if tc["_used"] or tc["_page"] != cc.get("Page", 0):
                continue
            tk = norm_name(tc.get("Name"))
            if not tk or tk == ck:
                continue
            if c_id:
                t_id = strict_id(tc.get("Voter_ID"))
                if t_id and t_id != c_id:
                    continue
            r = difflib.SequenceMatcher(None, tk, ck).ratio()
            if r >= 0.88 and r > best_r:
                best_ti, best_r = ti, r
        if best_ti is not None:
            use_pair(best_ti, ci, "name", f"fuzzy name match ({best_r:.2f})")

    # pass 2: strict IDs (global - unique per roll)
    tes_by_id, chr_by_id = {}, {}
    for i, tc in enumerate(tes_cards):
        if not tc["_used"] and strict_id(tc.get("Voter_ID")):
            tes_by_id.setdefault(strict_id(tc.get("Voter_ID")), i)
    for i, cc in enumerate(chr_cards):
        if not cc["_used"] and strict_id(cc.get("EPIC")):
            chr_by_id.setdefault(strict_id(cc.get("EPIC")), i)
    for vid in tes_by_id:
        if vid in chr_by_id:
            use_pair(tes_by_id[vid], chr_by_id[vid], "id", "linked by ID")

    # pass 3: (page, serial)
    tes_by_ser, chr_by_ser = {}, {}
    for i, tc in enumerate(tes_cards):
        if not tc["_used"] and str(tc.get("Serial_No") or "").strip():
            tes_by_ser.setdefault((tc["_page"], str(tc.get("Serial_No")).strip()), i)
    for i, cc in enumerate(chr_cards):
        if not cc["_used"] and cc.get("Serial") is not None:
            chr_by_ser.setdefault((cc.get("Page", 0), str(cc.get("Serial"))), i)
    for key in tes_by_ser:
        if key in chr_by_ser:
            use_pair(tes_by_ser[key], chr_by_ser[key], "serial", "linked by serial")

    for i, tc in enumerate(tes_cards):
        if not tc["_used"]:
            row, _ = singleton_row("tes", tc, tc["_page"])
            row["_hard"] = 0
            merged.append(row)
    for i, cc in enumerate(chr_cards):
        if not cc["_used"]:
            row, _ = singleton_row("chrome", cc, cc.get("Page", 0))
            row["_hard"] = 0
            merged.append(row)
    # same strict ID on two rows = at least one bad link: surface it
    by_id = defaultdict(list)
    for idx, r in enumerate(merged):
        vid = strict_id(r.get("Voter_ID"))
        if vid:
            by_id[vid].append(idx)
    for vid, idxs in by_id.items():
        if len(idxs) > 1:
            for idx in idxs:
                r = merged[idx]
                r["Conflicts"] = "; ".join(
                    x for x in [r.get("Conflicts", ""),
                                f"shared ID {vid} across {len(idxs)} rows"] if x)
                r["_hard"] = r.get("_hard", 0) + 1
    return merged


# ================================================================== OUTPUTS
ENSEMBLE_COLS = ["Source_Page", "Serial_No", "Voter_ID", "Name", "Relation_Type",
                 "Relative_Name", "Age", "Gender", "House_No", "Part_Section",
                 "Match", "Conflicts", "Engine_Note"]
ILLEGAL_XLSX_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_cell(v):
    if isinstance(v, str):
        return ILLEGAL_XLSX_RE.sub("", v).strip()
    return v


def finalize_merged(rows):
    df = pd.DataFrame(rows)
    for col in ENSEMBLE_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[ENSEMBLE_COLS]
    before = len(df)
    df.drop_duplicates(inplace=True)
    df = df[~((df["Name"].fillna("").astype(str).str.strip() == "")
               & (df["Voter_ID"].fillna("").astype(str).str.strip() == ""))]
    for col in ["Voter_ID", "Name", "Relation_Type", "Relative_Name", "Gender",
                "House_No", "Part_Section", "Match", "Conflicts", "Engine_Note"]:
        df[col] = df[col].apply(clean_cell).replace("", pd.NA).astype("string")
    for col in ["Source_Page", "Serial_No", "Age"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.sort_values(["Source_Page", "Serial_No"]).reset_index(drop=True)
    return df, before


def write_ensemble(df, xlsx_path, logger):
    csv_path = Path(str(xlsx_path)).with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)
    try:
        from openpyxl import load_workbook
        wb = load_workbook(xlsx_path)
        ws = wb.active
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        widths = {"Source_Page": 11, "Serial_No": 10, "Voter_ID": 20, "Name": 24,
                  "Relation_Type": 12, "Relative_Name": 24, "Age": 7, "Gender": 9,
                  "House_No": 12, "Part_Section": 12, "Match": 11, "Conflicts": 40,
                  "Engine_Note": 30}
        from openpyxl.styles import PatternFill
        from openpyxl.utils import get_column_letter
        conflict_idx = ENSEMBLE_COLS.index("Conflicts") + 1
        flag = PatternFill("solid", start_color="FCE4EC")
        for r in ws.iter_rows(min_row=2):
            if r[conflict_idx - 1].value:
                for c_ in r:
                    c_.fill = flag
        for idx, col in enumerate(ws.columns, 1):
            ws.column_dimensions[col[0].column_letter].width = widths.get(
                ENSEMBLE_COLS[idx - 1], 15)
        wb.save(xlsx_path)
    except Exception as e:
        if logger:
            logger.warning("excel polish skipped (%s)", e)
    return csv_path


# ====================================================================== CLI
def build_parser():
    p = argparse.ArgumentParser(description="Voter-roll ENSEMBLE v2: Tesseract + Chrome on one PDF.")
    p.add_argument("--input", required=True, help="Input PDF file (single file)")
    p.add_argument("--output", default="D:\\Election_Output", help="Output folder")
    p.add_argument("--pages", type=int, default=0, help="Max pages per PDF, testing (0=all)")
    p.add_argument("--start-page", type=int, default=1)
    p.add_argument("--end-page", type=int, default=0, help="0 = last page")
    p.add_argument("--skip-pages", default="1,2", help="p1 = summary, p2 = images only")
    p.add_argument("--dpi", type=int, default=216, help="Tesseract render DPI")
    p.add_argument("--chrome-dpi", type=int, default=200, help="Chrome raster DPI")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--padding", type=int, default=15)
    p.add_argument("--ocr-lang", default="hin+eng", help="Tesseract langs for boxes")
    p.add_argument("--resume", action="store_true", help="Skip if ensemble XLSX is fresh")
    p.add_argument("--log-file", default=None)
    p.add_argument("--verbose", action="store_true")
    return p


def parse_pages_arg(spec):
    if not spec or spec.strip().lower() == "all":
        return None
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a), int(b) + 1))
        elif part:
            pages.add(int(part))
    return sorted(pages)


def run_tesseract(pdf_path, stem, page_list, args, logger):
    """Box-level OCR per page. Returns (records, audit_text, n_blocks)."""
    config_full = "--oem 3 --psm 6 -l %s" % args.ocr_lang
    config_epic = r"--oem 3 --psm 7 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/"
    doc = _pdf_lib.open(str(pdf_path))
    zoom = args.dpi / 72.0
    records, audit, n_blocks = [], [], 0
    for pno in page_list:
        page = doc.load_page(pno - 1)
        pix = page.get_pixmap(matrix=_pdf_lib.Matrix(zoom, zoom))
        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("tes p%d: render failed", pno)
            continue
        boxes = detect_boxes(img, logger)
        tag = "%s_p%04d" % (stem, pno)
        if not boxes:
            audit.append("FILENAME: %s\nERROR: No blocks found." % tag)
            continue
        h_i, w_i = img.shape[:2]

        def _one(b):
            x, y, w, h = b
            x0, y0 = max(0, x - args.padding), max(0, y - args.padding)
            x1, y1 = min(w_i, x + w + args.padding), min(h_i, y + h + args.padding)
            return ocr_block(img[y0:y1, x0:x1], config_full, config_epic)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            ocr_out = list(ex.map(_one, boxes))
        audit.append("FILENAME: %s" % tag)
        for idx, (txt, hint) in enumerate(ocr_out):
            audit.append("--- BLOCK %d ---\n%s" % (idx + 1, txt))
            try:
                rec = parse_block(txt, tag, epic_hint=hint)
            except Exception as e:
                logger.warning("tes p%d block %d parse failed: %s", pno, idx + 1, e)
                rec = None
            if rec and (rec.get("Name") or rec.get("Voter_ID")):
                records.append(rec)
        n_blocks += len(boxes)
        logger.info("tes: page %d done (%d blocks)", pno, len(boxes))
    doc.close()
    return records, "\n".join(audit), n_blocks


def run_chrome(pdf_path, page_list, args, logger):
    """Page-level Screen AI OCR + chunked parse. Returns (cards, raw_sections)."""
    try:
        with quiet_chrome_stderr(not args.verbose):
            from chrome_ocr import ScreenAIEngine, ocr_pdf
    except ImportError as e:
        raise RuntimeError("chrome-ocr not installed. Run: pip install -r requirements.txt") from e
    try:
        with quiet_chrome_stderr(not args.verbose):
            engine = ScreenAIEngine()
    except Exception as e:
        raise RuntimeError("Screen AI engine init failed: %s" % e) from e
    cards, raws = [], []
    try:
        for i, pno in enumerate(page_list, 1):
            with quiet_chrome_stderr(not args.verbose):
                md = ocr_pdf(str(pdf_path), dpi=args.chrome_dpi, pages=[pno], engine=engine)
            md = md or ""
            raws.append("--- PAGE %d ---\n\n%s" % (pno, md))
            try:
                cards.extend(parse_md_page(pno, md))
            except Exception as e:
                logger.warning("chrome p%d parse failed: %s", pno, e)
            if i % 5 == 0 or i == len(page_list):
                logger.info("chrome: %d/%d pages", i, len(page_list))
    except Exception:
        logger.exception("Chrome extraction failed")
        raise
    return cards, raws


@contextlib.contextmanager
def quiet_chrome_stderr(quiet=True):
    if not quiet:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old, 2)
        os.close(devnull)
        os.close(old)


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.log_file or str(out_dir / "ensemble.log"), args.verbose)

    pdf = Path(args.input)
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        logger.error("Single PDF file required (bulk = scripts/main.py). Got: %s", args.input)
        return 1
    stem = pdf.stem
    xlsx_path = out_dir / (stem + "_ensemble.xlsx")
    if args.resume and xlsx_path.exists():
        try:
            if xlsx_path.stat().st_mtime >= pdf.stat().st_mtime:
                logger.info("SKIP (resume): ensemble XLSX is fresh.")
                return 0
        except OSError:
            pass

    if not ensure_tesseract(logger):
        return 2
    _silence_fitz_shim_v2()

    doc = _pdf_lib.open(str(pdf))
    n_pdf = doc.page_count
    doc.close()
    s = max(1, args.start_page)
    e = args.end_page if args.end_page > 0 else n_pdf
    e = min(e, n_pdf)
    pages = list(range(s, e + 1))
    if args.pages > 0:
        pages = pages[:args.pages]
    skip = set(parse_pages_arg(args.skip_pages) or [])
    pages = [p for p in pages if p not in skip]
    logger.info("%s: %d pages (%d PDF pages, skip %s)", stem, len(pages), n_pdf,
                args.skip_pages or "none")
    if not pages:
        logger.error("No pages left.")
        return 1

    t0 = time.time()
    logger.info("=== STAGE 1a: Tesseract (%d pages) ===", len(pages))
    tes_cards, tes_audit, n_blocks = run_tesseract(str(pdf), stem, pages, args, logger)
    (out_dir / (stem + "_tes.txt")).write_text(
        "TES audit %s | %d blocks\n\n" % (stem, n_blocks) + "\n".join(tes_audit),
        encoding="utf-8")
    logger.info("tes: %d blocks -> %d records", n_blocks, len(tes_cards))

    logger.info("=== STAGE 1b: Chrome (%d pages) ===", len(pages))
    try:
        chr_cards, chr_raw = run_chrome(str(pdf), pages, args, logger)
    except RuntimeError as err:
        logger.error("%s", err)
        return 1
    (out_dir / (stem + "_chrome_raw.md")).write_text(
        "# %s (raw shuffled)\n\n" % stem + "\n\n".join(chr_raw), encoding="utf-8")
    logger.info("chrome: %d cards", len(chr_cards))

    logger.info("=== STAGE 2: merge ===")
    merged = merge_cards(tes_cards, chr_cards)
    df, before = finalize_merged(merged)
    try:
        write_ensemble(df, str(xlsx_path), logger)
    except PermissionError:
        logger.error("CLOSE %s in Excel and re-run (file locked).", xlsx_path.name)
        return 1
    by_match = df["Match"].fillna("").value_counts().to_dict()
    n_conf = int((df["Conflicts"].fillna("") != "").sum())
    n_hard = sum(1 for r in merged if r.get("_hard"))
    logger.info("MERGED: %d rows (match=%s), %d noted (%d hard conflicts), %.1fs -> %s",
                len(df), by_match, n_conf, n_hard, time.time() - t0, xlsx_path.name)
    return 0


def _silence_fitz_shim_v2():
    try:
        import pymupdf
        if hasattr(pymupdf, "set_messages"):
            pymupdf.set_messages(stream=open(os.devnull, "w"))
    except ImportError:
        pass
    if "fitz" in sys.modules:
        return
    try:
        import fitz  # noqa: F401
    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(main())
