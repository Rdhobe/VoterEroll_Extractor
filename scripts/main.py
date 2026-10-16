"""Unified UP voter-roll pipeline: PDF -> OCR text -> Excel.

Single entry point replacing the 3 separate scripts:
    python scripts/main.py --input "D:\\Election_Download\\Uttar_Pradesh" --output "D:\\Election_Output" --resume

Features:
- multi-PDF discovery (district/assembly tree), --resume where-left-off
- streaming render (no multi-GB PNG cache; --keep-images optional)
- accuracy fixes: box IoU dedup (60->30), header re-OCR with whitelist,
  Devanagari digit normalization, EPIC OCR-confusion correction
- proper logging (console + file), per-PDF stats, dry-run mode
"""
import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
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
    logger = logging.getLogger("voterpipe")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter(LOG_FMT))
    logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(LOG_FMT))
        logger.addHandler(fh)
    return logger


def ensure_tesseract(logger):
    """Point pytesseract at tesseract.exe (PATH or standard install spots)."""
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
    # last try: rely on PATH
    if shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
        return True
    logger.error("Tesseract not found. Install it or set pytesseract.pytesseract.tesseract_cmd.")
    return False


# ------------------------------------------------------- accuracy helpers
DEV2ARAB = str.maketrans("०१२३४५६७८९", "0123456789")
OCR_CONFIG_FULL = r"--oem 3 --psm 6 -l hin+eng"
OCR_CONFIG_EPIC = r"--oem 3 --psm 7 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/"
EPIC_RE = re.compile(r"\b([A-Z]{3}[0-9]{7})\b")
UPROLL_RE = re.compile(r"\bUP/\d+/\d+/\d+\b")
EPIC_LOOSE_RE = re.compile(r"\b([A-Z0-9]{2,}/?[A-Z0-9/]{5,})\b")

# first-letter confusions seen in the wild: 2SY->ZSY, 5->S, 1->I ...
_LEAD_FIX = {"2": "Z", "5": "S", "1": "I", "0": "O", "8": "B", "6": "G"}
_DIGIT_FIX = {"I": "1", "L": "1", "O": "0", "S": "5", "B": "8", "Z": "2", "A": "4"}


def norm_digits(s):
    return s.translate(DEV2ARAB) if s else s


def correct_epic(token):
    """Fix common Tesseract confusions. Returns (id_or_empty, fixed_bool).

    Strict only: a value is returned ONLY if it is a valid EPIC (AAA1234567)
    or UP roll number. Garbage is returned as '' so bad IDs never pollute
    the sheet or break Voter_ID dedup (falls back to Name+Age).
    """
    if not token:
        return "", False
    t = token.strip().upper().replace(" ", "").strip("|!l:.,")
    m0 = UPROLL_RE.search(t)
    if m0:
        return m0.group(0), False
    m = EPIC_RE.search(t)
    if m:
        return m.group(1), False
    # try leading-char fix: e.g. 2SY3337284 -> ZSY3337284
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
    """Drop nested/duplicate detections (fixes the 60-instead-of-30 bug)."""
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        if all(box_iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


# ------------------------------------------------------- vision + OCR
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
    # reading order: group rows (dy<50), left-to-right inside row
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows, cur, out = [], [], None
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


def ocr_block(roi, padding=15):
    """Full-block OCR + header-strip re-OCR for the EPIC line."""
    proc = preprocess_roi(roi)
    try:
        full = pytesseract.image_to_string(proc, config=OCR_CONFIG_FULL)
    except Exception:
        full = ""
    lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
    full = "\n".join(lines)
    # header strip: top 32% re-OCR with eng whitelist (much better IDs)
    epic_fix = None
    try:
        h = proc.shape[0]
        strip = proc[0:max(20, int(h * 0.32)), :]
        s = pytesseract.image_to_string(strip, config=OCR_CONFIG_EPIC)
        s = norm_digits(s.upper())
        m = UPROLL_RE.search(s) or EPIC_RE.search(s)
        if m:
            epic_fix = m.group(0)
        # NOTE: no loose fallback here - a made-up ID is worse than blank
        # (blank falls back to Name+Age dedup; garbage breaks it).
    except Exception:
        pass
    return full, epic_fix


# ------------------------------------------------------- parsing
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
    # whitelist header OCR wins when regex found nothing usable
    cur = entry["Voter_ID"] or ""
    if epic_hint and not (EPIC_RE.search(cur) or UPROLL_RE.search(cur)):
        entry["Voter_ID"] = epic_hint

    mn = re.search(r"(?:\u092e\u0924\u0926\u093e\u0930\u093e\u091a\u0947|\u092a\u0942\u0930\u094d\u0923|Waa|Hse)(?:.*?[:\-']\s*|\s+)(?:Ol\s+)?(?:['']\s*)?(.*)", t, re.IGNORECASE)
    if mn:
        entry["Name"] = mn.group(1).split("\n")[0].strip()
    if not entry["Name"]:
        hn = re.search(r"\u0928\u093e\u092e\s*[:\-']\s*(.*)", t)
        if hn:
            entry["Name"] = hn.group(1).strip()

    mr = re.search(r"(\u0935\u0921\u093f\u0932\u093e\u0902\u091a\u0947|\u092a\u0924\u0940\u091a\u0947|\u0906\u0908\u091a\u0947|\u0907\u0924\u0930)\s*\u0928\u093e\u0935(?:\u093e)?\s*[:\-']\s*(.*)", t)
    if mr:
        entry["Relation_Type"] = mr.group(1)
        entry["Relative_Name"] = mr.group(2).strip()
    if not entry["Relative_Name"]:
        hr = re.search(r"(\u092a\u093f\u0924\u093e|\u092a\u0924\u093f|\u092e\u093e\u0924\u093e|\u092a\u0924\u094d\u0928\u0940|\u0905\u0928\u094d\u092f)\s*(?:\u0915\u093e|\u0915\u0940|\u0915\u0947)?\s*\u0928\u093e\u092e\s*[:\-']\s*(.*)", t)
        if hr:
            entry["Relation_Type"] = hr.group(1).strip()
            entry["Relative_Name"] = hr.group(2).strip()

    mh2 = re.search(r"(?:\u0918\u0930 \u0915\u094d\u0930\u092e\u093e\u0902\u0915|UX PATH).*?[:\-']\s*(.*)", t, re.IGNORECASE)
    if mh2:
        entry["House_No"] = mh2.group(1).strip()
    if not entry["House_No"]:
        hh = re.search(r"(?:\u092e\u0915\u093e\u0928|Wart|Fart|makan|den)\s*(?:\u0938\u0902\u0916\u094d\u092f\u093e|sankhya|number)?\s*[:\-']?\s*(.*)", t, re.IGNORECASE)
        if hh:
            entry["House_No"] = hh.group(1).strip()

    ma = re.search(r"\u0935\u092f\s*[:\-']\s*(\d+)", t) or re.search(r"\u0906\u092f\u0941\s*[:\-']\s*(\d+)", t)
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


# ------------------------------------------------------- discovery
def discover_pdfs(input_root, district=None, ac=None, max_pdfs=0):
    root = Path(input_root)
    if root.is_file() and root.suffix.lower() == ".pdf":
        return [root]
    pdfs = sorted(root.rglob("*.pdf"))
    if district:
        keys = [district] if isinstance(district, str) else district
        pdfs = [p for p in pdfs if any(k.lower() in str(p).lower() for k in keys)]
    if ac:
        keys = [ac] if isinstance(ac, str) else ac
        pdfs = [p for p in pdfs if any(k.lower() in str(p).lower() for k in keys)]
    if max_pdfs and max_pdfs > 0:
        pdfs = pdfs[:max_pdfs]
    return pdfs


def load_state(state_path):
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"completed": {}}


def save_state(state_path, state):
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(state_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, state_path)


# ------------------------------------------------------- per-PDF run
def process_pdf(pdf_path, input_root, output_root, args, logger, state):
    in_root = Path(input_root)
    if in_root.is_file():
        rel = Path(Path(pdf_path).name)
    else:
        rel = Path(pdf_path).relative_to(in_root)
    out_dir = Path(output_root) / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem
    txt_path = out_dir / (stem + ".txt")
    xlsx_path = out_dir / (stem + ".xlsx")

    csv_path = out_dir / (stem + ".csv")
    table_path = csv_path if getattr(args, "no_xlsx", False) else xlsx_path

    if args.resume:
        done = state.get("completed", {}).get(str(rel))
        if done and table_path.exists():
            try:
                if table_path.stat().st_mtime >= Path(pdf_path).stat().st_mtime:
                    logger.info("SKIP (resume): %s -> %s", rel, table_path.name)
                    return "skipped"
            except OSError:
                pass
        elif table_path.exists() and not done:
            logger.info("SKIP (resume, table exists): %s", rel)
            return "skipped"

    t0 = time.time()
    logger.info("START %s (%s MB)", rel, round(Path(pdf_path).stat().st_size / 1e6, 1))
    try:
        doc = _pdf_lib.open(str(pdf_path))
    except Exception as e:
        logger.error("Cannot open %s: %s", rel, e)
        return "error"
    n_pages = doc.page_count
    s, e = max(1, args.start_page), args.end_page if args.end_page > 0 else n_pages
    e = min(e, n_pages)
    if args.pages > 0:
        e = min(e, s + args.pages - 1)
    logger.info("%s: pages %d-%d of %d, dpi=%d", stem, s, e, n_pages, args.dpi)

    img_dir = None
    if args.keep_images:
        img_dir = out_dir / (stem + "_pages")
        img_dir.mkdir(parents=True, exist_ok=True)

    records, n_blocks, n_pages_ok = [], 0, 0
    zoom = args.dpi / 72.0
    with open(txt_path, "w", encoding="utf-8") as fout:
        for pno in range(s, e + 1):
            try:
                page = doc.load_page(pno - 1)
                pix = page.get_pixmap(matrix=_pdf_lib.Matrix(zoom, zoom))
                img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning("%s p%d: render failed", stem, pno)
                    continue
                if img_dir:
                    cv2.imwrite(str(img_dir / ("page_%04d.png" % pno)), img)
                boxes = detect_boxes(img)
                p_tag = "%s_p%04d" % (stem, pno)  # zero-padded so p0003 sorts before p0010
                if not boxes:
                    fout.write("\n%s\nFILENAME: %s\n%s\nERROR: No blocks found.\n" % ("=" * 50, p_tag, "=" * 50))
                    continue
                h_i, w_i = img.shape[:2]

                def _ocr_one(b):
                    x, y, w, h = b
                    x0, y0 = max(0, x - args.padding), max(0, y - args.padding)
                    x1, y1 = min(w_i, x + w + args.padding), min(h_i, y + h + args.padding)
                    return ocr_block(img[y0:y1, x0:x1])

                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    ocr_out = list(ex.map(_ocr_one, boxes))
                fout.write("\n%s\nFILENAME: %s\n%s\n" % ("=" * 50, p_tag, "=" * 50))
                for idx, (txt, hint) in enumerate(ocr_out):
                    fout.write("--- BLOCK %d ---\n%s\n%s\n" % (idx + 1, txt, "-" * 30))
                    try:
                        rec = parse_block(txt, p_tag, epic_hint=hint)
                    except Exception as exb:
                        logger.warning("%s p%d block %d parse failed: %s", stem, pno, idx + 1, exb)
                        rec = None
                    if rec:
                        records.append(rec)
                n_blocks += len(boxes)
                n_pages_ok += 1
                if n_pages_ok % 5 == 0:
                    fout.flush()
                    logger.info("%s: %d/%d pages, %d blocks", stem, n_pages_ok, e - s + 1, n_blocks)
            except Exception as ex2:
                logger.exception("%s p%d failed: %s", stem, pno, ex2)
    doc.close()

    if not records:
        logger.error("%s: no records extracted", stem)
        return "error"
    df, before = finalize_dataframe(records)
    try:
        csv_path = write_table(df, xlsx_path, logger, stem,
                               write_csv=not args.no_csv, write_xlsx=not args.no_xlsx)
    except PermissionError:
        logger.error("CLOSE %s in Excel and re-run with --resume (file locked). TXT kept; table not written.", table_path.name)
        return "error"
    dt = time.time() - t0
    vid_fill = int(df["Voter_ID"].notna().sum())
    made = [n for n, on in (("xlsx", not args.no_xlsx), ("csv", not args.no_csv)) if on]
    logger.info("DONE %s: %d pages, %d blocks -> %d rows (%d dedup), %d IDs, %.1fs -> %s (%s)",
                stem, n_pages_ok, n_blocks, len(df), before - len(df),
                vid_fill, dt, table_path.name, "+".join(made))
    state.setdefault("completed", {})[str(rel)] = {
        "table": str(table_path), "rows": int(len(df)), "ids": vid_fill,
        "pages": n_pages_ok, "seconds": round(dt, 1), "ts": time.time()}
    return "done"

# Raw OCR text lives ONLY in the per-PDF .txt (audit trail). The Excel/CSV
# carry structured fields only, so the files stay clean for analysis/ML.
EXCEL_COLS = ["Source_File", "Serial_No", "Voter_ID", "Name", "Relation_Type",
              "Relative_Name", "Age", "Gender", "House_No", "Part_Section"]
ILLEGAL_XLSX_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_cell(v):
    """Strip control chars illegal in OOXML (the Excel 'repair' popup) + trim."""
    if isinstance(v, str):
        return ILLEGAL_XLSX_RE.sub("", v).strip()
    return v


def finalize_dataframe(records):
    """Dedup + sanitize + type columns for a model-ready table."""
    df = pd.DataFrame(records)
    for c in EXCEL_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[EXCEL_COLS]
    before = len(df)
    df.drop_duplicates(inplace=True)
    for c in ["Source_File", "Voter_ID", "Name", "Relation_Type",
              "Relative_Name", "Gender", "House_No", "Part_Section"]:
        df[c] = df[c].apply(clean_cell).replace("", pd.NA).astype("string")
    # a "name" with no actual letter is OCR junk (e.g. "|" or "---") -> treat as empty
    # NB: plain str.isalpha loop, NOT a regex: \p{L} raises under stdlib re
    # (object dtype) while [^\W\d_] is ASCII-only under PyArrow compute -
    # backend roulette. isalpha() is Unicode-correct on every backend.
    has_letter = df["Name"].fillna("").map(lambda s: any(ch.isalpha() for ch in s))
    df.loc[~has_letter, "Name"] = pd.NA
    # drop AFTER cleaning so junk-only rows die (they slipped through before)
    df = df[~((df["Name"].isna()) & (df["Voter_ID"].isna()))]
    has_id = df["Voter_ID"].notna()
    df = pd.concat([df[has_id].drop_duplicates(subset=["Voter_ID"]),
                    df[~has_id].drop_duplicates(subset=["Name", "Relative_Name", "Age"])],
                   ignore_index=True)
    for c in ["Serial_No", "Age"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df = df.sort_values(["Source_File", "Serial_No"]).reset_index(drop=True)
    return df, before


def write_table(df, xlsx_path, logger, stem="", write_csv=True, write_xlsx=True):
    """Write XLSX (+CSV twin for ML) with filter/freeze/widths. Never raises."""
    csv_path = Path(str(xlsx_path)).with_suffix(".csv")
    if write_csv:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if not write_xlsx:
        try:
            if Path(xlsx_path).exists():
                Path(xlsx_path).unlink()
        except OSError:
            pass
        return csv_path
    df.to_excel(xlsx_path, index=False)
    try:
        from openpyxl import load_workbook
        wb = load_workbook(xlsx_path)
        ws = wb.active
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        widths = {"Source_File": 22, "Serial_No": 10, "Voter_ID": 20, "Name": 24,
                  "Relation_Type": 12, "Relative_Name": 24, "Age": 7, "Gender": 9,
                  "House_No": 14, "Part_Section": 13}
        for idx, col in enumerate(ws.columns, 1):
            name = EXCEL_COLS[idx - 1] if idx - 1 < len(EXCEL_COLS) else ""
            ws.column_dimensions[col[0].column_letter].width = widths.get(name, 15)
        wb.save(xlsx_path)
    except Exception as e:
        if logger:
            logger.warning("%s: excel polish skipped (%s)", stem, e)
    return csv_path


# ------------------------------------------------------- cli
def build_parser():
    p = argparse.ArgumentParser(description="UP voter-roll OCR pipeline (PDF->TXT->XLSX).")
    p.add_argument("--input", required=True, help="PDF file or root dir (e.g. D:\\Election_Download\\Uttar_Pradesh)")
    p.add_argument("--output", default="D:\\Election_Output", help="Output root (default D:\\Election_Output)")
    p.add_argument("--district", action="append", default=None, help="Filter by district name (repeatable)")
    p.add_argument("--ac", action="append", default=None, help="Filter by assembly code/name (repeatable)")
    p.add_argument("--max-pdfs", type=int, default=0, help="Process at most N PDFs (0=all)")
    p.add_argument("--pages", type=int, default=0, help="Max pages per PDF, for testing (0=all)")
    p.add_argument("--start-page", type=int, default=1)
    p.add_argument("--end-page", type=int, default=0, help="0 = last page")
    p.add_argument("--dpi", type=int, default=216)
    p.add_argument("--padding", type=int, default=15)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true", help="Skip PDFs whose XLSX is already fresh")
    p.add_argument("--keep-images", action="store_true", help="Save page PNGs (huge; default streams in RAM)")
    p.add_argument("--no-csv", action="store_true", help="Skip CSV output (XLSX only)")
    p.add_argument("--no-xlsx", action="store_true", help="Skip XLSX output (CSV only)")
    p.add_argument("--dry-run", action="store_true", help="List PDFs only, process nothing")
    p.add_argument("--log-file", default=None)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.log_file or str(out_root / "pipeline.log"), args.verbose)
    logger.info("Input=%s Output=%s resume=%s dpi=%d workers=%d",
                args.input, args.output, args.resume, args.dpi, args.workers)
    if not ensure_tesseract(logger):
        return 2
    pdfs = discover_pdfs(args.input, args.district, args.ac, args.max_pdfs)
    logger.info("Found %d PDFs", len(pdfs))
    if args.dry_run:
        for p in pdfs[:50]:
            logger.info("  %s", p)
        if len(pdfs) > 50:
            logger.info("  ... +%d more", len(pdfs) - 50)
        return 0
    if not pdfs:
        logger.error("No PDFs under %s", args.input)
        return 1
    # throughput warning for TB-scale runs
    if len(pdfs) > 100:
        logger.warning("%d PDFs: at ~15s/page single-machine this is a multi-week job. "
                       "Start with --district X --max-pdfs 5 --pages 3 to validate.", len(pdfs))
    state_path = out_root / "state.json"
    state = load_state(state_path)
    stats = {"done": 0, "skipped": 0, "error": 0}
    for i, pdf in enumerate(pdfs, 1):
        logger.info("[%d/%d] %s", i, len(pdfs), pdf)
        try:
            r = process_pdf(str(pdf), args.input, str(out_root), args, logger, state)
            stats[r] = stats.get(r, 0) + 1
        except KeyboardInterrupt:
            logger.warning("Interrupted; progress saved. Re-run with --resume.")
            save_state(state_path, state)
            return 130
        except Exception:
            logger.exception("Fatal on %s", pdf)
            stats["error"] += 1
        save_state(state_path, state)  # checkpoint after every PDF
    logger.info("FINISHED: %s", stats)
    return 0 if stats.get("error", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
