# 🗳️ Shakkkti AI: Voter Data Extraction Pipeline

This document explains the technical implementation of the pipeline used to convert
PDF voter lists into structured Excel/CSV data. The production entry point is the
**unified `scripts/main.py`** (PDF → TXT + XLSX + CSV in one command); the original
three-script flow is documented as legacy at the bottom.

---

## 🏗️ Pipeline Overview (`scripts/main.py`)

One process per PDF, streaming pages through RAM (no multi-GB PNG cache):

1.  **Render**: PyMuPDF renders each page at `--dpi` (default 216).
2.  **Detect**: OpenCV morphology finds voter boxes; **IoU-NMS dedups** nested
    double detections (60 → 30 blocks/page).
3.  **OCR**: Tesseract LSTM on each box (`--oem 3 --psm 6 -l hin+eng`, Marathi rolls
    add `mar`), photo area masked out; the EPIC header strip is **re-OCR'd with an
    eng whitelist** (`--psm 7`, `A-Z0-9/`).
4.  **Parse & structure**: bilingual regex + EPIC correction → pandas table →
    per-PDF `.txt` (raw dump) + `.xlsx` + `.csv`, with `state.json` checkpoints.

Multi-PDF discovery (`--district`/`--ac` filters), `--resume` where-left-off, and
file+console logging are built in – see `README.md §3` for the CLI.

---

## 🛤️ Stage 1: PDF to Image (in-memory)

**Core Library**: [PyMuPDF](https://pymupdf.readthedocs.io/)

### Key Technical Details:
- **Resolution**: `--dpi 216` (≈ old `ZOOM_FACTOR 3.0`) – best speed/accuracy trade-off.
  288 DPI is the most ever worth trying (~2× cost for ~1–3% char gain); past ~300 DPI
  Tesseract gains nothing (each box is already upscaled 2× before OCR).
- **Streaming**: `page.get_pixmap()` → `cv2.imdecode` in RAM. PNGs hit disk only with
  `--keep-images` (off by default – the 1.3 TB corpus would otherwise need PBs).
- **Page window**: `--start-page/--end-page/--pages` (e.g. `--pages 2` for smoke tests).

---

## 🔍 Stage 2: Box Detection + OCR

**Core Libraries**: [OpenCV](https://opencv.org/), [pytesseract](https://github.com/madmaze/pytesseract)

### Grid Detection
- **Morphological Operations**: horizontal/vertical `MORPH_RECT` kernels
  (`w//40`, `h//40`) isolate grid lines; close+dilate builds the grid skeleton.
- **Contour Filtering**: keeps boxes 30–150% of `expected_area = (w/3)·(h/10)` with
  aspect ratio 1.5–5.0 (the 3-column × 10-row layout).
- **IoU-NMS dedup (new)**: `RETR_LIST` returns each box twice (inner+outer contour).
  Boxes are sorted by area and any box with IoU > 0.4 against a kept box is dropped.
  Reading order is then row-grouped (`Δy ≤ 50`) and left-to-right sorted.
- **Masking**: the photo area (right 28%, below 25% height) is whited out, plus a
  10 px white border is added so edge-touching glyphs survive thresholding.

### OCR Engine
- **Model Type**: Tesseract OCR (LSTM Engine).
- **Full block**: `--oem 3 --psm 6 -l hin+eng` after 2× cubic upscale + OTSU.
- **Header strip (new)**: top 32% of the box re-OCR'd with
  `--oem 3 --psm 7 -l eng -c tessedit_char_whitelist=ABC…789/` – dramatically better
  EPIC reads. Accepted **only** on strict match (`AAA1234567` or `UP/…`).
- **Throughput**: ~6 s/page (was ~12–16 s before NMS halved the OCR calls).

---

## 📊 Stage 3: Parsing, Correction & Structuring

**Core Libraries**: [Python Re](https://docs.python.org/3/library/re.html), [Pandas](https://pandas.pydata.org/)

### Extraction Logic (bilingual Marathi + Hindi):
- **Serial & Voter ID**: Marathi 3-field header, else Hindi 2-field
  (`332 ZSY439728`, `33 UP/70/355/0000290`); Devanagari digits normalized first.
- **EPIC correction**: `2→Z / 5→S / 1→I / 0→O …` confusion map; result kept **only**
  if it validates – otherwise blank (Name+Age dedup fallback). Never fabricates IDs.
- **Name**: `मतदाराचे/पूर्ण` (mar) or `नाम :` (hin).
- **Relation**: `वडिलांचे/पतीचे/आईचे/इतर + नाव` (mar) or `पिता/पति/माता/पत्नी + (का/की) + नाम` (hin).
- **House**: `घर क्रमांक` (mar) or `मकान संख्या` + OCR variants `Wart/Fart/den` (hin).
- **Age/Gender**: `वय`/`आयु`, `लिंग` → `Male` (`पु/पुरुष/M`) / `Female` (`स्त्री/महिला/F`).
- **Dedup**: exact dupes → drop empty rows → `Voter_ID` dedup → `Name+Relative+Age`
  dedup for ID-less rows.

### Output shaping (model-ready):
- **10 structured columns only** (`Source_File … Part_Section`) – raw OCR text lives
  exclusively in the `.txt`. Types: `Serial_No`/`Age` → nullable `Int64`, text →
  `string` with true missing values, rows sorted by file+serial.
- **OOXML sanitized**: control chars `[\x00-\x08\x0b\x0c\x0e-\x1f]` stripped – Excel
  no longer shows the "problem with some content / recover?" popup.
- **Files**: `.xlsx` (frozen header, autofilter, widths – for humans) + `.csv`
  twin in `utf-8-sig` (for `pandas.read_csv` / training).
- **Threading**: `ThreadPoolExecutor` over boxes within a page (Tesseract releases
  into subprocesses, so threads help on CPU).

---

## 🛠️ Tech Stack Summary

| Stage | Tool | Purpose |
| :--- | :--- | :--- |
| **Render** | PyMuPDF | PDF → image in RAM |
| **Vision** | OpenCV | Grid detection + IoU-NMS + masking |
| **OCR** | Tesseract 5 (LSTM) | Hindi/English (+Marathi) recognition, whitelist header pass |
| **Parsing** | Regex | Bilingual field extraction + EPIC correction |
| **Backend** | Pandas / Openpyxl | Typed tables, XLSX + CSV output |
| **Ops** | logging + state.json | Resume, per-PDF stats, scale-out by district |

---

## 🧬 Ensemble v2 (`scripts/ensemble_v2.py`)

Self-contained merger (v1 files frozen): Tesseract side + Chrome side run on the
same page set (default pp. 3+, `--skip-pages 1,2`), then voters are matched
**name → fuzzy name (≥0.88, blocked on conflicting strict IDs) → strict ID →
(page, serial)**; leftovers survive as `tes-only` / `chrome-only` rows.

Adjudication (a blank never beats a value; conflicts always flagged in
`Conflicts`, hard/soft split in logs, conflicted rows shaded pink):
- Serial/Name/Relation/Age/Gender/House: Tesseract pick on conflict
  (measured: age TES-right 14 vs CHR-right 3 on disagreements)
- EPIC `AAA1234567` conflict: Tesseract (measured 71.6% vs 24.4%)
- `UP/…` conflict: longer (more complete)
- Relative name: dissimilar pairs take the longer read, close pairs chrome
- Age: triple vote (tes full-read, tes age-strip whitelist re-OCR, chrome);
  2/3 majority wins, full split falls back to Tesseract
- Chrome EPIC failing strict validation is dropped + noted, never linked on
- Shared strict ID across rows is detected and flagged
- Name assembly: order-zip primary (correct for both `[N,F]*` and `[Ns,Fs]`
  OCR layouts), positional-chunking fallback with `chunk fallback` flag

Measured on the Agra roll (see `README.md` §6): 806 rows, ENS wins detection
(93.2%) and 4/7 fields; TES stays EPIC king (71.6%). Chrome Screen AI is mildly
nondeterministic across reruns – same page can differ by a character.

---

## Legacy three-script flow (superseded, reference only)

- `main_pdf_to_img.py` – all PNGs rendered up front (`ZOOM_FACTOR 3.0`).
- `img_to_txt_14_to_6905.py` – hard-coded folders, no NMS (60 blocks/page), single
  `mar+eng` OCR pass, progress flushed every 10 pages.
- `cnn_pipeline_folder_with_gpu.py` – regex parse of merged TXT batches; "GPU" only
  changed thread count; `torch` import now optional.

> [!NOTE]
> The "CNN"/"GPU" suffixes historically referred to an intended neural classifier /
> CUDA run, but production uses Computer Vision grid detection + Tesseract LSTM on CPU.
