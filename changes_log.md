# Change Log – Shakkti OCR to Text Conversion

All changes made while adapting the Marathi prototype to Hindi (UP) rolls and
scaling it to bulk runs on `D:\Election_Download\Uttar_Pradesh`. Dates in 2026-10-16 session.

---

## 1. Environment – Tesseract setup

- Pointed the shell at the real install: `C:\Users\sec-gen\AppData\Local\Programs\Tesseract-OCR`.
- Session: `$env:Path += ";…\Tesseract-OCR"`, `$env:TESSDATA_PREFIX = "…\tessdata"`.
- Permanent: `setx PATH …` / `setx TESSDATA_PREFIX …`; verified with
  `tesseract --list-langs` (`eng`, `hin`, `mar` present).
- `scripts/main.py` now auto-detects `tesseract.exe` (PATH + both install paths) and
  logs its version at startup, so this step is optional going forward.

## 2. `scripts/main_pdf_to_img.py` – wrong PDF path

- `PDF_PATH` pointed at a filename in CWD that didn't exist; the real file was
  `data/01_input/2026-EROLLGEN-S24-86-SIR-FinalRoll-Revision1-HIN-1-WI.pdf`.
- Fixed to `PDF_PATH = "data/01_input/2026-EROLLGEN-S24-86-SIR-FinalRoll-Revision1-HIN-1-WI.pdf"`.
- Result: 15 PNGs (`page_0014–0028.png`) from the 28-page roll (pages 1–13 are covers).

## 3. `scripts/img_to_txt_14_to_6905.py` – Hindi retarget

- `INPUT_FOLDER`: `total_pages_5008_6905` → `total_pages`.
- `OUTPUT_FILE`: → `combined_voter_data_HIN_WI.txt`.
- `OCR_CONFIG`: `-l mar+eng` → `-l hin+eng` (file is a HIN roll).
- Result: 15 pages in ~243 s → 210 KB text. Found the systematic bug below.

## 4. `scripts/cnn_pipeline_folder_with_gpu.py` – bilingual + robustness

- **Bilingual parsing**: Hindi fallbacks added for name (`नाम`), relation
  (`पिता/पति/माता का नाम`), house (`मकान संख्या` + OCR variants `Wart/Fart/den`),
  age (`आयु`), gender (`पुरुष/महिला`).
- **Header regex**: Marathi 3-field pattern never matched Hindi's 2-field header
  (`332 ZSY439728`, `33 UP/70/355/0000290`) → all `Serial_No`/`Voter_ID` empty.
  Added Hindi 2-field + EPIC-only fallbacks. `Voter_ID` fill: 0 → 640/809.
- **Smarter dedup**: exact-row `drop_duplicates` → + drop empty rows + `Voter_ID`
  dedup + `Name+Relative+Age` fallback. 815 → 539 rows.
- **torch optional**: `import torch` (cosmetic CUDA check, broken DLL on this machine)
  wrapped in try/except; parser runs without it.
- First Hindi result: Name 99.8%, Age 99.8%, Gender 99.1%, Voter_ID 77.4%,
  strict EPIC only ~16–20% → motivated the accuracy pass (§5).

## 5. Accuracy upgrades (in `scripts/main.py`)

| # | Change | Before → After (2 voter pages) |
|---|---|---|
| 1 | **IoU-NMS box dedup** (`nms_boxes`, threshold 0.4): `RETR_LIST` returned every box twice | 60 → **30 blocks/page**; ~16 → **~6 s/page** (2.6× faster); dup rows gone at source |
| 2 | **Header-strip re-OCR**: top 32% of box, `--psm 7 -l eng` + `A-Z0-9/` whitelist; wins when full pass has no strict ID | garbage IDs (`BSCTWT7055`, `221439728`) eliminated |
| 3 | **EPIC correction** (`correct_epic`): `2→Z/5→S/1→I/0→O…` map, Devanagari digit normalization | `2SY3337284` → `ZSY3337284` |
| 4 | **Strict-only IDs**: value kept only if `AAA1234567` or `UP/…`, else blank (Name+Age dedup fallback) | 57 mixed → **46/59 filled, 100% strict-valid** (32 EPIC + 14 UP/) |
| 5 | **Preprocessing**: +10 px white border (edge glyphs), photo mask kept, per-block parse in try/except (one bad box never kills a page) | no more whole-page losses from single-block exceptions |

## 6. Bulk run – unified `scripts/main.py` (new file, replaces 3-script flow)

- **One command** for PDF(s) → TXT + XLSX + CSV: file or folder `--input`
  (district → AC → PDFs), outputs mirror the tree under `--output`.
- **Streaming render**: pages via PyMuPDF → OpenCV in RAM; `--keep-images` off by
  default (a PNG cache of 1.3 TB input would need PBs).
- **Multi-file**: `--district/--ac` substring filters (repeatable), `--max-pdfs`,
  `--pages/--start-page/--end-page` windows, `--dry-run` listing, `--dpi/--padding/--workers`.
- **`--resume`**: `state.json` checkpoint after every PDF + fresh-XLSX check;
  crash/restart skips finished PDFs (`SKIP (resume)`).
- **Logger**: timestamped console + `<output>/pipeline.log` (`--log-file/--verbose`),
  per-PDF stats (pages/blocks/rows/IDs/seconds), `[i/N]` progress, 100+-PDF scale warning.
- **Bugfixes found in testing**: single-file `rel.parent` crash; `UPROLL_RE.group(1)`
  `IndexError` (no capture group); per-block try/except; `PermissionError` on locked
  XLSX → clear "close in Excel, re-run with --resume" message.

## 7. Model-ready tables (in `scripts/main.py`)

- **Raw dump out of Excel**: `Raw_Text_Dump` lives only in the per-PDF `.txt`; tables
  carry 10 columns (`Source_File … Part_Section`).
- **Typed**: `Serial_No`/`Age` → nullable `Int64` (no `332.0`), text → `string` with
  true missing values, rows sorted by file+serial.
- **Excel-safe**: `[\x00-\x08\x0b\x0c\x0e-\x1f]` stripped → the "problem with some
  content / recover?" popup is gone.
- **XLSX polish** (openpyxl, best-effort): frozen header, autofilter, column widths.
- **CSV twin** (`utf-8-sig`) next to every XLSX – load this for training/analysis.

## 8. Docs

- `README.md`: rewritten around `main.py` (CLI table, resume, scale notes, new outputs,
  accuracy table); legacy 3-script flow moved to a reference section.
- `docs/README_OCR_PIPELINE.md`: 4-stage technical overview (render → NMS detect →
  two-pass OCR → bilingual parse/structure), updated stack table.
- `docs/IMG_TO_TXT_EXPLAINED.md`: now documents `detect_boxes/preprocess_roi/ocr_block`,
  adaptation guide retargeted at `main.py`; legacy script notes kept.
- This file (`changes_log.md`).

## 18. Ground-truth accuracy table (`gijauli_voter_list.md`, 728 cards)

- GT parsed to `gt_parsed.csv` (page/serial/EPIC/name/relation/house/age/gender;
  GT itself is per-card OCR: only 210 strict EPICs, 361 clean age/sex lines).
- Method: (a) detection = GT name found anywhere in output; (b) field accuracy
  on normalized-name-matched voters (denominators = GT-valid values only).

| Metric (denominator) | TES (main.py) | CHR (chrome) | ENS (v2) |
|---|---|---|---|
| Voter detection (/725 names) | 82.5% (598) | 90.2% (654) | **92.4% (670)** |
| EPIC exact (/~150 GT EPICs) | **71.6%** | 24.4% | 58.8% |
| Relation type | 91.8% | 92.5% | **96.3%** |
| Relative name | 75.1% | 78.1% | **78.4%** |
| House | 34.9% | 25.9% | **40.8%** |
| Age | **83.8%** | 81.2% | 83.4% |
| Gender | **95.5%** | 90.5% | 95.1% |
| Duplicate (page,serial) keys | 1.6% | 0.0% | 10.3% (fix: merge leftovers sharing serial) |

- Reading: content OCR is strong everywhere (gender 90–95%, relation ~92–96%);
  the war is **linking**: TES box-crops localize EPICs (71.6%) but miss cards;
  chrome finds cards (90%) but mislinks EPICs serially (24.4%).
- Verdict: ENS wins detection + 4/7 fields; TES stays EPIC king. Known v2 gap:
  10.3% duplicate serial keys from unmerged same-serial leftovers.

## 17. Ensemble v2 (`scripts/ensemble_v2.py`, v1 frozen)

- New self-contained file: Tesseract (render→NMS→box OCR→parse) + Chrome
  (page OCR→chunked parse) + merger on a **single PDF**. `main.py` and
  `chrome_script.py` untouched – v1 fallback intact.
- Matching: normalized name (same page; dupes disambiguated by serial/id/age)
  → fuzzy names ≥0.88 (blocked when both strict IDs disagree) → strict ID →
  (page, serial); leftovers kept as `tes-only` / `chrome-only`.
- Adjudication: blank never beats a value; conflicts flagged, never guessed –
  EPIC conflicts prefer Tesseract (box-crop), UP/ conflicts prefer longer,
  everything else prefers Tesseract; invalid chrome IDs dropped + noted.
- Hard-vs-soft conflict split in the log; shared-ID-across-rows detector.
- Full Agra roll: **804 rows** (604 name / 87 tes-only / 77 chrome-only / 32 id /
  4 serial), conflicts pinpointed (e.g. photo card: ID conflict flagged, tes pick
  `ZSY3999304` vs truth `ZSY3999364`; रघुराज linked via serial with name flagged).
- Caveats found: Chrome Screen AI is nondeterministic across reruns
  (रघुराज→रघुराज same page); close-but-gated pairs (0.9+ with conflicting IDs)
  stay split by design; EPIC assignment still needs image review on conflicts.

## 16. Card-assembly fix + head-to-head accuracy comparison

- Bug (your रघुराज report): index-zipping field lists stitched neighbours'
  fragments into one card (`पति = self`, wrong sex). Fixed by positional
  chunking (cut page at Name lines, parse within) + validators (relation==name
  dropped, पति+पुरुष flagged). Self-links now 0; p3 fully internally consistent.
- Also fixed en route: `re \\p{L}` crashes stdlib `re` (object dtype) – replaced
  with backend-proof `str.isalpha()` loop in `main.py`.
- Both full runs (your exact commands): TES 717 rows / 512 IDs, CHR 717 rows /
  692 EPICs. Name-matched agreement (469 shared voters): Gender **98.4%**,
  Relative **88.5%**, Age **80.4%**, House **24.6%**, Voter_ID **26.5%**.
- Diagnosis: content OCR is good on both; **ID↔voter linking is the weak link**
  on both sides (chrome's leftover-fill forces EPICs onto nearly every card;
  TES strict-only blanks instead). Houses show systematic `7xx` (TES) vs `1xx`
  (chrome) leading-digit split – needs image audit to crown the winner.
- Refined ensemble rule: trust names/gender/relations from either engine;
  adjudicate IDs + houses per-field (agree = confident, else review queue).

## 15. Blocked MD + reusable extraction + quiet-logs fix

- `extract_pdf_to_md()` factored out (importable; returns paged sections).
  New root `test_chrome_md.py`: page-3-only MD trial → `test_page3.md` + preview.
- `--output` MD is now **one block per voter** (Serial | EPIC, नाम, relation,
  मकान, आयु | लिंग, Check) under `--- PAGE n ---` + meta header; photo-caption
  noise (`फोटो उपलब्ध है`) dropped; raw shuffled MD available via `--keep-raw-md`.
- Excel is built straight from the cards (no re-parsing); heading-mark stripping
  restored in-parser (parity: 687 rows, was 656 without it); stray `नाम:` inside
  relative names stripped.
- Silence done right: stderr-only fd redirect + pymupdf `set_messages(devnull)`
  + fitz pre-import (earlier fd-1 redirect crashed pymupdf's print: WinError 1).
- Page-3 block 1 vs your photo: Serial/नाम/पति/मकान/आयु-लिंग all exact;
  EPIC differs (Chrome `ZSY3309846` vs card `ZSY3999364`; Tesseract read
  `ZSY3999304`) – ensemble check recommended per §11.

## 14. Reusable extraction + root-level MD trial

- `scripts/chrome_script.py`: stage 1 factored into importable
  `extract_pdf_to_md(pdf, md, pages_spec, skip_pages, dpi, logger, verbose)`
  (same behavior; CLI `main()` now just calls it, then stage 2).
- New `test_chrome_md.py` in project root (outside `scripts/`): MD-only trial,
  Agra WI-1 page 3 → `D:\Election_Output\test_page3.md` + 1500-char preview.
  Page 3 reads clean (serials/EPICs/cards); CLI re-verified (`--pages 3,4` → 59 rows).

## 13. Chrome MD → Excel merge (in `scripts/chrome_script.py`)

- Merged `scripts/roll_to_excel.py` (kept as reference) into `chrome_script.py`:
  one run now does PDF → tagged Markdown → voter workbook
  (`--excel PATH`, default `<md-stem>.xlsx`; `--no-excel` for MD only).
- Page convention wired in: `--skip-pages` default `1,2` (p1 = summary, p2 = images
  only, data from p3). Full WI-1 roll: 26 pages OCR'd, p1–p2 never rendered.
- Crash fix: pages with cards but no serial lines raised `KeyError: 'Serial'`
  (same bug exists in standalone `roll_to_excel.py`) – excess cards now keep
  `Serial=None` + "serial missing" flag, sort uses `na_position="last"`.
- Serial-stray filter: bare-number lines pick up junk (notably a lone `1` from
  `भाग संख्या :: 1` splits). Rule = keep the largest gap≤5 cluster
  (clean pages: single cluster, behavior unchanged; p25: dropped only `[1]`,
  kept true `658–687` run with its genuine 669 gap).
- Full-roll result: **687 rows, 671 EPICs**, pp.3–27 (`D:\Election_Output\chrome_roll.*`).
  Supplement pp.25–28 links verified structurally (no serial-1 hordes); shuffled
  card↔serial/EPIC links stay flagged in `Check` (677/687) – review queue by design.
- Columns: Page, Part No, Section, Serial No, EPIC ID, Name, Relation,
  Relative Name, House No, Age, Gender, Check (flagged rows shaded), Assembly.
- Post-merge crash fix: redirecting stdout (fd 1) at OS level broke pymupdf's
  internal `print(flush=1)` inside chrome_ocr's lazy `import fitz`
  (`OSError: [WinError 1]`, killed the run). Redirect is now stderr-only (where
  all DLL spam goes, proven safe); the one-time shim notice is pre-fired under a
  Python-level redirect at startup so the in-library import stays silent.
  Verified on the exact failing full-roll command: 687 rows again.

## 12. Repo self-containment (clone-and-go)

- Root cause of the venv error: `chrome-ocr` had been installed into system Python,
  not the project `.venv`. Installed into `.venv` (0.1.0 verified).
- `requirements.txt`: added `chrome-ocr[pdf]@git+https://github.com/ayismas/chrome-ocr.git`
  (verified resolvable via `pip install --dry-run -r requirements.txt`); `torch`
  commented out (unused – imports guarded, saves ~800 MB on fresh installs).
- `scripts/main.py`: Tesseract candidates no longer hardcode a username –
  built from `%ProgramFiles%` / `%LOCALAPPDATA%`.
- `README.md`: install spots generalized; Chrome requirement noted for the optional
  chrome trial. Grep-verified: no `sec-gen`/temp paths remain in scripts or docs
  (only this log's history).
- `scripts/chrome_script.py`: fixed `n_chars` overwrite bug (log showed last page's
  count instead of the total). Temp clone outside the repo deleted.
- Verified under `.venv`: `chrome_script.py --pages 14` (1 page, 3971 chars) and
  `main.py --dry-run --district Agra --max-pdfs 2` (finds PDFs, Tesseract detected).

## 11. Chrome Screen AI trial (`scripts/chrome_script.py`, chrome-ocr @ ayismas)

- Installed `chrome-ocr` (pip from git clone, `[pdf]` extra). Machine already had
  Chrome 155 + `chrome_screen_ai.dll` (153.3) – `doctor` reports ready.
- New `scripts/chrome_script.py`: `--input/--output/--pages/--dpi/--log-file/--verbose`,
  per-page `--- PAGE n ---` tagged Markdown, shared engine, GLOG noise reduced.
- Trial on Agra WI-1 pp.14–15 → `D:\Election_Output\chrome_trial_p14-15.md`:
  2 pages in ~5 s (**2.5 s/page vs ~6 s Tesseract**), Hindi text cleaner
  (correct `मकान संख्या`, no `Wart/Fart` garbage).
- ID comparison (same 2 pages): Chrome **60 unique strict IDs** (44 EPIC + 16 UP)
  vs Tesseract pipeline **45** – overlap only **34**. Each engine misreads different
  records (Tesseract truncated several UP numbers Chrome got in full, and vice versa).
- Caveats: whole-page output (serial/ID↔voter by order only), `फोटो उपलब्ध है`
  caption noise, ICU conjunct warnings on stderr (harmless).
- Verdict: **ensemble, don't replace** – agree = high confidence, conflict = review
  queue. Parser for chrome markdown → table is the next step if you want it.
- Log hygiene: the DLL's absl logs ignore `GLOG_minloglevel` and chrome-ocr's lazy
  `import fitz` prints to stdout, so `chrome_script.py` wraps engine/OCR calls in a
  `quiet_native()` fd 1+2 redirect (bypass with `--verbose`).

## 10. Table-order + junk-row fixes (this round, `scripts/main.py`)

- **Pages "missing" from Excel were a sort illusion**: `Source_File` used `_p3…_p10`
  and text sort puts `_p10` before `_p3`. Tags are now zero-padded (`_p0003`) –
  verified: sheet runs `_p0003 → _p0027`, `sorted() == True`.
- **Junk rows** (punctuation-only "names" like `|`): empty-row drop now runs *after*
  cleaning, and names with no real letter are nulled first. Verified: 0 letter-less
  names, 0 fully-empty rows in the 717-row output.
- **`\p{L}` landmine found by testing** (and fixed twice): stdlib `re` (object
  dtype) *raises* on `\p`, while PyArrow compute treats `[^\W\d_]` as ASCII-only
  (19/716 Devanagari matches) – the two backends need opposite patterns.
  Final fix: plain `str.isalpha()` loop, Unicode-correct on every backend.
- **New flags**: `--no-csv` / `--no-xlsx` (CSV = training input, XLSX = human view);
  resume-skip and DONE log follow whichever table is primary.
- Full-roll result: 27 pages w/ blocks, 736 blocks → **717 rows, 512 strict-valid IDs
  (354 EPIC + 158 UP/)**, 0 illegal-control-char cells, CSV shape == XLSX shape.

## 9. Scale notes / open items

- Drive layout confirmed: 75 districts → `AC_*` folders → `2026-EROLLGEN-…-WI.pdf`
  parts (Agra: 9 ACs, 3,932 PDFs). Output root `D:\Election_Output` created.
- Estimate: ~6 s/page × ~30 pages ≈ 3 min/PDF → full corpus is multi-month on one
  machine. Run per-assembly, validate, then split `--district`s across machines.
- DPI: 216 + 2× box upscale is the sweet spot; 288 may add 1–3% chars at ~2× cost;
  nothing to gain past ~300 DPI.
- Residual: a few near-duplicate boxes with differing OCR text survive dedup; ~20% of
  rows lack strict IDs (unreadable headers – see `.txt`); test artifacts cleared from
  `D:\Election_Output` (fresh `state.json` on next run).
