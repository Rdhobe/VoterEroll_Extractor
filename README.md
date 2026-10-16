# OCR to Text Conversion – Voter List Extraction Pipeline

Converts scanned/printed **voter-list PDFs** into structured tables with one row per voter.
Started with the **Marathi** Pune Prabhag-29 draft roll (6,905 pages); now also handles
**Hindi (UP)** rolls – e.g. `D:\Election_Download\Uttar_Pradesh` (75 districts, ~1.3 TB).

```
PDF  ──►  OCR text file  ──►  Excel + CSV
  └─ pages rendered in RAM (no PNG cache) ─┘
  single command: scripts/main.py (OpenCV + Tesseract + regex + pandas)
```

Legacy three-script flow (`main_pdf_to_img.py` → `img_to_txt_*.py` →
`cnn_pipeline_folder_with_gpu.py`) still exists under `scripts/` but is **superseded**
by `scripts/main.py`. Old notes about it are kept at the bottom of this file.

---

## 1. What is in this folder

| Path | Purpose |
|---|---|
| `scripts/main.py` | **Unified pipeline (use this)** – PDF(s) → TXT + XLSX + CSV, multi-file, `--resume`, logging |
| `scripts/main_pdf_to_img.py` | Legacy Stage 1 – render PDF pages to PNG images |
| `scripts/img_to_txt_14_to_6905.py` | Legacy Stage 2 – box detection + OCR to a text file |
| `scripts/cnn_pipeline_folder_with_gpu.py` | Legacy Stage 3 – parse OCR text into Excel (now bilingual mar/hin) |
| `docs/README_OCR_PIPELINE.md` | Technical overview of the pipeline |
| `docs/IMG_TO_TXT_EXPLAINED.md` | Deep dive on the vision/OCR stage + layout-adaptation guide |
| `requirements.txt` | Python dependencies |
| `sample/` | 3 example page images, the OCR text produced from them, and the resulting Excel |
| `data/01_input/DraftList_Ward_29.pdf` | Original Marathi source roll (25 MB) |
| `data/02_ocr_text/combined_voter_data_*.txt` | Raw legacy Stage-2 output, 4 batch files (see note below) |
| `data/03_parsed/Final_Voter_List_14_6905_with_gpu_v6.xlsx` | Legacy final Excel (v6) on the 4 text files |

Outputs of `main.py` go to `--output` (default `D:\Election_Output`), mirroring the
input tree: `<output>/<District>/<AC>/<roll-stem>.txt + .xlsx + .csv`, plus
`pipeline.log` and `state.json` (resume checkpoints). Page PNGs are **not** saved
unless `--keep-images` is passed – pages stream through RAM.

**Batch files in `data/02_ocr_text/`** – the file names are the page ranges the batches were
*run* on, and they overlap slightly: `14_3060` actually ends at page 2297 and `5008_6905`
starts at 5008, while `3061_5008` runs to page 5009. The legacy Stage 3 merges all four
and drops identical duplicate rows; pages 2297 and 5008–5009 are read twice.

---

## 2. Prerequisites

1. **Python 3.12+** (3.10+ should work).
2. **Tesseract OCR 5.x** with **Hindi (`hin`)** and **English (`eng`)** data
   (`mar` too if you re-run the old Marathi roll).
   - Windows installer: <https://github.com/UB-Mannheim/tesseract/wiki> – tick the needed languages.
   - `main.py` auto-detects `tesseract.exe` on `PATH` and at the standard spots
     (`%ProgramFiles%\Tesseract-OCR`, `%LOCALAPPDATA%\Programs\Tesseract-OCR`).
   - Check: `tesseract --list-langs` must list `eng` and `hin` (and `mar` for Marathi).
3. Python packages:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

No GPU is needed – everything is CPU (Tesseract + regex + threads).
`pip install -r requirements.txt` also pulls `chrome-ocr` (for the optional
`scripts/chrome_script.py` trial), which additionally needs Google Chrome on
Windows – skip it if you only run `main.py`.
On Windows PowerShell set `$env:PYTHONIOENCODING='utf-8'` (or run `python -X utf8`)
so Devanagari log lines don't crash printing.

---

## 3. How to run (unified)

One command does everything. `--input` accepts a single PDF **or** a folder tree
(district → assembly → PDFs, as on the external drive).

```powershell
# quick test: 2 pages of one roll
python scripts/main.py --input "D:\Election_Download\Uttar_Pradesh\Agra\AC_86_Etmadpur\2026-EROLLGEN-S24-86-SIR-FinalRoll-Revision1-HIN-1-WI.pdf" --output "D:\Election_Output" --pages 2

# one assembly, resumable
python scripts/main.py --input "D:\Election_Download\Uttar_Pradesh" --district Agra --ac AC_86_Etmadpur --output "D:\Election_Output" --resume

# list what would run, without processing
python scripts/main.py --input "D:\Election_Download\Uttar_Pradesh" --district Agra --dry-run --max-pdfs 5
```

| Flag | Meaning | Default |
|---|---|---|
| `--input` | PDF file or root folder | (required) |
| `--output` | Output root | `D:\Election_Output` |
| `--district`, `--ac` | Substring filters, repeatable | all |
| `--max-pdfs` | Process at most N PDFs (`0` = all) | `0` |
| `--pages` | Max pages per PDF – for testing (`0` = all) | `0` |
| `--start-page`, `--end-page` | Page window, 1-based (`0` = last page) | `1`, `0` |
| `--dpi` | Render resolution (216 ≈ old zoom 3.0; 288 max worth trying) | `216` |
| `--padding` | Pixels around each detected box | `15` |
| `--workers` | OCR threads per page | `4` |
| `--resume` | Skip PDFs whose XLSX is already newer than the PDF | off |
| `--keep-images` | Also save page PNGs (very large – off = stream in RAM) | off |
| `--dry-run` | List PDFs only | off |
| `--log-file` | Log destination | `<output>/pipeline.log` |
| `--verbose` | Debug logging | off |

**Resume:** progress checkpoints to `<output>/state.json` after every PDF. If the run
is interrupted, re-run the same command with `--resume` – finished PDFs print
`SKIP (resume)`. If a `.xlsx` is open in Excel when the pipeline writes it, you get a
clear `CLOSE … in Excel and re-run with --resume` error instead of a traceback
(the `.txt` is kept, only the table is retried).

**Scale note:** Agra alone holds 3,932 PDFs; 75 districts ≈ 200k+ PDFs. At ~6 s/page
× ~30 pages ≈ 3 min/PDF on one machine, the full 1.3 TB is a multi-month job –
run assembly-by-assembly, validate the first XLSXs, then split `--district`s across
machines (one output folder each).

### Ensemble v2 – Tesseract + Chrome (`scripts/ensemble_v2.py`)

`scripts/main.py` (Tesseract) and `scripts/chrome_script.py` (Chrome) are frozen
v1 – this self-contained v2 runs **both engines and merges them** field-by-field,
so it can never break the v1 commands. Same three run shapes as `main.py`:

```powershell
# quick test: 2 data pages of one roll
python scripts/ensemble_v2.py --input "D:\Election_Download\Uttar_Pradesh\Agra\AC_86_Etmadpur\2026-EROLLGEN-S24-86-SIR-FinalRoll-Revision1-HIN-1-WI.pdf" --output "D:\Election_Output" --pages 2

# one assembly, resumable (outputs mirror Agra/AC_86_Etmadpur/ under output)
python scripts/ensemble_v2.py --input "D:\Election_Download\Uttar_Pradesh" --district Agra --ac AC_86_Etmadpur --output "D:\Election_Output" --resume

# list what would run, without processing
python scripts/ensemble_v2.py --input "D:\Election_Download\Uttar_Pradesh" --district Agra --dry-run --max-pdfs 5
```

Extra flags over `main.py`: `--chrome-dpi` (default 200), `--ocr-lang`
(default `hin+eng`); `--pages` counts data pages *after* `--skip-pages`
(default `1,2`); resume state in `<output>/ensemble_state.json`.
Single-file outputs sit directly in `--output`:
`<stem>_ensemble.xlsx + .csv` (merged table),
`<stem>_tes.txt` + `<stem>_chrome_raw.md` (per-engine audits). Matching runs
name → fuzzy name → ID → serial; every disagreement lands in `Conflicts`,
single-engine-only rows are kept (`tes-only` / `chrome-only`), agreed rows need
no review.
Measured on the Agra test roll: 806 rows (606 name / 89 tes-only / 78 chrome-only /
29 id / 4 serial). See §6 for the full accuracy table.
Note: Chrome Screen AI is slightly nondeterministic – reruns of
the same page can differ by a character or two.

---

## 4. Accuracy upgrades in `main.py` (vs legacy scripts)

1. **Box IoU dedup (NMS).** The old contour pass returned each voter box twice
   (inner + outer contour → 60 blocks/page instead of 30). Overlapping boxes
   (IoU > 0.4) are now merged → 30 blocks/page, ~2.6× faster (~6 s vs ~16 s/page),
   and the duplicate rows disappear at the source.
2. **Header-strip re-OCR.** The EPIC/Voter-ID line (top ~32% of each box) is OCR'd a
   second time with an English whitelist (`A-Z0-9/`), PSM 7. Whitelist result wins
   when the full-block pass finds nothing strict.
3. **EPIC correction + strict-only IDs.** Common confusions fixed (`2SY→ZSY`,
   `I↔1`, `O↔0`, …) and Devanagari digits normalized (`३→3`). A `Voter_ID` is kept
   **only** if it validates as `AAA1234567` or `UP/…`; otherwise blank (falls back to
   Name+Age dedup). A fabricated ID is worse than none.
4. **Bilingual parsing.** Marathi (`मतदाराचे/वडिलांचे/घर क्रमांक/वय`) and Hindi
   (`नाम/पिता का नाम/मकान संख्या/आयु`, incl. OCR variants `Wart/Fart`) regexes.
5. **Excel-safe output.** Control characters illegal in OOXML are stripped, so Excel
   no longer shows the "problem with some content / recover?" popup.
6. Measured on 2 Hindi voter pages: **30 blocks/page, 59 rows, 46/59 IDs filled,
   100% of filled strict-valid** (old flow on the same data: 60 blocks/page with
   duplicates, ~16% strict IDs plus garbage tokens).

---

## 5. Output columns

Raw OCR text lives **only** in the per-PDF `.txt` (audit trail). The Excel/CSV carry
10 structured columns, model-ready:

| Column | Type | Description |
|---|---|---|
| `Source_File` | string | Roll stem + page that the voter came from |
| `Serial_No` | Int64 | Serial number in the list |
| `Voter_ID` | string | EPIC (`ZSY…`) or old roll no. (`UP/70/355/…`); empty if unreadable |
| `Name` | string | Voter's full name (Hindi/Marathi) |
| `Relation_Type` | string | पिता / पति / माता (Hindi) or वडिलांचे / पतीचे / … (Marathi) |
| `Relative_Name` | string | Name of that relative |
| `Age` | Int64 | Age |
| `Gender` | string | `Male` / `Female` (empty if unreadable) |
| `House_No` | string | House number as printed |
| `Part_Section` | string | Marathi rolls: `209/268/1`; Hindi rolls: empty (no such field) |

Each XLSX has frozen header, autofilter and column widths; the `.csv` twin
(`utf-8-sig`) next to it is what to load for analysis/model training
(`pandas.read_csv` reads it losslessly – Excel is for eyeballing).

---

## 6. Measured accuracy (ground truth: `gijauli_voter_list.md`)

728 hand-mapped voter cards (PDF pp. 3–27, same Agra roll) with per-card OCR text
were parsed to `Serial/Page/EPIC/Name/Relation/House/Age/Gender` and scored
against all three pipelines. Method: normalized-name match (spaces, punctuation,
nukta, anusvara, zero-width and junk-tail fragments like `जाना`/`oe` folded on
*both* sides); every denominator below is GT-valid values only (GT itself is
per-card OCR – 210 strict EPICs, 361 clean age/sex lines – and disclaims 100%).
±0.5 pt wobble comes from scoring-normalization refinements between runs.

| Metric (denominator) | TES `main.py` | CHR `chrome_script.py` | ENS `ensemble_v2.py` |
|---|---|---|---|
| Voter detection (/725 names) | 83.2% (603) | 91.2% (661) | **93.2% (676)** |
| EPIC exact | **72.1%** (101/140) | 23.6% (39/165) | 59.1% (94/159) |
| Relation type | 91.8% (404/440) | 92.6% (461/498) | **96.1%** (472/491) |
| Relative name | 80.0% (352/440) | 84.5% (421/498) | **86.2%** (423/491) |
| House | 35.2% (124/352) | 26.1% (105/403) | **41.3%** (164/397) |
| Age | **83.8%** (171/204) | 81.2% (182/224) | 83.0% (186/224) |
| Gender | **95.5%** (212/222) | 90.5% (220/243) | 94.7% (233/246) |

Ensemble output on the test roll: **806 rows** – 606 name-matched, 89 tes-only,
78 chrome-only, 29 ID-linked, 4 serial-linked; 692 rows carry notes, ~640 hard
conflicts. Deduplicate keys are clean on v1 (TES 1.6%, CHR 0.0% duplicate
page+serial); v2 shows 10.3% from unmerged same-serial leftovers (known gap).

Reading: content OCR is strong everywhere (gender 90–95%, relation ~92–96%);
the war is **linking** – TES box crops localize EPICs but miss cards, chrome
finds cards but mislinks EPICs serially. Target for model-training fields was
all-five >90% (Name/Gender/Age/Relative/Relation): Relation 96.1 ✓, Gender 94.7 ✓,
Name 93.2 ✓, Relative 86.2, Age 83.0 – residuals are both-engines-wrong misreads,
shuffled-page mislinks and GT's own OCR noise (both engines agree against GT in
~18% of age and ~27% of relative agreements). The remaining gap is a **~160-row
review queue**, not more tuning – filter it with:

```python
df = pd.read_excel("<stem>_ensemble.xlsx")
age_q = df[df["Conflicts"].str.contains("age ", na=False)]
reln_q = df[df["Conflicts"].str.contains("Relative_Name|relation==", na=False)]
```

Full campaign history (rules tried, measurements, reversions) is in
`changes_log.md` §19; the evaluator lives outside the repo
(`eval_acc2.py` – GT parse + scoring, denominators = GT-valid only).

---

## 7. Known limitations (please read before using the data)

- **OCR quality.** Even after the fixes, ~20% of rows have no strict `Voter_ID`
  (header line unreadable) and `Serial_No` is found on ~1/3 of Hindi rows – use the
  `.txt` dump to review. Hindi name/age/gender extract at ~95–100% completeness.
- **Residual duplicates.** NMS removes the systematic double-detection, but a few
  near-identical boxes with slightly different OCR text can survive – sort by
  `Voter_ID`/`Name` to spot them.
- **Empty rows** from page headers are dropped automatically (rows with neither
  `Name` nor `Voter_ID`).
- **Layout-specific.** Grid detection assumes the 3-column voter-box layout with the
  photo at top-right. For another layout see "How to Adapt for Different Grids" in
  `docs/IMG_TO_TXT_EXPLAINED.md` (kernel sizes, area/aspect filters, mask, row gap).
- **DPI trade-off.** 216 DPI + 2× box upscale is the sweet spot; 288 DPI may add
  1–3% character accuracy at ~2× cost. Past ~300 DPI there is no gain.
- **"GPU"/"CNN" names are historical.** Nothing uses a GPU or neural net besides
  Tesseract's built-in LSTM. The `torch` import in the legacy Stage-3 script is only
  a cosmetic CUDA check (now optional – the script runs without torch).

### Legacy three-script flow (superseded – reference only)

- *Stage 1* `main_pdf_to_img.py`: hard-coded `PDF_PATH`, `OUTPUT_FOLDER`,
  `START_PAGE/END_PAGE`, `ZOOM_FACTOR`; renders all PNGs up front (GBs of disk).
- *Stage 2* `img_to_txt_14_to_6905.py`: hard-coded `INPUT_FOLDER`, `OUTPUT_FILE`,
  `OCR_CONFIG` (`-l mar+eng`), `PADDING`; ~12 s/page; no box dedup (60 blocks/page).
- *Stage 3* `cnn_pipeline_folder_with_gpu.py`: hard-coded `INPUT_FILES`,
  `OUTPUT_EXCEL_FILE`, `USE_GPU` (thread count only); bilingual since the Hindi
  update; `torch` import now guarded in try/except.
- The word "CNN" in file names is historical – no neural network is used besides
  Tesseract's built-in LSTM.
