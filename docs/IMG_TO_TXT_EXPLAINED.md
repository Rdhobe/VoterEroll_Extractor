# 📖 Deep Dive: Box Detection + OCR Explained

Covers the vision/OCR stage as implemented in **`scripts/main.py`**
(`detect_boxes` / `preprocess_roi` / `ocr_block`). The legacy script
`img_to_txt_14_to_6905.py` (`extract_grid_blocks`, `preprocess_roi_raw`,
`process_batch`) used the same morphology without the NMS/whitelist upgrades –
notes on it are at the bottom.

---

## 🏗️ 1. Configuration

`main.py` takes configuration from CLI flags, not file constants:

| Flag | Meaning | Default |
|---|---|---|
| `--dpi` | Render resolution | `216` |
| `--padding` | Margin around each detected box | `15` |
| `--workers` | OCR threads per page | `4` |
| `--keep-images` | Save page PNGs (default: RAM only) | off |

Tesseract configs used internally:
- Full block: `--oem 3 --psm 6 -l hin+eng` (`--oem 3` = LSTM engine,
  `--psm 6` = uniform text block, `-l` = Hindi+English; add `mar` for Marathi rolls).
- Header strip: `--oem 3 --psm 7 -l eng -c tessedit_char_whitelist=ABC…789/`
  (`--psm 7` = single text line + strict character whitelist → clean EPIC reads).

---

## 🔢 2. Ordering Logic

### Reading order (row grouping)
Pages return contours in random order, so boxes are grouped into rows: boxes whose
`y` differs by ≤ 50 px belong to one row; rows are emitted top-to-bottom with boxes
left-to-right inside each row. This keeps the `.txt` in visual reading order and the
final table sortable by `Source_File + Serial_No`.

---

## 🖼️ 3. Preprocessing & Masking (`preprocess_roi`)

Each voter box ("region of interest") is prepared for Tesseract:
- **Upscaling**: 2× cubic – OCR engines read large crisp glyphs far better.
- **Photo mask (crucial)**: white rectangle over `x > 0.72·w, y > 0.25·h` (right ~28%
  where the photograph sits) so faces/background aren't "read" as garbage text.
- **Border**: 10 px white padding so edge-touching glyphs survive thresholding.
- **Thresholding**: OTSU binarization (automatic black/white cutoff).

---

## 🏁 4. Grid Detection (`detect_boxes`)

How the system finds voter boxes without being told where they are:
1.  **Binary inversion**: lines become white on black.
2.  **Morphological kernels**: wide-flat `w//40` kernel keeps only horizontal lines;
    tall-thin `h//40` kernel keeps only vertical lines; their sum is the grid skeleton
    (after close+dilate cleanup).
3.  **Contour filtering**: `expected_area = (w/3)·(h/10)` (≈ 1/30 of the page); keep
    30–150% of it with aspect ratio 1.5–5.0 (wide voter cells; banners rejected).
4.  **IoU-NMS dedup (the 60→30 fix)**: `RETR_LIST` yields every box twice
    (inner+outer contour). Boxes sorted by area; any box with
    `IoU(box, kept) > 0.4` is dropped. Before this fix every voter was OCR'd twice
    and appeared as a duplicate row.

---

## 🔤 5. Two-pass OCR (`ocr_block`)

1. **Full block** (`psm 6`, `hin+eng`) → cleaned lines → written to the `.txt` as
   `--- BLOCK n ---` and parsed for all fields.
2. **Header strip** (top 32% of the box, `psm 7`, eng whitelist) → accepted **only**
   on strict `AAA1234567` / `UP/…` match; overrides an empty/garbled full-block ID.
   Loose near-matches are deliberately discarded – a blank ID (Name+Age dedup
   fallback) beats a fabricated one.

Per-block parse failures are caught and logged (`pN block k parse failed`) so one
bad box never kills its page; page failures likewise log and continue to the next
page/PDF.

---

## 🛠️ How to Adapt for Different Grids

If a future layout changes (2 columns, different cell size, moved photo):

### 1. Box counting/filtering (`detect_boxes`)
- Larger cells: change `expected_area` divisors (`w/3`, `h/10` → e.g. `w/2`).
- Squarer cells: widen the `1.5 < aspect < 5.0` band.

### 2. Line sensitivity (kernels)
- Faint/dotted lines: use smaller divisors (e.g. `w//60`) – smaller kernels are
  more sensitive to thin lines.

### 3. Photo mask (`preprocess_roi`)
- Photo on the left: mask `x < 0.3·w` instead of `x > 0.72·w`.

### 4. Row grouping
- Tightly packed rows: lower the `Δy > 50` split threshold (e.g. `30`).

### 5. Languages
- Marathi rolls: full-block langs become `-l mar+eng` (header strip stays `eng`).

---

## Legacy `img_to_txt_14_to_6905.py` (superseded)

Same morphology (`extract_grid_blocks`, `sort_contours`, `preprocess_roi_raw`) but:
hard-coded `INPUT_FOLDER/OUTPUT_FILE/OCR_CONFIG/PADDING`, PNGs pre-rendered by
Stage 1, no IoU dedup (60 blocks/page), single OCR pass, progress flushed every 10
pages. Kept for reference; use `main.py --pages N` for experiments instead.
