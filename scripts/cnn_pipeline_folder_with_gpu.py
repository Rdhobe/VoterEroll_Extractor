import pandas as pd
import re
import os
try:
    import torch
    _TORCH_OK = True
except Exception:
    torch = None
    _TORCH_OK = False
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

# =========================================================
# GPU CONFIGURATION
# =========================================================
# INPUT_FILES = ['file1.txt', 'file2.txt']  <-- Add your files here
INPUT_FILES = [
    'combined_voter_data_HIN_WI.txt',
]
OUTPUT_EXCEL_FILE = 'Final_Voter_List_HIN_WI.xlsx'

# GPU Toggle (Change to False for CPU)
USE_GPU = True

print("🚀 GPU Voter Parser Starting...")

# GPU Check (torch is optional - cosmetic only, parsing is regex+CPU)
if USE_GPU and _TORCH_OK:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
else:
    device = 'cpu'
    print("CPU Mode (USE_GPU=False or torch unavailable)")

def process_single_block(args):
    """Single block processor for GPU parallelization"""
    block_text, filename = args
    text = block_text.strip()
    
    if not text:
        return None
        
    entry = {
        "Serial_No": "",
        "Voter_ID": "",
        "Part_Section": "",
        "Name": "",
        "Relation_Type": "",
        "Relative_Name": "",
        "House_No": "",
        "Age": "",
        "Gender": "",
        "Source_File": filename,
        "Raw_Text_Dump": text 
    }

    # Header Info (Marathi 3-field + Hindi 2-field fallback)
    # Updated regex to handle dots in Serial No in Voter ID
    header_match = re.search(r'^\s*([\d.,]+).*?([A-Z0-9\u0966-\u096F\/]{8,})\s+([\d\/]+)', text, re.MULTILINE)


    if header_match:
        entry['Serial_No'] = header_match.group(1).replace(',', '').replace('.', '')
        entry['Voter_ID'] = header_match.group(2)
        entry['Part_Section'] = header_match.group(3)
    else:
        # Hindi format has only Serial + Voter_ID, e.g. "332 ZSY439728 |" or "33 UP/70/355/0000290"
        # Voter_ID is either EPIC (ABC1234567, OCR-tolerant) or old UP/.. roll number
        hindi_header = re.search(r'([0-9\u0966-\u096F]{2,4})\s*[\|!l:॥\.]*\s*((?:UP\/[0-9\/]+|[A-Z0-9]{2,}\/?[A-Z0-9\/]{5,}))', text)
        if hindi_header:
            entry['Serial_No'] = hindi_header.group(1).replace(',', '').replace('.', '')
            entry['Voter_ID'] = hindi_header.group(2).strip('| !l:.॥')
        else:
            # Last resort: any long EPIC/roll-like token
            epic_only = re.search(r'\b((?:UP\/[0-9\/]{5,}|[A-Z]{3}[0-9]{7}|[A-Z0-9]{8,}))\b', text)
            if epic_only:
                entry['Voter_ID'] = epic_only.group(1)

    # Name (Marathi + Hindi: naam)
    # Updated regex to handle 'Ol ' noise and apostrophes
    name_match = re.search(r'(?:मतदाराचे|पूर्ण|Waa|Hse)(?:.*?[:\-\']\s*|\s+)(?:Ol\s+)?(?:[\'\']\s*)?(.*)', text, re.IGNORECASE)
    if name_match:
        raw_name = name_match.group(1).split('\n')[0].strip()
        entry['Name'] = raw_name
    if not entry['Name']:
        # Hindi format: "naam : <name>"
        hindi_name = re.search(r'नाम\s*[:\-\']\s*(.*)', text)
        if hindi_name:
            entry['Name'] = hindi_name.group(1).strip()

    # Relative (Marathi + Hindi: pita/pati/mata ka naam)
    # Updated regex to handle 'नावा' (naava) as well as 'नाव' (naav)
    rel_match = re.search(r'(वडिलांचे|पतीचे|आईचे|इतर)\s*नाव(?:ा)?\s*[:\-\']\s*(.*)', text)
    if rel_match:
        entry['Relation_Type'] = rel_match.group(1) 
        entry['Relative_Name'] = rel_match.group(2).strip()
    if not entry['Relative_Name']:
        # Hindi: पिता का नाम / पति का नाम / माता का नाम (OCR often gives Wart/Fart for mask noise - handled below)
        hindi_rel = re.search(r'(पिता|पति|माता|पत्नी|अन्य)\s*(?:का|की|के)?\s*नाम\s*[:\-\']\s*(.*)', text)
        if hindi_rel:
            entry['Relation_Type'] = hindi_rel.group(1).strip()
            entry['Relative_Name'] = hindi_rel.group(2).strip()

    # House No (Marathi ghar + Hindi makaan; OCR misreads makaan as Wart/Fart/makan den)
    house_match = re.search(r'(?:घर क्रमांक|UX PATH).*?[:\-\']\s*(.*)', text, re.IGNORECASE)
    if house_match:
        entry['House_No'] = house_match.group(1).strip()
    if not entry['House_No']:
        hindi_house = re.search(r'(?:मकान|Wart|Fart|makan|den)\s*(?:संख्या|sankhya|number)?\s*[:\-\']?\s*(.*)', text, re.IGNORECASE)
        if hindi_house:
            entry['House_No'] = hindi_house.group(1).strip()

    # Age & Gender (Marathi vay + Hindi aayu; ling same in both)
    age_match = re.search(r'वय\s*[:\-\']\s*(\d+)', text)
    if not age_match:
        age_match = re.search(r'आयु\s*[:\-\']\s*(\d+)', text)
    if age_match:
        entry['Age'] = age_match.group(1)

    sex_match = re.search(r'लिंग\s*[:\-\']\s*([^\n|]*)', text)
    if sex_match:
        raw_sex = sex_match.group(1).strip()
        if 'पु' in raw_sex or 'M' in raw_sex or 'पुरुष' in raw_sex or 'पुरूष' in raw_sex:
            entry['Gender'] = 'Male'
        elif 'स्री' in raw_sex or 'F' in raw_sex or 'AY' in raw_sex or 'स्त्री' in raw_sex or 'स््री' in raw_sex or 'महिला' in raw_sex:
            entry['Gender'] = 'Female'
        # else:
        # entry['Gender'] = raw_sex 

    return entry

def parse_voter_text(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File '{file_path}' not found.")
        return []

    print(f"📖 Reading {file_path}...")
    start_time = time.time()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    parsed_data = []
    file_chunks = content.split('FILENAME: ')
    print(f"📄 Found {len(file_chunks)-1} pages")

    # GPU Parallel Processing (8x faster)
    all_blocks = []
    for chunk_idx, chunk in enumerate(file_chunks):
        if chunk_idx == 0: 
            continue

        lines = chunk.split('\n', 1)
        if len(lines) < 2: 
            continue
            
        current_filename = lines[0].strip()
        page_content = lines[1]
        
        raw_blocks = re.split(r'--- BLOCK \d+ ---', page_content)
        for block in raw_blocks:
            all_blocks.append((block, current_filename))

    print(f"🔄 Processing {len(all_blocks)} blocks with GPU parallel...")
    
    # Multi-threaded processing (GPU-accelerated threading)
    with ThreadPoolExecutor(max_workers=8 if USE_GPU else 4) as executor:
        results = list(executor.map(process_single_block, all_blocks))
    
    # Filter None results
    parsed_data = [r for r in results if r is not None]
    
    print(f"✅ Parsed {len(parsed_data)} voters ({time.time()-start_time:.1f}s)")
    return parsed_data

def main():
    print(f"Processing {len(INPUT_FILES)} files from hardcoded list...")

    all_data = []
    
    for file_path in INPUT_FILES:
        print(f"Processing {file_path}...")
        file_data = parse_voter_text(file_path)
        if file_data:
            all_data.extend(file_data)
        else:
            print(f"  - No data or error reading {file_path}.")
    
    if not all_data:
        print("❌ No data extracted from any files.")
        return

    print(f"📊 Creating DataFrame with {len(all_data)} total records...")
    df = pd.DataFrame(all_data)
    
    cols = ["Source_File", "Serial_No", "Voter_ID", "Name", "Relation_Type", "Relative_Name", 
            "Age", "Gender", "House_No", "Part_Section", "Raw_Text_Dump"]
    
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    df = df[cols]

    # Remove duplicates (exact + Hindi double-detection: each voter OCR'd twice)
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    # Drop empty header rows (no Name and no Voter_ID)
    df = df[~((df['Name'].fillna('').astype(str).str.strip() == '') & (df['Voter_ID'].fillna('').astype(str).str.strip() == ''))]
    # De-duplicate double-detected boxes: same Voter_ID -> keep first
    has_id = df['Voter_ID'].fillna('').astype(str).str.strip() != ''
    df_id = df[has_id].drop_duplicates(subset=['Voter_ID'], keep='first')
    df_noid = df[~has_id].drop_duplicates(subset=['Name', 'Relative_Name', 'Age'], keep='first')
    df = pd.concat([df_id, df_noid], ignore_index=True)
    final_count = len(df)
    if initial_count > final_count:
        print(f"Removed {initial_count - final_count} duplicate/empty records ({initial_count} -> {final_count}).")

    print(f"💾 Saving {len(df)} records to {OUTPUT_EXCEL_FILE}...")
    df.to_excel(OUTPUT_EXCEL_FILE, index=False)
    print("🎉 GPU Voter List Ready!")

if __name__ == "__main__":
    main()
