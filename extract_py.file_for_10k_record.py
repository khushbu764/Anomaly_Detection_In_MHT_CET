"""
Extract records from the MHT CET Merit List PDF (PCMAI) directly into a CSV file.

Filters out:
  - Document headers & footers (Government notices, dates, page numbers)
  - Repeated table header rows on every page (header is written ONCE, at the top)

v2 FIXES: column names are now detected properly
  - handles headers that are split over several rows
  - handles headers that sit outside the detected table grid
  - no longer mistakes the first data row for the header
  - MANUAL_HEADERS lets you type the column names yourself if auto-detect fails
"""

import csv
import os
import sys
import urllib.error
import urllib.request
import pymupdf  # Install via: pip install pymupdf

# --- Configuration Settings ---
PDF_URL = "https://cappublicdocs2026.blob.core.windows.net/meritlists/final/FE2026_PCMAI_MeritList_Final.pdf"
LOCAL_PDF_FILE = "FE2026_PCMAI_MeritList_Final.pdf"   # <-- change to your PDF's file name if different

MAX_RECORDS = 10000         # Set to None for all records in the PDF
INCLUDE_CSV_HEADER = True   # Set to False for data only, without column names

# If the automatic column names come out wrong, type them here in the same order
# as the PDF columns, e.g. ["Merit No", "Application ID", "Candidate Name", ...]
# Leave as None to use automatic detection.
MANUAL_HEADERS = None

# The PDF's "Note :- ..." text sits in the same cell as the first column's title.
# The note is removed, and the first column is named with this text instead.
# Change it to match the real first column title in your PDF (e.g. "Merit No", "Sr. No").
FIRST_COLUMN_NAME = "Merit Number"

# Prints the first raw rows of the table so you can see what the PDF really contains
DEBUG_SHOW_RAW_ROWS = True

OUTPUT_CSV_FILE = (
    f"merit_list_PCMAI_top{MAX_RECORDS}.csv" if MAX_RECORDS else "merit_list_PCMAI_all.csv"
)

PROGRESS_EVERY_N_PAGES = 25


def download_pdf_if_missing(url: str, dest_path: str):
    """Download the PDF if it isn't already in the folder."""
    if os.path.exists(dest_path):
        print(f"[✓] Found local PDF file: {dest_path}")
        return dest_path

    print(f"[*] Downloading PDF from:\n    {url}")
    print(f"[*] Saving locally to: {dest_path}")

    def progress(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            pct = min(downloaded * 100 / total_size, 100)
            sys.stdout.write(
                f"\r    Progress: {downloaded / 1048576:6.1f} / {total_size / 1048576:6.1f} MB [{pct:5.1f}%]"
            )
        else:
            sys.stdout.write(f"\r    Progress: {downloaded / 1048576:6.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress)
    except (urllib.error.URLError, OSError) as exc:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        print(f"\n\n[X] Could not download the PDF: {exc}")
        print("    Download it manually in your browser, save it in THIS folder")
        print(f"    with the exact name: {dest_path}  and run the script again.")
        sys.exit(1)

    print("\n[✓] Download finished successfully!\n")
    return dest_path


def clean_str(val):
    """Remove line breaks and collapse multiple whitespaces."""
    if val is None:
        return ""
    return " ".join(str(val).split())


def clean_header_title(title):
    """Format a column title nicely, removing internal break characters."""
    if not title:
        return ""
    return " ".join(str(title).replace("-\n", "-").replace("\n", " ").split())


def clean_header_cell(raw):
    """Clean one header cell. If it contains the PDF's 'Note :- ...' text, drop the note."""
    text = clean_header_title(raw)
    if text.lower().startswith("note"):
        return FIRST_COLUMN_NAME
    return text


def is_note_row(row):
    """A row that holds only the 'Note :- ...' text (all other cells empty)."""
    if not row or not clean_str(row[0]).lower().startswith("note"):
        return False
    return all(not clean_str(c) for c in row[1:])


def is_data_row(row):
    """A real candidate record starts with a numeric merit rank (1, 2, 3...)."""
    return bool(row) and clean_str(row[0]).isdigit()


def is_generic_names(names):
    """True if PyMuPDF only produced placeholder names such as 'Col1', 'Col2'."""
    real = [n for n in names if n and not str(n).strip().lower().startswith("col")]
    return len(real) == 0


def detect_headers(table, rows):
    """
    Work out the column names, trying in order:
      1. MANUAL_HEADERS (if you set it)
      2. Header rows inside the table = every row above the first numbered row
         (several rows are merged column by column)
      3. The header PyMuPDF found outside the table grid
      4. Generic names: Column_1, Column_2, ...
    """
    ncols = max(len(r) for r in rows)

    if MANUAL_HEADERS:
        headers = list(MANUAL_HEADERS)
        if len(headers) != ncols:
            print(f"[!] MANUAL_HEADERS has {len(headers)} names but the table has {ncols} columns.")
        return headers

    # 2. Rows above the first numbered row
    header_rows = []
    for r in rows:
        if is_data_row(r):
            break
        if is_note_row(r):      # skip a row that contains only the Note text
            continue
        header_rows.append(r)

    if header_rows:
        headers = []
        for c in range(ncols):
            parts = []
            for r in header_rows:
                if c < len(r):
                    piece = clean_header_cell(r[c])
                    if piece and piece not in parts:
                        parts.append(piece)
            headers.append(" ".join(parts))
        if any(headers):
            return headers

    # 3. Header found outside the grid by PyMuPDF
    try:
        names = [clean_header_title(n) for n in table.header.names]
        if names and not is_generic_names(names):
            return names
    except Exception:
        pass

    # 4. Fallback
    return [f"Column_{i + 1}" for i in range(ncols)]


def extract_and_save_to_csv(pdf_path: str, output_csv: str, max_records: int = 10000, include_header: bool = True):
    """Parse the PDF, extract table rows, clean cell data and stream them into the CSV."""
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    print(f"[*] Scanning PDF ({total_pages} total pages)...")

    extracted_count = 0
    headers = None
    debug_done = False

    # utf-8-sig makes Excel show special characters correctly
    with open(output_csv, mode="w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)

        for page_idx in range(total_pages):
            page = doc[page_idx]
            tab_finder = page.find_tables()

            if not tab_finder.tables:
                continue

            table = tab_finder.tables[0]
            rows = table.extract()

            if not rows:
                continue

            # Show raw rows once, so wrong column names are easy to diagnose
            if DEBUG_SHOW_RAW_ROWS and not debug_done:
                print(f"\n--- RAW first rows of the table (page {page_idx + 1}) ---")
                for raw in rows[:6]:
                    print([clean_str(c) for c in raw])
                print("--- end raw rows ---\n")
                debug_done = True

            # Column names: detected once, from the first page that has a table
            if headers is None:
                headers = detect_headers(table, rows)
                if include_header:
                    writer.writerow(headers)
                print(f"[✓] {len(headers)} columns: {headers}")

            # Every row that starts with a merit number is a record.
            # Header rows, repeated headers and footers fail isdigit() and are dropped.
            for row in rows:
                if is_data_row(row):
                    writer.writerow([clean_str(cell) for cell in row])
                    extracted_count += 1

                    if max_records and extracted_count >= max_records:
                        break

            if (page_idx + 1) % PROGRESS_EVERY_N_PAGES == 0:
                target = f"/{max_records}" if max_records else ""
                print(f"    ...page {page_idx + 1}/{total_pages} done | records so far: {extracted_count}{target}")

            if max_records and extracted_count >= max_records:
                print(f"[✓] Target of {max_records} records reached at page {page_idx + 1}.")
                break

    doc.close()
    return headers, extracted_count


def main():
    pdf_file = download_pdf_if_missing(PDF_URL, LOCAL_PDF_FILE)

    target_label = MAX_RECORDS if MAX_RECORDS else "ALL"
    print(f"[*] Extracting top {target_label} records directly to '{OUTPUT_CSV_FILE}'...")
    headers, count = extract_and_save_to_csv(
        pdf_path=pdf_file,
        output_csv=OUTPUT_CSV_FILE,
        max_records=MAX_RECORDS,
        include_header=INCLUDE_CSV_HEADER,
    )

    print(f"\n[SUCCESS] Extracted {count} records successfully!")
    if MAX_RECORDS and count < MAX_RECORDS:
        print(f"[!] Note: PDF contained only {count} records, fewer than the requested {MAX_RECORDS}.")
    print(f"[SUCCESS] CSV file saved to: {os.path.abspath(OUTPUT_CSV_FILE)}")

    with open(OUTPUT_CSV_FILE, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        preview_rows = [next(reader) for _ in range(min(6, count + (1 if INCLUDE_CSV_HEADER else 0)))]

    print("\n--- Preview of First Few Rows in CSV ---")
    for r in preview_rows:
        print(r[:5])


if __name__ == "__main__":
    main()
