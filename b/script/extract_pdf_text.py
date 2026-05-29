#!/usr/bin/env python3
"""Extract text from all PDFs in p/ subfolders into .txt files for analysis."""

import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import fitz  # PyMuPDF

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P_DIR = os.path.join(BASE_DIR, "p")


def extract_folder(folder_path, folder_name):
    pdfs = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')])
    if not pdfs:
        return

    out_path = os.path.join(folder_path, "_extracted_text.txt")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
        print(f"  SKIP (exists): {folder_name}/_extracted_text.txt", flush=True)
        return

    all_text = []
    for pdf_name in pdfs:
        pdf_path = os.path.join(folder_path, pdf_name)
        try:
            doc = fitz.open(pdf_path)
            all_text.append(f"{'='*80}")
            all_text.append(f"FILE: {pdf_name} ({len(doc)} pages)")
            all_text.append(f"{'='*80}\n")
            for i in range(len(doc)):
                text = doc[i].get_text()
                all_text.append(f"--- Page {i+1}/{len(doc)} ---")
                all_text.append(text)
            doc.close()
        except Exception as e:
            all_text.append(f"ERROR reading {pdf_name}: {e}")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_text))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  OK: {folder_name}/_extracted_text.txt ({size_kb:.0f} KB)", flush=True)


def main():
    folders = sorted(os.listdir(P_DIR))
    print(f"Processing {len(folders)} folders in p/\n", flush=True)

    for folder_name in folders:
        folder_path = os.path.join(P_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        extract_folder(folder_path, folder_name)

    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
