#!/usr/bin/env python3
"""
PDF Ingestion Pipeline for KERCA
=================================
Extracts text from PDFs, chunks them, feeds into KERCA kernel.

Usage: python ingest_pdfs.py --dir "D:\Kerca\papers"
"""

import sys
from pathlib import Path

# Ensure the script directory is in the path BEFORE imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from PyPDF2 import PdfReader
except ImportError:
    print("PyPDF2 not found. Install: pip install PyPDF2")
    sys.exit(1)
from Kerca_kernel import bootstrap_kernel

# --- Config ---
PDF_DIR = Path("D:/Kerca/papers")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250
MAX_PAPERS = None  # Set to a number to limit, or None for all

# --- Extraction ---

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    parts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            parts.append(f"--- Page {i+1} ---\n{text}")
    return "\n\n".join(parts)

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks

def main():
    pdf_dir = PDF_DIR
    
    # Check for command line argument
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=str(PDF_DIR))
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    
    pdf_dir = Path(args.dir)
    if args.max:
        global MAX_PAPERS
        MAX_PAPERS = args.max
    
    if not pdf_dir.exists():
        print(f"Directory not found: {pdf_dir}")
        print("Create it and add PDFs.")
        return
    
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files in {pdf_dir}")
        return
    
    if MAX_PAPERS:
        pdf_files = pdf_files[:MAX_PAPERS]
    
    print("=" * 60)
    print(f"KERCA PDF Ingestion — {len(pdf_files)} papers")
    print("=" * 60)
    
    kernel = bootstrap_kernel(load_existing=True)
    before_count = len(kernel.capsules)
    total_chunks = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] {pdf_file.name}")
        try:
            text = extract_text_from_pdf(pdf_file)
            if not text.strip():
                print("  [SKIP] No extractable text")
                continue
            
            title = text.strip().split("\n")[0][:150]
            print(f"  Title: {title}")
            print(f"  Characters: {len(text):,}")
            
            # Ingest metadata
            kernel.ingest_input(
                f"PAPER: {title}\nFILE: {pdf_file.name}",
                source=f"paper_meta:{pdf_file.stem}",
                activate=True
            )
            
            # Chunk and ingest body
            chunks = chunk_text(text)
            for j, chunk in enumerate(chunks):
                kernel.ingest_input(
                    f"From '{title}':\n\n{chunk}",
                    source=f"paper:{pdf_file.stem}:chunk{j+1}",
                    activate=(j < 2)
                )
                total_chunks += 1
            
            print(f"  Chunks: {len(chunks)}")
            
            # Periodic cycle
            if i % 5 == 0:
                r = kernel.run_cycle()
                print(f"  [Cycle {r['cycle']}] {r['merges_this_cycle']} merges, "
                      f"{r['tensions_created']} tensions, "
                      f"{r['active_capsules']} active")
        
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    # Final cognition push
    print("\n" + "=" * 60)
    print(f"Ingestion complete — {total_chunks} total chunks")
    print("Running final cognition cycles...")
    print("=" * 60)
    
    for _ in range(10):
        r = kernel.run_cycle()
        if r['merges_this_cycle'] or r['tensions_created'] or r['ergo_resolutions']:
            print(f"  Cycle {r['cycle']:>3}: "
                  f"{r['merges_this_cycle']} merges, "
                  f"{r['tensions_created']} tensions, "
                  f"{r['tensions_resolved']} resolved, "
                  f"{r['ergo_resolutions']} ergo, "
                  f"{r['reprocessed']} reproc, "
                  f"{r['challenges_created']} challenges")
    
    kernel.store.save(kernel.capsules, clusters=kernel.cluster_mgr.to_dict())
    
    final = kernel.status()
    new_capsules = final['total_capsules'] - before_count
    
    print(f"\n{'=' * 60}")
    print(f"INGESTION REPORT")
    print(f"{'=' * 60}")
    print(f"  Papers processed:      {len(pdf_files)}")
    print(f"  Total chunks:          {total_chunks}")
    print(f"  Capsules before:       {before_count}")
    print(f"  Capsules after:        {final['total_capsules']}")
    print(f"  New capsules:          {new_capsules}")
    print(f"  Active capsules:       {final['active_capsules']}")
    print(f"  Shadows (merged):      {final['shadows']}")
    print(f"  Tensions:              {final['tensions']}")
    print(f"  Challenges:            {final['challenges']}")
    print(f"  Clusters:              {final['total_clusters']}")
    print(f"  Active clusters:       {final['active_clusters']}")
    print(f"  Merges total:          {final['routing_stats']['merges_performed']}")
    print(f"  Ergo resolutions:      {final['routing_stats']['ergo_resolutions']}")
    print(f"  Tensions created:      {final['routing_stats']['tensions_created']}")
    print(f"  Tensions compressed:   {final['routing_stats']['tensions_compressed']}")
    print(f"  Tensions resolved:     {final['routing_stats']['tensions_resolved_stale']}")
    print(f"  Challenges created:    {final['routing_stats']['challenges_created']}")
    print(f"  Reprocess exhausted:   {final['reprocess_exhausted']}")
    print(f"  Timeline frames:       {final['timeline_stats']['total_frames']}")
    print(f"  Cycle count:           {final['cycle_count']}")
    print(f"\n  Knowledge base saved:  {kernel.store.store_path}")

if __name__ == "__main__":
    main()