# Corpus folder

Drop your source documents here. Supported formats: **PDF, DOCX, TXT, MD, HTML**.

Organise however you like — subfolders are scanned recursively. A sensible layout:

```
corpus/
├── statutes/        # Republic Acts, PDs, BPs (Medical Act, UHC Act, ...)
├── prc/             # PRC resolutions, rules on complaints & discipline
├── doh/             # Department of Health administrative orders
├── jurisprudence/   # Supreme Court decisions on medical malpractice
├── billing/         # PhilHealth circulars, discount IRRs, collection rules
├── civil_criminal/  # Civil Code, Revised Penal Code relevant provisions
└── _fetched/        # (auto) documents downloaded by the fetch scripts
```

After adding files, (re)build the index:

```
python cli.py ingest          # add/update
python cli.py ingest --reset  # full clean rebuild
```

## Tips for best retrieval quality
- Scanned/image PDFs are handled automatically via built-in OCR (Tesseract).
  Install the binary once: `winget install UB-Mannheim.TesseractOCR`.
  Text-based PDFs still extract faster and cleaner, so prefer them when you
  have the choice.
- Keep one law per file when you can — it makes citations cleaner.
- Filenames are used as a fallback source label, so name them meaningfully,
  e.g. `RA-11223-Universal-Health-Care-Act.pdf`.
