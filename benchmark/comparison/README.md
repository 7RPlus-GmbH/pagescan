# pagescan Comparison Benchmark

End-to-end comparison of document scanning packages on real phone photos.

## Dataset Structure

Place your test photos in `images/` with this naming convention:

```
images/
├── invoice_wood.jpg
├── invoice_white.jpg
├── invoice_dark.jpg
├── invoice_fabric.jpg
├── invoice_hand.jpg
├── receipt_wood.jpg
├── receipt_white.jpg
├── ...
```

Format: `{doctype}_{background}.jpg`

**Document types:** invoice, receipt, bank_statement, energy_bill, letter,
tax_form, insurance, kassenbon, id_card, mixed

**Backgrounds:** wood, white, dark, fabric, hand

## Running

```bash
python benchmark/comparison/run_comparison.py
```

Results are saved to `benchmark/comparison/results/` with side-by-side output PDFs.
