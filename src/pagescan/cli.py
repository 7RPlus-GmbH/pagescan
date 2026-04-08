"""Command-line interface for pagescan."""

import argparse
import logging
import sys

from pagescan.config import ScanConfig
from pagescan.pipeline import scan, scan_batch


def main():
    parser = argparse.ArgumentParser(
        prog='pagescan',
        description='Turn phone photos of documents into clean, deskewed, print-ready PDFs.',
    )
    parser.add_argument('input', nargs='?', help='Input image file')
    parser.add_argument('output', nargs='?', help='Output PDF file')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Batch process all images in input directory')
    parser.add_argument('--input-dir', default='.', help='Input directory for batch mode')
    parser.add_argument('--output-dir', default=None, help='Output directory for batch mode')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Parallel workers for batch (default: min(4, cpus), 1=sequential)')
    parser.add_argument('--no-rotate', action='store_true',
                        help='Skip OCR-based auto-rotation (faster)')
    parser.add_argument('--no-deskew', action='store_true',
                        help='Skip Hough-based deskew')
    parser.add_argument('--no-enhance', action='store_true',
                        help='Skip scan-like enhancement (keep color)')
    parser.add_argument('--no-ml', action='store_true',
                        help='Skip ML corner detection (conservative crop only)')
    parser.add_argument('--raw', action='store_true',
                        help='No enhancement, shadow removal, or white balance')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Save intermediate images to debug directory')
    parser.add_argument('--debug-dir', default='pagescan_debug',
                        help='Directory for debug output')
    parser.add_argument('--quality', '-q', type=int, default=50,
                        help='JPEG quality for PDF output (1-100, default: 50)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    config = ScanConfig(
        auto_orient=not args.no_rotate,
        deskew=not args.no_deskew,
        enhance=not (args.no_enhance or args.raw),
        shadow_removal=not args.raw,
        white_balance=not args.raw,
        use_ml=not args.no_ml,
        debug=args.debug,
        debug_dir=args.debug_dir,
        jpeg_quality=args.quality,
    )

    if args.batch:
        results = scan_batch(args.input_dir, args.output_dir,
                             config=config, workers=args.workers)
        sys.exit(0 if results['failed'] == 0 else 1)
    elif args.input:
        output = args.output or args.input.rsplit('.', 1)[0] + '.pdf'
        result = scan(args.input, output, config=config)
        if result['success']:
            print(f"Saved: {output}")
            print(f"Quality: {result['quality_score']:.2f}")
        else:
            print(f"Failed: {result['message']}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
