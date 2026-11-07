"""Save a PKLot image with its parking-space annotations drawn on top."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree.ElementTree import ParseError

import cv2

from src.data.pklot_parser import parse_annotation
from src.visualization.annotations import render_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize PKLot parking-space annotations")
    parser.add_argument("--image", required=True, type=Path, help="PKLot JPG image")
    parser.add_argument("--xml", required=True, type=Path, help="Matching PKLot XML annotation")
    parser.add_argument("--mode", choices=("geometry", "ground-truth"), default="geometry")
    parser.add_argument("--output", type=Path, help="Output image path")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not args.xml.is_file():
        parser.error(f"Annotation not found: {args.xml}")
    if args.image.stem != args.xml.stem:
        parser.error("Image and annotation filenames do not match")

    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Image cannot be read: {args.image}")

    try:
        spaces, ground_truth = parse_annotation(args.xml)
        annotated = render_annotations(image, spaces, ground_truth, args.mode)
    except (ParseError, ValueError) as error:
        parser.error(str(error))

    output_path = args.output or Path("outputs") / f"{args.image.stem}_{args.mode}.jpg"
    if output_path.resolve() == args.image.resolve():
        parser.error("Output path must differ from the input image")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        parser.error(f"Could not save output image: {output_path}")
    print(f"Rendered {len(spaces)} spaces in {args.mode} mode: {output_path}")


if __name__ == "__main__":
    main()
