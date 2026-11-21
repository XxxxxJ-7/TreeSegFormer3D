from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FILL_BLUE = "#d9ecf2"
STROKE_BLUE = "#2f5fa7"
TEXT = "#000000"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = TEXT,
) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)


def _arrow_head(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: str,
    size: int,
) -> None:
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    wing = math.pi / 7
    p1 = (
        end[0] - size * math.cos(angle - wing),
        end[1] - size * math.sin(angle - wing),
    )
    p2 = (
        end[0] - size * math.cos(angle + wing),
        end[1] - size * math.sin(angle + wing),
    )
    draw.polygon([end, p1, p2], fill=fill)


def _dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    fill: str,
    width: int,
    dash: int,
    gap: int,
) -> None:
    for start, end in zip(points[:-1], points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        ux, uy = dx / length, dy / length
        position = 0.0
        while position < length:
            segment_end = min(position + dash, length)
            p0 = (start[0] + ux * position, start[1] + uy * position)
            p1 = (start[0] + ux * segment_end, start[1] + uy * segment_end)
            draw.line([p0, p1], fill=fill, width=width)
            position += dash + gap


def revise_figure(input_path: Path, output_path: Path) -> None:
    source = Image.open(input_path).convert("RGB")
    scale = 2
    image = source.resize(
        (source.width * scale, source.height * scale),
        Image.Resampling.LANCZOS,
    )
    draw = ImageDraw.Draw(image)

    def box(coords: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return tuple(value * scale for value in coords)

    # Correct the activation label.
    relu_box = box((681, 247, 846, 288))
    draw.rounded_rectangle(
        relu_box,
        radius=8 * scale,
        fill=FILL_BLUE,
        outline=STROKE_BLUE,
        width=2 * scale,
    )
    _center_text(draw, relu_box, "ReLU", _font(26 * scale))

    # Make the query initialization and stage-wise predictions explicit.
    initial_query_box = box((339, 489, 586, 532))
    draw.rounded_rectangle(
        initial_query_box,
        radius=7 * scale,
        fill="#dcefd7",
        outline="#000000",
        width=2 * scale,
    )
    _center_text(
        draw,
        initial_query_box,
        "Initial Queries (Q0)",
        _font(19 * scale),
    )

    decoder_box = box((68, 563, 587, 606))
    draw.rounded_rectangle(
        decoder_box,
        radius=7 * scale,
        fill="#e4e8ee",
        outline=STROKE_BLUE,
        width=2 * scale,
    )
    _center_text(
        draw,
        decoder_box,
        "Decoder Layer × 6",
        _font(24 * scale),
    )

    stage_query_box = box((68, 632, 587, 673))
    draw.rounded_rectangle(
        stage_query_box,
        radius=7 * scale,
        fill="#d6e0f2",
        outline=STROKE_BLUE,
        width=2 * scale,
    )
    _center_text(
        draw,
        stage_query_box,
        "Stage Queries Q(ℓ),  ℓ = 0, ..., 6",
        _font(18 * scale),
    )

    # Add the mask-guided feedback used by the next cross-attention stage.
    feedback = [
        (715 * scale, 861 * scale),
        (889 * scale, 861 * scale),
        (889 * scale, 585 * scale),
        (587 * scale, 585 * scale),
    ]
    _dashed_polyline(
        draw,
        feedback,
        fill="#6b6b6b",
        width=2 * scale,
        dash=7 * scale,
        gap=6 * scale,
    )
    _arrow_head(
        draw,
        feedback[-2],
        feedback[-1],
        fill="#6b6b6b",
        size=10 * scale,
    )

    label = "Mask-guided attention"
    label_font = _font(13 * scale)
    label_bbox = draw.textbbox((0, 0), label, font=label_font)
    label_img = Image.new(
        "RGBA",
        (
            label_bbox[2] - label_bbox[0] + 10 * scale,
            label_bbox[3] - label_bbox[1] + 8 * scale,
        ),
        (255, 255, 255, 230),
    )
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text(
        (5 * scale - label_bbox[0], 4 * scale - label_bbox[1]),
        label,
        font=label_font,
        fill="#555555",
    )
    image.paste(
        label_img,
        (700 * scale, 548 * scale),
        label_img,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, dpi=(600, 600), optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    revise_figure(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
