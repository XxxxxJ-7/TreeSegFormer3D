from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


CONCLUSION_HEADING = "VI. CONCLUSION"
NEXT_HEADING = "Acknowledgment"


def _find_paragraph(document: Document, text: str) -> Paragraph:
    for paragraph in document.paragraphs:
        if paragraph.text.strip() == text:
            return paragraph
    raise ValueError(f"Paragraph not found: {text}")


def _insert_paragraph(before: Paragraph, ppr, text: str) -> Paragraph:
    paragraph_element = OxmlElement("w:p")
    if ppr is not None:
        paragraph_element.append(deepcopy(ppr))
    before._p.addprevious(paragraph_element)
    paragraph = Paragraph(paragraph_element, before._parent)
    paragraph.add_run(text)
    return paragraph


def rewrite_conclusion(input_path: Path, output_path: Path) -> None:
    document = Document(input_path)
    conclusion_heading = _find_paragraph(document, CONCLUSION_HEADING)
    next_heading = _find_paragraph(document, NEXT_HEADING)

    body = document.element.body
    body_children = list(body)
    start = body_children.index(conclusion_heading._p)
    end = body_children.index(next_heading._p)
    old_elements = body_children[start + 1 : end]
    if len(old_elements) != 1:
        raise ValueError(
            f"Expected one existing conclusion paragraph, found {len(old_elements)}"
        )

    body_ppr = deepcopy(old_elements[0].pPr)
    for element in old_elements:
        body.remove(element)

    paragraphs = [
        (
            "This work introduced TreeSegFormer3D, an end-to-end query-based "
            "framework for individual-tree instance segmentation in forest LiDAR "
            "point clouds. IAQG selects foreground voxels through cluster-balanced "
            "sampling and converts their backbone descriptors into instance queries, "
            "while boundary contrastive supervision is used to promote the "
            "separability of adjacent-tree embeddings. SPMD injects explicit 3-D "
            "position information "
            "into both attention aggregation and mask prediction and applies dual "
            "nonlinear projections during stage-wise decoding. We also constructed "
            "the Xiongan Forest LiDAR Dataset, which contains 28 plantation plots and "
            "1,932 annotated trees."
        ),
        (
            "TreeSegFormer3D achieved the highest F1 score and coverage on both "
            "evaluation datasets. On Xiongan, it obtained an F1 score of 90.38% and "
            "coverage of 82.35%, improving over ForestFormer3D by 4.72 and 3.43 "
            "percentage points, respectively. On FOR-instanceV2, it achieved an F1 "
            "score of 86.68% and coverage of 78.81%, corresponding to gains of 0.33 "
            "and 1.00 points. The ablation results further show that IAQG provides "
            "most of the coverage improvement, whereas SPMD primarily strengthens "
            "recall; combining the two modules yields the best overall F1 score. "
            "These results indicate that balanced query allocation and explicit "
            "spatial modeling are complementary for separating neighboring trees in "
            "dense forest point clouds."
        ),
        (
            "The present evidence remains bounded by the geographic and sensor "
            "coverage of the evaluated datasets. The smaller improvement on "
            "FOR-instanceV2 and the semantic mIoU of 78.17% also indicate that "
            "cross-domain transfer and semantic-instance consistency are not fully "
            "resolved. Future work should therefore evaluate TreeSegFormer3D across "
            "additional regions, sensing platforms, and mixed-species natural "
            "forests, while improving semantic-instance coupling and the efficiency "
            "of large-area inference."
        ),
    ]

    for text in paragraphs:
        _insert_paragraph(next_heading, body_ppr, text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rewrite_conclusion(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
