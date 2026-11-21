from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph


LOSS_HEADING = "D. Loss Functions"
NEXT_HEADING = "V. EXPERIMENTS"


def _find_paragraph(document: Document, text: str) -> Paragraph:
    for paragraph in document.paragraphs:
        if paragraph.text.strip() == text:
            return paragraph
    raise ValueError(f"Paragraph not found: {text}")


def _set_equation_number(paragraph_element, old: str, new: str) -> None:
    replaced = False
    for text_element in paragraph_element.xpath(".//m:t"):
        if text_element.text and old in text_element.text:
            text_element.text = text_element.text.replace(old, new)
            replaced = True
    if not replaced:
        raise ValueError(f"Equation number {old} not found")


def _remove_dice_batch_fraction(paragraph_element) -> None:
    fractions = paragraph_element.xpath(".//m:f")
    if len(fractions) != 1:
        raise ValueError(
            f"Expected one Dice batch-correction fraction, found {len(fractions)}"
        )
    fraction = fractions[0]
    fraction.getparent().remove(fraction)


def _rename_first_math_text(paragraph_element, old: str, new: str) -> None:
    for text_element in paragraph_element.iter(qn("m:t")):
        if text_element.text == old:
            text_element.text = new
            return
    raise ValueError(f"Math text {old!r} not found")


def _split_total_equation(source_element):
    base_equation = deepcopy(source_element)
    total_equation = deepcopy(source_element)

    base_expression = base_equation.xpath(".//m:eqArr/m:e")[0]
    base_children = list(base_expression)
    _rename_first_math_text(base_equation, "total", "base")
    for child in base_children[7:11]:
        base_expression.remove(child)
    _set_equation_number(base_equation, "#(21)", "#(13)")

    total_expression = total_equation.xpath(".//m:eqArr/m:e")[0]
    total_children = list(total_expression)
    base_term = deepcopy(total_children[0])
    _rename_first_math_text(base_term, "total", "base")
    for child in total_children[2:7]:
        total_expression.remove(child)
    total_expression.insert(2, base_term)
    _set_equation_number(total_equation, "#(21)", "#(14)")

    return base_equation, total_equation


def _new_body_paragraph(before: Paragraph, source_ppr, text: str) -> Paragraph:
    paragraph_element = OxmlElement("w:p")
    if source_ppr is not None:
        paragraph_element.append(deepcopy(source_ppr))
    before._p.addprevious(paragraph_element)
    paragraph = Paragraph(paragraph_element, before._parent)
    paragraph.add_run(text)
    return paragraph


def rewrite_loss_section(input_path: Path, output_path: Path) -> None:
    document = Document(input_path)
    loss_heading = _find_paragraph(document, LOSS_HEADING)
    next_heading = _find_paragraph(document, NEXT_HEADING)

    body = document.element.body
    children = list(body)
    start = children.index(loss_heading._p)
    end = children.index(next_heading._p)
    section_elements = children[start + 1 : end]

    if len(section_elements) < 18:
        raise ValueError(
            "The loss section is shorter than expected; refusing an unsafe rewrite."
        )

    body_ppr = deepcopy(section_elements[0].pPr)
    equation_embedding = deepcopy(section_elements[1])
    equation_instance = deepcopy(section_elements[14])
    equation_base, equation_total = _split_total_equation(section_elements[16])

    _set_equation_number(equation_embedding, "#(11)", "#(11)")
    _remove_dice_batch_fraction(equation_instance)
    _set_equation_number(equation_instance, "#(20)", "#(12)")

    for element in section_elements:
        body.remove(element)

    _new_body_paragraph(
        next_heading,
        body_ppr,
        (
            "TreeSegFormer3D is optimized using a staged joint objective. "
            "The discriminative embedding loss, foreground/background classification "
            "loss, and boundary contrastive loss are active throughout training. "
            "The embedding objective combines intra-instance variance, inter-instance "
            "distance, and center regularization:"
        ),
    )
    next_heading._p.addprevious(equation_embedding)

    _new_body_paragraph(
        next_heading,
        body_ppr,
        (
            "The variance term penalizes foreground embeddings whose L1 distance from "
            "the corresponding instance center exceeds 0.5, whereas the distance term "
            "encourages different instance centers to remain at least 3.0 apart, twice "
            "the inter-instance margin of 1.5. The regularization term constrains the "
            "center norms. Foreground/background classification uses negative "
            "log-likelihood loss. The boundary contrastive loss is the weighted sum of "
            "the repulsive and attractive terms, with a common weight of 0.5."
        ),
    )

    _new_body_paragraph(
        next_heading,
        body_ppr,
        (
            "Once query-based segmentation is activated, each query is associated with "
            "the ground-truth instance containing its source voxel; unmatched queries "
            "receive an all-zero target mask and zero target confidence. Mask predictions "
            "are supervised by binary cross-entropy and Dice losses, while the confidence "
            "branch uses mean squared error against the crop-adjusted IoU. The instance "
            "branch has no query-classification loss. The nominal BCE, Dice, and "
            "confidence weights are 1.0, 1.0, and 0.5, respectively, and the Dice term "
            "includes the configured batch-size correction. The per-output instance "
            "objective is"
        ),
    )
    next_heading._p.addprevious(equation_instance)

    _new_body_paragraph(
        next_heading,
        body_ppr,
        (
            "The embedding, foreground/background classification, and boundary "
            "contrastive losses that remain active throughout training are grouped as"
        ),
    )
    next_heading._p.addprevious(equation_base)

    _new_body_paragraph(
        next_heading,
        body_ppr,
        (
            "The semantic branch uses cross-entropy loss with a weight of 0.2. Masks are "
            "predicted before the first decoder layer and after each of the six decoder "
            "layers, yielding seven supervised outputs; one-to-many association is "
            "recomputed for each output. The complete objective is"
        ),
    )
    next_heading._p.addprevious(equation_total)

    _new_body_paragraph(
        next_heading,
        body_ppr,
        (
            "where t denotes the current training epoch and I(.) is the indicator "
            "function. With prepare_epoch=50, semantic and query-instance supervision is "
            "enabled from epoch 51 onward, whereas the embedding, foreground/background "
            "classification, and boundary contrastive losses remain active throughout "
            "training."
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rewrite_loss_section(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
