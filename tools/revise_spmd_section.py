from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from lxml import etree


HEADING = "C. Spatial Position-Aware Mask Decoder"
NEXT_HEADING = "D. Loss Functions"
MATHML_NS = "http://www.w3.org/1998/Math/MathML"
MML2OMML = Path(
    r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"
)


def _all_text(paragraph: Paragraph) -> str:
    return "".join(paragraph._p.xpath(".//w:t/text() | .//m:t/text()"))


def _find_paragraph(document: Document, text: str) -> Paragraph:
    for paragraph in document.paragraphs:
        if paragraph.text.strip() == text:
            return paragraph
    raise ValueError(f"Paragraph not found: {text}")


def _math_transform() -> etree.XSLT:
    if not MML2OMML.exists():
        raise FileNotFoundError(f"Office Math transform not found: {MML2OMML}")
    return etree.XSLT(etree.parse(str(MML2OMML)))


def _mathml_to_omml(fragment: str, transform: etree.XSLT):
    mathml = etree.fromstring(
        f'<math xmlns="{MATHML_NS}">{fragment}</math>'.encode("utf-8")
    )
    converted = transform(mathml).getroot()
    if converted is None:
        raise ValueError("MathML-to-OMML conversion returned no element")
    if etree.QName(converted).localname == "oMath":
        return deepcopy(converted)
    matches = converted.xpath(".//m:oMath", namespaces={"m": qn("m:m").rsplit("}", 1)[0][1:]})
    if not matches:
        raise ValueError("Converted MathML contains no m:oMath element")
    return deepcopy(matches[0])


def _new_paragraph(before: Paragraph, source_ppr) -> Paragraph:
    paragraph_element = OxmlElement("w:p")
    if source_ppr is not None:
        paragraph_element.append(deepcopy(source_ppr))
    before._p.addprevious(paragraph_element)
    return Paragraph(paragraph_element, before._parent)


def _append_parts(
    paragraph: Paragraph,
    parts: list[str | tuple[str, str]],
    transform: etree.XSLT,
) -> None:
    for part in parts:
        if isinstance(part, str):
            paragraph.add_run(part)
        else:
            kind, value = part
            if kind != "math":
                raise ValueError(f"Unknown paragraph part: {kind}")
            paragraph._p.append(_mathml_to_omml(value, transform))


def _unwrap_accents(paragraph_element) -> None:
    for accent in list(paragraph_element.xpath(".//m:acc")):
        expression = accent.find(qn("m:e"))
        if expression is None:
            continue
        parent = accent.getparent()
        index = parent.index(accent)
        for child in list(expression):
            parent.insert(index, deepcopy(child))
            index += 1
        parent.remove(accent)


def _set_figure_alt_text(figure_element) -> None:
    description = (
        "Architecture of the spatial position-aware mask decoder, including "
        "dual voxel-feature projections, three-dimensional positional encoding, "
        "six iterative decoder layers, and score and mask heads."
    )
    for doc_pr in figure_element.xpath(".//wp:docPr"):
        doc_pr.set("descr", description)
        doc_pr.set("title", "Spatial position-aware mask decoder")


def _replace_figure_blob(document: Document, figure: Paragraph, image_path: Path) -> None:
    blips = figure._p.xpath(".//a:blip")
    if len(blips) != 1:
        raise ValueError(f"Expected one embedded image in Fig. 6, found {len(blips)}")
    relationship_id = blips[0].get(qn("r:embed"))
    image_part = document.part.related_parts[relationship_id]
    image_part._blob = image_path.read_bytes()


def revise_spmd_section(
    input_path: Path,
    output_path: Path,
    figure_path: Path,
) -> None:
    document = Document(input_path)
    transform = _math_transform()

    heading = _find_paragraph(document, HEADING)
    next_heading = _find_paragraph(document, NEXT_HEADING)

    paragraphs = document.paragraphs
    heading_index = next(
        i for i, paragraph in enumerate(paragraphs) if paragraph._p is heading._p
    )
    next_index = next(
        i for i, paragraph in enumerate(paragraphs) if paragraph._p is next_heading._p
    )
    section = paragraphs[heading_index + 1 : next_index]
    if len(section) != 16:
        raise ValueError(
            f"Expected 16 paragraphs in the SPMD section, found {len(section)}"
        )

    normal_ppr = deepcopy(section[0]._p.pPr)
    equations = {
        number: deepcopy(
            next(p._p for p in section if f"#({number})" in _all_text(p))
        )
        for number in range(4, 11)
    }
    for number in (8, 9, 10):
        _unwrap_accents(equations[number])

    figure_paragraph = next(
        p for p in section if p._p.xpath(".//a:blip")
    )
    figure_element = deepcopy(figure_paragraph._p)
    _set_figure_alt_text(figure_element)
    _replace_figure_blob(document, figure_paragraph, figure_path)

    caption = next(
        p
        for p in section
        if p.text.strip().startswith("Fig. 6.")
    )
    caption_element = deepcopy(caption._p)
    caption_runs = caption_element.xpath("./w:r")
    if len(caption_runs) < 2:
        raise ValueError("Fig. 6 caption does not have the expected run structure")
    caption_texts = caption_runs[1].xpath(".//w:t")
    if not caption_texts:
        raise ValueError("Fig. 6 caption text element not found")
    caption_texts[0].text = (
        "Architecture of the spatial position-aware mask decoder. "
        "The dashed path denotes mask-guided attention between successive "
        "decoding stages."
    )

    body = document.element.body
    body_children = list(body)
    start = body_children.index(heading._p)
    end = body_children.index(next_heading._p)
    for element in body_children[start + 1 : end]:
        body.remove(element)

    bold_q0 = (
        '<msup><mi mathvariant="bold">Q</mi>'
        '<mrow><mo>(</mo><mn>0</mn><mo>)</mo></mrow></msup>'
    )
    qtfp = (
        '<msup><mi mathvariant="bold">q</mi><mi mathvariant="normal">T</mi></msup>'
        '<msub><mi mathvariant="bold">f</mi><mi>p</mi></msub>'
    )
    x_shape = (
        '<mi mathvariant="bold">X</mi><mo>∈</mo><msup><mi>R</mi>'
        '<mrow><mi>P</mi><mo>×</mo><mi>C</mi></mrow></msup>'
    )
    f_inst = (
        '<msub><mi mathvariant="bold">F</mi>'
        '<mi mathvariant="normal">inst</mi></msub>'
    )
    f_mask = (
        '<msub><mi mathvariant="bold">F</mi>'
        '<mi mathvariant="normal">mask</mi></msub>'
    )
    d_axis = (
        '<msub><mi>d</mi><mi>a</mi></msub><mo>=</mo><mo>⌊</mo>'
        '<mfrac><mi>d</mi><mn>3</mn></mfrac><mo>⌋</mo>'
    )
    n_freq = (
        '<msub><mi>n</mi><mi>f</mi></msub><mo>=</mo><mo>⌈</mo>'
        '<mfrac><msub><mi>d</mi><mi>a</mi></msub><mn>2</mn></mfrac><mo>⌉</mo>'
    )
    omega_value = '<mi>ω</mi><mo>=</mo><mn>10000</mn>'
    pe_shape = (
        '<msub><mi mathvariant="bold">P</mi><mi>e</mi></msub>'
        '<mo>∈</mo><msup><mi>R</mi>'
        '<mrow><mi>P</mi><mo>×</mo><mi>d</mi></mrow></msup>'
    )
    fp_inst = (
        '<msup><msub><mi mathvariant="bold">F</mi>'
        '<mi mathvariant="normal">inst</mi></msub><mo>′</mo></msup>'
    )
    fp_mask = (
        '<msup><msub><mi mathvariant="bold">F</mi>'
        '<mi mathvariant="normal">mask</mi></msub><mo>′</mo></msup>'
    )
    ell_range = (
        '<mi>ℓ</mi><mo>=</mo><mn>0</mn><mo>,</mo><mo>…</mo>'
        '<mo>,</mo><mn>6</mn>'
    )
    nqxp = (
        '<msub><mi>N</mi><mi>q</mi></msub><mo>×</mo><mi>P</mi>'
    )
    nqx_h = (
        '<msub><mi>N</mi><mi>q</mi></msub><mo>×</mo><mi>h</mi>'
    )
    px_h = '<mi>P</mi><mo>×</mo><mi>h</mi>'
    phi_q = '<msub><mi>φ</mi><mi>q</mi></msub>'
    phi_f = '<msub><mi>φ</mi><mi>f</mi></msub>'

    p = _new_paragraph(next_heading, normal_ppr)
    _append_parts(
        p,
        [
            (
                "SPMD receives instance queries initialized by IAQG. Rather than "
                "producing free-standing query embeddings, IAQG outputs the indices "
                "of selected foreground voxels. These indices retrieve the "
                "corresponding descriptors from the final backbone feature map. "
                "The retrieved instance descriptors are appended to learnable "
                "semantic-query embeddings along the query (token) dimension, and "
                "each token is projected to the decoder dimension to form the "
                "initial query matrix "
            ),
            ("math", bold_q0),
            (
                ', corresponding to the "Initial Queries (Q0)" block in Fig. 6. '
                "Thus, a selected query point is converted into a query vector by "
                "indexed feature retrieval rather than by directly encoding its "
                "3-D coordinate; the retrieved descriptor contains the local "
                "geometric and semantic context learned by the backbone."
            ),
        ],
        transform,
    )

    p = _new_paragraph(next_heading, normal_ppr)
    _append_parts(
        p,
        [
            (
                "In the baseline query-based Transformer decoder, cross-attention "
                "aggregates voxel features into the queries and a mask head produces "
                "voxelwise instance logits. However, the key and value features lack "
                "explicit 3-D positional identity, and the linear compatibility score "
            ),
            ("math", qtfp),
            (
                " is restricted to a dot product in the projected feature space. "
                "SPMD addresses these limitations by combining 3-D sinusoidal "
                "positional encoding, six-layer iterative query refinement, and a "
                "dual-branch multilayer-perceptron mask head in one decoding process."
            ),
        ],
        transform,
    )

    p = _new_paragraph(next_heading, normal_ppr)
    _append_parts(
        p,
        [
            "As shown in Fig. 6, the final backbone voxel features ",
            ("math", x_shape),
            (
                " enter two parallel projection branches, where P and C denote "
                "the numbers of voxels and backbone channels, respectively. The "
                "left branch applies a linear layer, layer normalization, and ReLU "
                "to produce the cross-attention feature matrix "
            ),
            ("math", f_inst),
            (
                "; the right branch applies a linear layer, ReLU, and a second "
                "linear layer to produce the mask feature matrix "
            ),
            ("math", f_mask),
            (
                ". In parallel, a 3-D sinusoidal encoder maps each discrete sparse-"
                "voxel grid coordinate (x, y, z) to a d-dimensional positional "
                "vector. Each spatial axis is assigned "
            ),
            ("math", d_axis),
            " channels and uses ",
            ("math", n_freq),
            " frequency scales. For coordinate p on axis a, frequency k is encoded as",
        ],
        transform,
    )
    next_heading._p.addprevious(equations[4])
    next_heading._p.addprevious(equations[5])

    p = _new_paragraph(next_heading, normal_ppr)
    _append_parts(
        p,
        [
            "where ",
            ("math", omega_value),
            (
                " is the exponential frequency-scaling base and "
                "k = 0, ..., n_f - 1. The implementation interleaves the sine and "
                "cosine components while retaining exactly d_a channels per axis. "
                "When d_a is odd, the final cosine channel is omitted. The three "
                "axiswise encodings are concatenated along the channel dimension, "
                "and the remaining d - 3d_a channels are zero-padded to obtain "
            ),
            ("math", pe_shape),
            (
                ". As indicated by the two addition nodes in Fig. 6, the same "
                "positional code is injected residually into both feature branches:"
            ),
        ],
        transform,
    )
    next_heading._p.addprevious(equations[6])
    next_heading._p.addprevious(equations[7])

    p = _new_paragraph(next_heading, normal_ppr)
    _append_parts(
        p,
        [
            "The position-enhanced ",
            ("math", fp_inst),
            (
                " provides the keys and values for cross-attention, while "
            ),
            ("math", bold_q0),
            (
                " supplies the initial queries. Each of the six decoder layers "
                "sequentially applies cross-attention, self-attention, and a "
                "feed-forward network. With iterative prediction, scores and masks "
                "are generated at stage "
            ),
            ("math", '<mi>ℓ</mi><mo>=</mo><mn>0</mn>'),
            (
                " before the first decoder layer and after each layer at stages "
            ),
            ("math", '<mi>ℓ</mi><mo>=</mo><mn>1</mn><mo>,</mo><mo>…</mo><mo>,</mo><mn>6</mn>'),
            (
                ", yielding seven prediction sets. At each nonfinal stage, mask "
                "logits are thresholded at 0.5 to form a detached Boolean attention "
                "mask for the next cross-attention layer; a row that masks all voxels "
                "is reset to prevent invalid attention. The stage-wise queries are "
                "layer-normalized before entering the score head and the query-side "
                "mask projection, whereas "
            ),
            ("math", fp_mask),
            (
                " remains the voxel-side input of the mask head. Positional identity "
                "is therefore available during both attention aggregation and mask "
                "prediction."
            ),
        ],
        transform,
    )

    next_heading._p.addprevious(figure_element)
    next_heading._p.addprevious(caption_element)

    p = _new_paragraph(next_heading, normal_ppr)
    _append_parts(
        p,
        [
            (
                "The MLP mask head in Fig. 6 contains two independently parameterized "
                "projection branches. The layer-normalized stage query and "
            ),
            ("math", fp_mask),
            (
                " are each mapped into a shared h-dimensional space by a "
                "linear-ReLU-linear network, producing the projected features q and f "
                "shown in the figure. Their matrix product yields the mask logits. "
                "The final compatibility remains bilinear in the projected space, "
                "while the two projections introduce nonlinear transformations of "
                "the original query and voxel features:"
            ),
        ],
        transform,
    )
    next_heading._p.addprevious(equations[8])
    next_heading._p.addprevious(equations[9])
    next_heading._p.addprevious(equations[10])

    p = _new_paragraph(next_heading, normal_ppr)
    _append_parts(
        p,
        [
            "where ",
            ("math", phi_q),
            " and ",
            ("math", phi_f),
            (
                " denote the query-side and voxel-side projection networks, "
                "respectively, and "
            ),
            ("math", ell_range),
            (
                " indexes the decoding stage. The projected query and voxel "
                "features have sizes "
            ),
            ("math", nqx_h),
            " and ",
            ("math", px_h),
            ", respectively, so the mask-logit matrix has size ",
            ("math", nqxp),
            (
                ", where N_q and P denote the numbers of queries and voxels. This "
                "formulation corresponds to the query-side and "
            ),
            ("math", fp_mask),
            (
                "-side MLP branches shown in Fig. 6 and preserves the voxelwise "
                "output size."
            ),
        ],
        transform,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    args = parser.parse_args()
    revise_spmd_section(
        args.input.resolve(),
        args.output.resolve(),
        args.figure.resolve(),
    )


if __name__ == "__main__":
    main()
