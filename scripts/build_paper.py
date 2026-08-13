"""Build the polished InterveneSim-X technical report PDF from frozen results."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "intervenesim-x.md"
RESULTS = ROOT / "results" / "intervenesim-x"
OUTPUT = ROOT / "output" / "pdf" / "intervenesim-x-report.pdf"

INK = colors.HexColor("#14213D")
BLUE = colors.HexColor("#2563EB")
PURPLE = colors.HexColor("#6D28D9")
MINT = colors.HexColor("#10B981")
PALE = colors.HexColor("#EFF6FF")
LIGHT = colors.HexColor("#F8FAFC")
MUTED = colors.HexColor("#475569")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=29,
            leading=33,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=MUTED,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=10,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.6,
            leading=14.3,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=8,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            textColor=INK,
            leftIndent=12,
            rightIndent=12,
            borderColor=BLUE,
            borderWidth=1.5,
            borderPadding=12,
            backColor=PALE,
            spaceAfter=12,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.2,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=11,
        ),
        "metric": ParagraphStyle(
            "Metric",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=INK,
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.5,
            textColor=MUTED,
        ),
    }


def _clean(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.*?)`", r"<font name='Courier'>\1</font>", text)
    return text


def _page(canvas, document) -> None:  # noqa: ANN001
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(0.65 * inch, 0.48 * inch, width - 0.65 * inch, 0.48 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.65 * inch, 0.28 * inch, "InterveneSim-X | Ethan Villalovoz | 2026")
    canvas.drawRightString(width - 0.65 * inch, 0.28 * inch, str(document.page))
    canvas.restoreState()


def _cover(styles: dict[str, ParagraphStyle]) -> list:
    preview = Image(str(RESULTS / "comparison-preview.png"), width=7.05 * inch, height=2.64 * inch)
    metric_data = [
        [
            Paragraph("+22.8 pp", styles["metric"]),
            Paragraph("5 / 5", styles["metric"]),
            Paragraph("0.919", styles["metric"]),
        ],
        [
            Paragraph("recovery vs. clean labels", styles["metric_label"]),
            Paragraph("training-seed wins", styles["metric_label"]),
            Paragraph("temporal risk AUROC", styles["metric_label"]),
        ],
    ]
    metrics = Table(metric_data, colWidths=[2.25 * inch] * 3, rowHeights=[0.34 * inch, 0.35 * inch])
    metrics.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return [
        Spacer(1, 0.18 * inch),
        Paragraph("INTERVENESIM-X", styles["subtitle"]),
        Spacer(1, 0.08 * inch),
        Paragraph("What Should a Robot<br/>Learn From a Correction?", styles["title"]),
        Paragraph(
            "A controlled study of correction value, rejected actions, and selective help "
            "in simulated manipulation",
            styles["subtitle"],
        ),
        Spacer(1, 0.22 * inch),
        HRFlowable(width="100%", thickness=3, color=BLUE),
        Spacer(1, 0.22 * inch),
        preview,
        Paragraph(
            "Matched cereal gripper-slip rollout: baseline failure (left), recovery-data "
            "success (right). Identical evaluation seed and disturbance.",
            styles["caption"],
        ),
        Spacer(1, 0.1 * inch),
        metrics,
        Spacer(1, 0.28 * inch),
        Paragraph("Ethan Villalovoz", styles["subtitle"]),
        Paragraph("Independent research project | August 2026", styles["small"]),
        Spacer(1, 0.12 * inch),
        Paragraph(
            "Simulation-only. Runs locally on Apple Silicon or CPU. No CUDA, cloud compute, "
            "paid API, or robot hardware required.",
            styles["small"],
        ),
        PageBreak(),
    ]


def _result_table(styles: dict[str, ParagraphStyle]) -> Table:
    summary = pd.read_csv(RESULTS / "autonomous_summary.csv")
    disturbed = summary.loc[summary["disturbed"].astype(str).str.lower() == "true"].copy()
    order = ["baseline", "more_demos", "recovery_bc", "contrastive_recovery"]
    labels = {
        "baseline": "Baseline",
        "more_demos": "+ Clean labels",
        "recovery_bc": "+ Recovery labels",
        "contrastive_recovery": "+ Recovery + contrast",
    }
    rows = [["Condition", "Success", "Std.", "Hierarchical 95% CI"]]
    indexed = disturbed.set_index("condition")
    for condition in order:
        row = indexed.loc[condition]
        rows.append(
            [
                labels[condition],
                f"{100 * row['mean']:.1f}%",
                f"{100 * row['std']:.1f} pp",
                f"[{100 * row['ci95_low']:.1f}, {100 * row['ci95_high']:.1f}]",
            ]
        )
    table = Table(rows, colWidths=[2.45 * inch, 1.05 * inch, 1.0 * inch, 1.7 * inch])
    table.setStyle(_table_style())
    return table


def _help_table() -> Table:
    sweep = pd.read_csv(RESULTS / "help_threshold_sweep.csv")
    rows = [["Threshold", "Disturbed success", "Help on disturbed", "Help on nominal"]]
    for row in sweep.itertuples():
        rows.append(
            [
                f"{row.threshold:.2f}",
                f"{100 * row.disturbed_success_rate:.1f}%",
                f"{100 * row.disturbed_intervention_rate:.1f}%",
                f"{100 * row.nominal_intervention_rate:.1f}%",
            ]
        )
    table = Table(rows, colWidths=[1.0 * inch, 1.65 * inch, 1.75 * inch, 1.55 * inch])
    table.setStyle(_table_style())
    return table


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("LEADING", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]
    )


def _figure(filename: str, caption: str, styles: dict[str, ParagraphStyle], width: float = 6.7):
    path = RESULTS / filename
    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        ratio = image.height / image.width
    graphic = Image(str(path), width=width * inch, height=width * ratio * inch)
    return KeepTogether([graphic, Paragraph(caption, styles["caption"])])


def _paper_body(styles: dict[str, ParagraphStyle]) -> list:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story: list = []
    paragraphs: list[str] = []
    bullets: list[str] = []
    in_abstract = False

    def flush_paragraphs() -> None:
        nonlocal paragraphs
        if paragraphs:
            text = " ".join(part.strip() for part in paragraphs)
            style = styles["abstract"] if in_abstract else styles["body"]
            story.append(Paragraph(_clean(text), style))
            paragraphs = []

    def flush_bullets() -> None:
        nonlocal bullets
        if bullets:
            items = [ListItem(Paragraph(_clean(item), styles["body"])) for item in bullets]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=18, bulletColor=BLUE))
            story.append(Spacer(1, 4))
            bullets = []

    for line in lines[4:]:
        stripped = line.strip()
        if stripped.startswith("## "):
            flush_paragraphs()
            flush_bullets()
            title = stripped[3:]
            in_abstract = title == "Abstract"
            if title not in {"Abstract", "1. Motivation"}:
                story.append(PageBreak())
            story.append(Paragraph(title, styles["h1"]))
            if title.startswith("3."):
                story.extend(
                    [
                        _figure(
                            "multiseed_success.png",
                            "Figure 1. Disturbed success across five independent training seeds. "
                            "Bars show means and standard deviations; points are individual seeds.",
                            styles,
                        ),
                        _result_table(styles),
                        Spacer(1, 10),
                    ]
                )
            elif title.startswith("5."):
                story.extend(
                    [
                        _figure(
                            "help_efficiency.png",
                            "Figure 3. Assisted success versus expert-intervention rate. Purple "
                            "points show the complete exploratory threshold sweep.",
                            styles,
                        ),
                        _help_table(),
                        Spacer(1, 10),
                    ]
                )
            elif title.startswith("4."):
                story.append(
                    _figure(
                        "budget_efficiency.png",
                        "Figure 2. Reference-seed disturbed success across equal added-label "
                        "budgets. The recovery curve is not monotonic at the final budget.",
                        styles,
                    )
                )
            continue
        if not stripped:
            flush_paragraphs()
            flush_bullets()
            continue
        if stripped.startswith("- "):
            flush_paragraphs()
            bullets.append(stripped[2:])
        else:
            flush_bullets()
            paragraphs.append(stripped)
    flush_paragraphs()
    flush_bullets()
    return story


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.62 * inch,
        title="InterveneSim-X: What Should a Robot Learn From a Correction?",
        author="Ethan Villalovoz",
        subject="Intervention-guided imitation learning in simulated manipulation",
    )
    story = _cover(styles) + _paper_body(styles)
    document.build(story, onFirstPage=_page, onLaterPages=_page)
    return OUTPUT


if __name__ == "__main__":
    print(build())
