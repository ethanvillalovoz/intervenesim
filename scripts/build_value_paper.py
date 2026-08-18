"""Build the InterveneSim-Value technical report PDF from frozen results."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
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
SOURCE = ROOT / "paper" / "intervenesim-value.md"
RESULTS = ROOT / "results" / "intervenesim-value"
OUTPUT = ROOT / "output" / "pdf" / "intervenesim-value-report.pdf"

INK = colors.HexColor("#111A33")
BLUE = colors.HexColor("#2563EB")
LIGHT = colors.HexColor("#F8FAFC")
PALE = colors.HexColor("#EFF6FF")
MUTED = colors.HexColor("#475569")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleValue",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=29,
            leading=33,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleValue",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=MUTED,
        ),
        "h1": ParagraphStyle(
            "H1Value",
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
            "BodyValue",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14.1,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=8,
        ),
        "abstract": ParagraphStyle(
            "AbstractValue",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.7,
            leading=14.5,
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
            "CaptionValue",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.2,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=11,
        ),
        "metric": ParagraphStyle(
            "MetricValue",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=INK,
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabelValue",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "small": ParagraphStyle(
            "SmallValue",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.5,
            textColor=MUTED,
        ),
    }


def _clean(text: str) -> str:
    text = text.replace("+/-", "+/-")
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.*?)`", r"<font name='Courier'>\1</font>", text)


def _page(canvas, document) -> None:  # noqa: ANN001
    canvas.saveState()
    width, _ = letter
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(0.65 * inch, 0.48 * inch, width - 0.65 * inch, 0.48 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.65 * inch, 0.28 * inch, "InterveneSim-Value | Ethan Villalovoz | 2026")
    canvas.drawRightString(width - 0.65 * inch, 0.28 * inch, str(document.page))
    canvas.restoreState()


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.8),
            ("LEADING", (0, 0), (-1, -1), 9.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]
    )


def _cover(styles: dict[str, ParagraphStyle]) -> list:
    preview = Image(str(RESULTS / "comparison-preview.png"), width=7.05 * inch, height=2.64 * inch)
    metrics = Table(
        [
            [
                Paragraph("0.817", styles["metric"]),
                Paragraph("+8.1 pp", styles["metric"]),
                Paragraph("5 / 5", styles["metric"]),
            ],
            [
                Paragraph("held-out value AUROC", styles["metric_label"]),
                Paragraph("value vs. risk near 25%", styles["metric_label"]),
                Paragraph("cross-seed wins over risk", styles["metric_label"]),
            ],
        ],
        colWidths=[2.25 * inch] * 3,
        rowHeights=[0.34 * inch, 0.35 * inch],
    )
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
        Paragraph("INTERVENESIM-VALUE", styles["subtitle"]),
        Spacer(1, 0.08 * inch),
        Paragraph("Will Help Change<br/>the Outcome?", styles["title"]),
        Paragraph(
            "Learning causal value of expert takeover from exact simulator forks",
            styles["subtitle"],
        ),
        Spacer(1, 0.22 * inch),
        HRFlowable(width="100%", thickness=3, color=BLUE),
        Spacer(1, 0.22 * inch),
        preview,
        Paragraph(
            "Exact held-out simulator fork: autonomous continuation fails (left); "
            "scripted expert takeover succeeds (right).",
            styles["caption"],
        ),
        Spacer(1, 0.1 * inch),
        metrics,
        Spacer(1, 0.28 * inch),
        Paragraph("Ethan Villalovoz", styles["subtitle"]),
        Paragraph("Independent research project | August 2026 | v0.3", styles["small"]),
        Spacer(1, 0.12 * inch),
        Paragraph(
            "Simulation-only. Runs locally on Apple Silicon or CPU. No CUDA, cloud "
            "compute, paid API, or robot hardware required.",
            styles["small"],
        ),
        PageBreak(),
    ]


def _figure(path: Path, caption: str, styles: dict[str, ParagraphStyle], width: float = 6.65):
    with PILImage.open(path) as image:
        ratio = image.height / image.width
    graphic = Image(str(path), width=width * inch, height=width * ratio * inch)
    return KeepTogether([graphic, Paragraph(caption, styles["caption"])])


def _primary_table() -> Table:
    summary = pd.read_csv(RESULTS / "gate_summary.csv")
    selected = summary.loc[
        summary["method"].isin(["value_gate", "risk_gate", "uncertainty_gate"])
        & summary["target_budget"].isin([0.25, 0.5])
    ]
    labels = {"value_gate": "Value", "risk_gate": "Failure risk", "uncertainty_gate": "Uncertainty"}
    rows = [["Gate", "Target", "Success", "Intervention", "Useful requests", "Oracle regret"]]
    for row in selected.itertuples():
        rows.append(
            [
                labels[row.method],
                f"{100 * row.target_budget:.0f}%",
                f"{100 * row.success_rate:.1f}%",
                f"{100 * row.intervention_rate:.1f}%",
                f"{100 * row.request_precision:.1f}%",
                f"{100 * row.regret_to_matched_oracle:.1f} pp",
            ]
        )
    table = Table(
        rows, colWidths=[1.15 * inch, 0.65 * inch, 0.9 * inch, 1.05 * inch, 1.1 * inch, 1.05 * inch]
    )
    table.setStyle(_table_style())
    return table


def _crossval_table() -> Table:
    paired = pd.read_csv(RESULTS / "crossval_paired.csv")
    rows = [["Target", "Reference", "Mean value advantage", "Std.", "Positive seeds"]]
    labels = {"risk_gate": "Failure risk", "uncertainty_gate": "Uncertainty"}
    for row in paired.itertuples():
        rows.append(
            [
                f"{100 * row.budget:.0f}%",
                labels[row.reference],
                f"{100 * row.mean_delta:+.1f} pp",
                f"{100 * row.std_delta:.1f} pp",
                f"{row.positive_seeds}/{row.policy_seeds}",
            ]
        )
    table = Table(rows, colWidths=[0.8 * inch, 1.3 * inch, 1.7 * inch, 1.0 * inch, 1.2 * inch])
    table.setStyle(_table_style())
    return table


def _visual_table() -> Table:
    metrics = pd.read_json(RESULTS / "visual" / "visual_metrics.json").T
    rows = [["Camera", "AUROC", "Average precision", "Recall @ 0.5", "Positive rate"]]
    for camera in ("frontview", "agentview"):
        row = metrics.loc[camera]
        rows.append(
            [
                camera,
                f"{row.auroc:.3f}",
                f"{row.average_precision:.3f}",
                f"{100 * row.recall:.1f}%",
                f"{100 * row.positive_rate:.1f}%",
            ]
        )
    table = Table(rows, colWidths=[1.1 * inch, 0.9 * inch, 1.4 * inch, 1.25 * inch, 1.15 * inch])
    table.setStyle(_table_style())
    return table


def _paper_body(styles: dict[str, ParagraphStyle]) -> list:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story: list = []
    paragraphs: list[str] = []
    bullets: list[str] = []
    in_abstract = False

    def flush_paragraphs() -> None:
        nonlocal paragraphs
        if paragraphs:
            style = styles["abstract"] if in_abstract else styles["body"]
            story.append(Paragraph(_clean(" ".join(p.strip() for p in paragraphs)), style))
            paragraphs = []

    def flush_bullets() -> None:
        nonlocal bullets
        if bullets:
            items = [ListItem(Paragraph(_clean(item), styles["body"])) for item in bullets]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=18, bulletColor=BLUE))
            bullets = []

    for line in lines[4:]:
        stripped = line.strip()
        if stripped.startswith("## "):
            flush_paragraphs()
            flush_bullets()
            title = stripped[3:]
            in_abstract = title == "Abstract"
            if title not in {"Abstract", "1. Research question"}:
                story.append(PageBreak())
            story.append(Paragraph(title, styles["h1"]))
            if title.startswith("4."):
                story.extend(
                    [
                        _figure(
                            RESULTS / "value_frontier.png",
                            "Figure 1. Primary success-intervention frontier on two unseen "
                            "policy seeds. Crosses show never, always, and causal-oracle controls.",
                            styles,
                            width=5.7,
                        ),
                        _primary_table(),
                        Spacer(1, 8),
                    ]
                )
            elif title.startswith("5."):
                story.extend(
                    [
                        _figure(
                            RESULTS / "crossval_success.png",
                            "Figure 2. Exploratory five-fold policy-seed audit. Dots are "
                            "held-out policy seeds; bars show mean and standard deviation.",
                            styles,
                            width=5.7,
                        ),
                        _crossval_table(),
                        Spacer(1, 8),
                    ]
                )
            elif title.startswith("6."):
                story.extend(
                    [
                        _figure(
                            RESULTS / "visual" / "visual_camera_montage.png",
                            "Figure 3. Matched held-out-policy states from front-view and "
                            "zero-shot agent-view cameras.",
                            styles,
                            width=6.65,
                        ),
                        _visual_table(),
                        Spacer(1, 8),
                    ]
                )
            continue
        if not stripped:
            flush_paragraphs()
            flush_bullets()
        elif stripped.startswith("- "):
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
        title="InterveneSim-Value: Will Help Change the Outcome?",
        author="Ethan Villalovoz",
        subject="Causal value of expert intervention in simulated robot manipulation",
    )
    document.build(_cover(styles) + _paper_body(styles), onFirstPage=_page, onLaterPages=_page)
    return OUTPUT


if __name__ == "__main__":
    print(build())
