"""Build the InterveneSim-CF technical report PDF from frozen v0.4 results."""

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
SOURCE = ROOT / "paper" / "intervenesim-cf.md"
RESULTS = ROOT / "results" / "intervenesim-cf"
OUTPUT = ROOT / "output" / "pdf" / "intervenesim-cf-report.pdf"

INK = colors.HexColor("#111A33")
BLUE = colors.HexColor("#2563EB")
PALE = colors.HexColor("#EFF6FF")
LIGHT = colors.HexColor("#F8FAFC")
MUTED = colors.HexColor("#475569")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CFTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=31,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "CFSubtitle",
            parent=base["Normal"],
            fontSize=12.5,
            leading=18,
            textColor=MUTED,
        ),
        "h1": ParagraphStyle(
            "CFH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=10,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "CFBody",
            parent=base["BodyText"],
            fontSize=9.4,
            leading=14.1,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=8,
        ),
        "abstract": ParagraphStyle(
            "CFAbstract",
            parent=base["BodyText"],
            fontSize=9.5,
            leading=14.2,
            textColor=INK,
            leftIndent=12,
            rightIndent=12,
            borderColor=BLUE,
            borderWidth=1.3,
            borderPadding=11,
            backColor=PALE,
            spaceAfter=12,
        ),
        "caption": ParagraphStyle(
            "CFCaption",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10.5,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "metric": ParagraphStyle(
            "CFMetric",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=18,
            textColor=INK,
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "CFMetricLabel",
            parent=base["Normal"],
            fontSize=7.4,
            leading=9.5,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "small": ParagraphStyle(
            "CFSmall", parent=base["Normal"], fontSize=7.8, leading=10.5, textColor=MUTED
        ),
    }


def _clean(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.*?)`", r"<font name='Courier'>\1</font>", text)


def _page(canvas, document) -> None:  # noqa: ANN001
    canvas.saveState()
    width, _ = letter
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(0.65 * inch, 0.48 * inch, width - 0.65 * inch, 0.48 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.65 * inch, 0.28 * inch, "InterveneSim-CF | Ethan Villalovoz | 2026")
    canvas.drawRightString(width - 0.65 * inch, 0.28 * inch, str(document.page))
    canvas.restoreState()


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )


def _figure(path: Path, caption: str, styles: dict[str, ParagraphStyle], width: float = 6.4):
    with PILImage.open(path) as source:
        ratio = source.height / source.width
    image = Image(str(path), width=width * inch, height=width * ratio * inch)
    return KeepTogether([image, Paragraph(caption, styles["caption"])])


def _cover(styles: dict[str, ParagraphStyle]) -> list:
    metrics = Table(
        [
            [
                Paragraph("0.524", styles["metric"]),
                Paragraph("0.732", styles["metric"]),
                Paragraph("240", styles["metric"]),
            ],
            [
                Paragraph("T-learner PEHE", styles["metric_label"]),
                Paragraph("helpful AUROC", styles["metric_label"]),
                Paragraph("one-outcome episodes", styles["metric_label"]),
            ],
        ],
        colWidths=[2.2 * inch] * 3,
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
        Spacer(1, 0.2 * inch),
        Paragraph("INTERVENESIM-CF", styles["subtitle"]),
        Spacer(1, 0.08 * inch),
        Paragraph("Learning When to Ask<br/>from One Observed Future", styles["title"]),
        Paragraph(
            "A hidden-counterfactual benchmark for selective robot assistance",
            styles["subtitle"],
        ),
        Spacer(1, 0.18 * inch),
        HRFlowable(width="100%", thickness=3, color=BLUE),
        Spacer(1, 0.2 * inch),
        _figure(
            RESULTS / "effect_estimation.png",
            "Exact treatment-effect error and helpful-intervention ranking. Dots are factual "
            "logging seeds; paired supervision is privileged.",
            styles,
            width=6.55,
        ),
        metrics,
        Spacer(1, 0.25 * inch),
        Paragraph("Ethan Villalovoz", styles["subtitle"]),
        Paragraph("Independent research project | August 2026 | v0.4", styles["small"]),
        Spacer(1, 0.1 * inch),
        Paragraph(
            "Simulation-only. Free and reproducible on Apple Silicon or CPU. No CUDA, cloud "
            "compute, paid API, physical robot, or human-subject claim.",
            styles["small"],
        ),
        PageBreak(),
    ]


def _results_table() -> Table:
    metrics = pd.read_csv(RESULTS / "counterfactual_metrics.csv")
    aggregate = pd.read_csv(RESULTS / "gate_aggregate.csv")
    metric_mean = (
        metrics.loc[metrics["logging_scheme"] == "randomized"]
        .groupby("method")
        .mean(numeric_only=True)
    )
    labels = {
        "paired_label_oracle": "Paired supervision",
        "t_learner": "T-learner",
        "dr_learner": "DR learner",
        "reversibility_proxy": "Reversibility",
        "failure_risk": "Failure risk",
    }
    rows = [["Method", "PEHE", "AUROC", "Success @25%", "Success @50%"]]
    for method in labels:
        policy = aggregate.loc[
            (aggregate["logging_scheme"] == "randomized") & (aggregate["method"] == method)
        ].set_index("target_budget")
        rows.append(
            [
                labels[method],
                f"{metric_mean.loc[method, 'pehe']:.3f}",
                f"{metric_mean.loc[method, 'helpful_auroc']:.3f}",
                f"{100 * policy.loc[0.25, 'success_mean']:.1f}%",
                f"{100 * policy.loc[0.50, 'success_mean']:.1f}%",
            ]
        )
    table = Table(rows, colWidths=[1.45 * inch, 0.85 * inch, 0.85 * inch, 1.15 * inch, 1.15 * inch])
    table.setStyle(_table_style())
    return table


def _body(styles: dict[str, ParagraphStyle]) -> list:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story: list = []
    paragraphs: list[str] = []
    bullets: list[str] = []
    in_abstract = False

    def flush_paragraphs() -> None:
        nonlocal paragraphs
        if paragraphs:
            style = styles["abstract"] if in_abstract else styles["body"]
            story.append(Paragraph(_clean(" ".join(paragraphs)), style))
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
            if title not in {"Abstract", "1. Motivation"}:
                story.append(PageBreak())
            story.append(Paragraph(title, styles["h1"]))
            if title.startswith("5."):
                story.extend(
                    [
                        _figure(
                            RESULTS / "single_world_frontier.png",
                            "Figure 2. Equal-budget success on unseen policy seeds. Every method "
                            "receives the exact same episode-level intervention budget.",
                            styles,
                            width=5.65,
                        ),
                        _results_table(),
                        Spacer(1, 8),
                    ]
                )
            elif title.startswith("6."):
                story.extend(
                    [
                        _figure(
                            RESULTS / "open_world.png",
                            "Figure 3. Leave-one-task-out and leave-one-disturbance-out success "
                            "near 25% intervention.",
                            styles,
                            width=6.2,
                        ),
                        _figure(
                            RESULTS / "latency_audit.png",
                            "Figure 4. Immediate and delayed assistance under the same learned "
                            "gate and exact autonomous trajectories.",
                            styles,
                            width=5.8,
                        ),
                    ]
                )
            continue
        if stripped.startswith("- "):
            flush_paragraphs()
            bullets.append(stripped[2:])
        elif not stripped:
            flush_paragraphs()
            flush_bullets()
        elif not stripped.startswith("# "):
            paragraphs.append(stripped)
    flush_paragraphs()
    flush_bullets()
    return story


def main() -> None:
    required = [
        RESULTS / "effect_estimation.png",
        RESULTS / "single_world_frontier.png",
        RESULTS / "open_world.png",
        RESULTS / "latency_audit.png",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"run selective-research before building the paper: {missing}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.62 * inch,
        title="InterveneSim-CF: Learning When to Ask from One Observed Future",
        author="Ethan Villalovoz",
        subject="Single-world causal estimation for simulated robot assistance",
    )
    document.build(_cover(styles) + _body(styles), onFirstPage=_page, onLaterPages=_page)
    print(OUTPUT)


if __name__ == "__main__":
    main()
