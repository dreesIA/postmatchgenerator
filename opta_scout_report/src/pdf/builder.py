"""PDF builder for the scout report."""

import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, Flowable
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas

from .styles import ReportStyles
from ..report_sections.base import SectionContent


class PageBackground(Flowable):
    """Custom flowable to draw page background."""

    def __init__(self, styles: ReportStyles):
        Flowable.__init__(self)
        self.styles = styles
        self.width = styles.layout.page_width
        self.height = styles.layout.page_height

    def draw(self):
        self.canv.setFillColor(self.styles.background_color)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)


class PDFBuilder:
    """Build PDF scout reports from section content."""

    def __init__(self, output_path: str, player_name: str, player_team: str,
                 position: str, season: str = "2024/25"):
        """
        Initialize PDF builder.

        Args:
            output_path: Path for output PDF file
            player_name: Name of the player
            player_team: Player's team name
            position: Player's position
            season: Season string
        """
        self.output_path = Path(output_path)
        self.player_name = player_name
        self.player_team = player_team
        self.position = position
        self.season = season

        self.styles = ReportStyles()
        self.story = []
        self.page_number = 0

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def build(self, sections: List[SectionContent]) -> str:
        """
        Build the complete PDF report.

        Args:
            sections: List of SectionContent objects to include

        Returns:
            Path to the generated PDF file
        """
        # Create document
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=letter,
            topMargin=self.styles.layout.margin_top,
            bottomMargin=self.styles.layout.margin_bottom,
            leftMargin=self.styles.layout.margin_left,
            rightMargin=self.styles.layout.margin_right,
        )

        self.story = []

        # Add cover page
        self._add_cover_page()

        # Add table of contents
        self._add_table_of_contents(sections)

        # Add each section
        for i, section in enumerate(sections):
            self._add_section(section, i + 1)

        # Build PDF
        doc.build(
            self.story,
            onFirstPage=self._draw_page_background,
            onLaterPages=self._draw_page_with_header_footer,
        )

        return str(self.output_path)

    def _draw_page_background(self, canvas: canvas.Canvas, doc) -> None:
        """Draw background on first page (cover)."""
        canvas.saveState()
        canvas.setFillColor(self.styles.background_color)
        canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1, stroke=0)
        canvas.restoreState()

    def _draw_page_with_header_footer(self, canvas: canvas.Canvas, doc) -> None:
        """Draw background, header, and footer on content pages."""
        canvas.saveState()

        # Background
        canvas.setFillColor(self.styles.background_color)
        canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1, stroke=0)

        # Header
        canvas.setFillColor(colors.HexColor(self.styles.colors.text_secondary))
        canvas.setFont('Helvetica', 8)
        header_text = f"{self.player_name} | {self.position} | {self.player_team}"
        canvas.drawString(
            self.styles.layout.margin_left,
            doc.pagesize[1] - 0.5 * inch,
            header_text
        )

        # Date on right
        date_text = datetime.now().strftime("%B %d, %Y")
        canvas.drawRightString(
            doc.pagesize[0] - self.styles.layout.margin_right,
            doc.pagesize[1] - 0.5 * inch,
            date_text
        )

        # Footer with page number
        page_num = canvas.getPageNumber()
        footer_text = f"Page {page_num} | Scout Report | {self.season}"
        canvas.drawCentredString(
            doc.pagesize[0] / 2,
            0.5 * inch,
            footer_text
        )

        canvas.restoreState()

    def _add_cover_page(self) -> None:
        """Add the cover page."""
        # Spacer to center content
        self.story.append(Spacer(1, 2 * inch))

        # Title
        self.story.append(Paragraph(
            "PLAYER SCOUT REPORT",
            self.styles.get_style('ReportTitle')
        ))

        self.story.append(Spacer(1, 0.5 * inch))

        # Player name
        self.story.append(Paragraph(
            self.player_name.upper(),
            self.styles.get_style('ReportTitle')
        ))

        self.story.append(Spacer(1, 0.3 * inch))

        # Team and position
        self.story.append(Paragraph(
            f"{self.player_team} | {self.position}",
            self.styles.get_style('ReportSubtitle')
        ))

        self.story.append(Spacer(1, 0.5 * inch))

        # Season
        self.story.append(Paragraph(
            f"Season {self.season}",
            self.styles.get_style('ReportSubtitle')
        ))

        self.story.append(Spacer(1, 2 * inch))

        # Generation date
        date_str = datetime.now().strftime("%B %d, %Y")
        self.story.append(Paragraph(
            f"Generated: {date_str}",
            self.styles.get_style('PageFooter')
        ))

        self.story.append(PageBreak())

    def _add_table_of_contents(self, sections: List[SectionContent]) -> None:
        """Add table of contents page."""
        self.story.append(Paragraph(
            "Table of Contents",
            self.styles.get_style('SectionTitle')
        ))

        self.story.append(Spacer(1, 0.3 * inch))

        # TOC entries
        for i, section in enumerate(sections):
            toc_text = f"{i + 1}. {section.title}"
            self.story.append(Paragraph(
                toc_text,
                self.styles.get_style('BodyText')
            ))
            self.story.append(Spacer(1, 0.1 * inch))

        self.story.append(PageBreak())

    def _add_section(self, section: SectionContent, section_number: int) -> None:
        """Add a section to the report."""
        # Section title
        self.story.append(Paragraph(
            f"{section_number}. {section.title}",
            self.styles.get_style('SectionTitle')
        ))

        self.story.append(Spacer(1, 0.2 * inch))

        # Add figures with captions
        for fig, caption in zip(section.figures, section.figure_captions):
            self._add_figure(fig, caption)

        # Add takeaways
        if section.takeaways:
            self.story.append(Paragraph(
                "Key Insights",
                self.styles.get_style('SectionSubtitle')
            ))
            self.story.append(Spacer(1, 0.1 * inch))

            for takeaway in section.takeaways:
                bullet = "+" if takeaway.is_positive else "-"
                style_name = 'TakeawayPositive' if takeaway.is_positive else 'TakeawayNegative'

                self.story.append(Paragraph(
                    f"{bullet} {takeaway.text}",
                    self.styles.get_style(style_name)
                ))

            self.story.append(Spacer(1, 0.2 * inch))

        # Add tables
        for table_data, table_title in zip(section.tables, section.table_titles):
            self._add_table(table_data, table_title)

        self.story.append(PageBreak())

    def _add_figure(self, fig, caption: str) -> None:
        """Add a matplotlib figure to the report."""
        # Save figure to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)

        # Calculate size to fit page width while maintaining aspect ratio
        fig_width = fig.get_figwidth()
        fig_height = fig.get_figheight()
        aspect_ratio = fig_height / fig_width

        max_width = self.styles.layout.content_width
        max_height = 4 * inch  # Maximum figure height

        img_width = min(max_width, 6 * inch)
        img_height = img_width * aspect_ratio

        if img_height > max_height:
            img_height = max_height
            img_width = img_height / aspect_ratio

        # Create image
        img = Image(buf, width=img_width, height=img_height)

        # Keep figure and caption together
        elements = [
            img,
            Spacer(1, 0.1 * inch),
            Paragraph(caption, self.styles.get_style('FigureCaption')),
            Spacer(1, 0.2 * inch),
        ]

        self.story.append(KeepTogether(elements))

    def _add_table(self, table_data: Dict[str, Any], title: str) -> None:
        """Add a metrics table to the report."""
        self.story.append(Paragraph(
            title,
            self.styles.get_style('SectionSubtitle')
        ))
        self.story.append(Spacer(1, 0.1 * inch))

        # Convert dict to table data
        rows = [['Metric', 'Value', 'Percentile']]

        for metric_name, data in table_data.items():
            display_name = metric_name.replace('_', ' ').title()
            value = data.get('value', 0)
            percentile = data.get('percentile', 50)

            # Format value
            if 'rate' in metric_name.lower() or 'accuracy' in metric_name.lower():
                value_str = f"{value:.1f}%"
            elif value >= 10:
                value_str = f"{value:.1f}"
            else:
                value_str = f"{value:.2f}"

            percentile_str = f"{percentile:.0f}th"

            rows.append([display_name, value_str, percentile_str])

        # Create table
        table = Table(rows, colWidths=[3 * inch, 1.5 * inch, 1.5 * inch])
        table.setStyle(TableStyle(self.styles.get_table_style()))

        self.story.append(table)
        self.story.append(Spacer(1, 0.3 * inch))

    def add_executive_summary(self, player_info: Dict[str, Any],
                             recommendation: Dict[str, Any],
                             key_metrics: Dict[str, Any]) -> None:
        """Add a custom executive summary page."""
        self.story.append(Paragraph(
            "Executive Summary",
            self.styles.get_style('SectionTitle')
        ))
        self.story.append(Spacer(1, 0.2 * inch))

        # Player info
        info_text = f"""
        <b>Name:</b> {player_info.get('name', 'Unknown')}<br/>
        <b>Age:</b> {player_info.get('age', 'N/A')}<br/>
        <b>Team:</b> {player_info.get('team', 'Unknown')}<br/>
        <b>Position:</b> {player_info.get('position', 'Unknown')}<br/>
        <b>Minutes:</b> {player_info.get('total_minutes', 0):.0f}<br/>
        <b>Matches:</b> {player_info.get('matches_played', 0)}
        """
        self.story.append(Paragraph(info_text, self.styles.get_style('BodyText')))
        self.story.append(Spacer(1, 0.3 * inch))

        # Recommendation verdict
        verdict = recommendation.get('verdict', 'MONITOR')
        style_name = f'Verdict{verdict.title()}'
        if style_name not in ['VerdictRecommend', 'VerdictMonitor', 'VerdictPass']:
            style_name = 'VerdictMonitor'

        self.story.append(Paragraph(verdict, self.styles.get_style(style_name)))
        self.story.append(Spacer(1, 0.2 * inch))

        # Scores
        fit_score = recommendation.get('fit_score', 50)
        risk_score = recommendation.get('risk_score', 30)

        scores_data = [
            ['Fit Score', 'Risk Score'],
            [f"{fit_score:.0f}/100", f"{risk_score:.0f}/100"]
        ]
        scores_table = Table(scores_data, colWidths=[2.5 * inch, 2.5 * inch])
        scores_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, 1), 24),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor(self.styles.colors.text)),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
        ]))
        self.story.append(scores_table)

        self.story.append(PageBreak())

    def save(self) -> str:
        """
        Alternative method to build and save the PDF.

        Returns:
            Path to the generated PDF file
        """
        # This is called if sections are added manually
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=letter,
            topMargin=self.styles.layout.margin_top,
            bottomMargin=self.styles.layout.margin_bottom,
            leftMargin=self.styles.layout.margin_left,
            rightMargin=self.styles.layout.margin_right,
        )

        doc.build(
            self.story,
            onFirstPage=self._draw_page_background,
            onLaterPages=self._draw_page_with_header_footer,
        )

        return str(self.output_path)
