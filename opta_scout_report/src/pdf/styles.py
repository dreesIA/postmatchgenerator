"""PDF styling configuration for the scout report."""

from dataclasses import dataclass, field
from typing import Dict, Tuple
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


@dataclass
class ColorScheme:
    """Color scheme for the report."""
    primary: str = '#1a1a2e'
    secondary: str = '#16213e'
    accent: str = '#0f3460'
    highlight: str = '#e94560'
    positive: str = '#4ade80'
    negative: str = '#f87171'
    neutral: str = '#94a3b8'
    text: str = '#f1f5f9'
    text_secondary: str = '#94a3b8'
    white: str = '#ffffff'
    black: str = '#000000'

    def to_reportlab(self, color_name: str) -> colors.Color:
        """Convert hex color to reportlab color."""
        hex_color = getattr(self, color_name, self.text)
        return colors.HexColor(hex_color)


@dataclass
class PageLayout:
    """Page layout configuration."""
    page_width: float = 8.5 * inch
    page_height: float = 11 * inch
    margin_top: float = 0.75 * inch
    margin_bottom: float = 0.75 * inch
    margin_left: float = 0.75 * inch
    margin_right: float = 0.75 * inch

    @property
    def content_width(self) -> float:
        return self.page_width - self.margin_left - self.margin_right

    @property
    def content_height(self) -> float:
        return self.page_height - self.margin_top - self.margin_bottom


class ReportStyles:
    """Styles for the scout report PDF."""

    def __init__(self):
        self.colors = ColorScheme()
        self.layout = PageLayout()
        self._styles = getSampleStyleSheet()
        self._custom_styles = {}
        self._create_custom_styles()

    def _create_custom_styles(self) -> None:
        """Create custom paragraph styles."""

        # Title style (cover page)
        self._custom_styles['ReportTitle'] = ParagraphStyle(
            'ReportTitle',
            parent=self._styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=36,
            leading=42,
            textColor=colors.HexColor(self.colors.white),
            alignment=TA_CENTER,
            spaceAfter=20,
        )

        # Subtitle style
        self._custom_styles['ReportSubtitle'] = ParagraphStyle(
            'ReportSubtitle',
            parent=self._styles['Normal'],
            fontName='Helvetica',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor(self.colors.text_secondary),
            alignment=TA_CENTER,
            spaceAfter=30,
        )

        # Section title
        self._custom_styles['SectionTitle'] = ParagraphStyle(
            'SectionTitle',
            parent=self._styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=24,
            leading=28,
            textColor=colors.HexColor(self.colors.highlight),
            alignment=TA_LEFT,
            spaceBefore=10,
            spaceAfter=15,
            borderPadding=5,
        )

        # Section subtitle
        self._custom_styles['SectionSubtitle'] = ParagraphStyle(
            'SectionSubtitle',
            parent=self._styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            textColor=colors.HexColor(self.colors.text),
            alignment=TA_LEFT,
            spaceBefore=10,
            spaceAfter=8,
        )

        # Body text
        self._custom_styles['BodyText'] = ParagraphStyle(
            'BodyText',
            parent=self._styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor(self.colors.text),
            alignment=TA_LEFT,
            spaceBefore=4,
            spaceAfter=4,
        )

        # Takeaway - positive
        self._custom_styles['TakeawayPositive'] = ParagraphStyle(
            'TakeawayPositive',
            parent=self._styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor(self.colors.positive),
            alignment=TA_LEFT,
            leftIndent=15,
            spaceBefore=4,
            spaceAfter=4,
            bulletIndent=5,
        )

        # Takeaway - negative
        self._custom_styles['TakeawayNegative'] = ParagraphStyle(
            'TakeawayNegative',
            parent=self._styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor(self.colors.negative),
            alignment=TA_LEFT,
            leftIndent=15,
            spaceBefore=4,
            spaceAfter=4,
            bulletIndent=5,
        )

        # Takeaway - neutral
        self._custom_styles['TakeawayNeutral'] = ParagraphStyle(
            'TakeawayNeutral',
            parent=self._styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor(self.colors.text),
            alignment=TA_LEFT,
            leftIndent=15,
            spaceBefore=4,
            spaceAfter=4,
            bulletIndent=5,
        )

        # Figure caption
        self._custom_styles['FigureCaption'] = ParagraphStyle(
            'FigureCaption',
            parent=self._styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor(self.colors.text_secondary),
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=15,
        )

        # Table header
        self._custom_styles['TableHeader'] = ParagraphStyle(
            'TableHeader',
            parent=self._styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor(self.colors.white),
            alignment=TA_CENTER,
        )

        # Table cell
        self._custom_styles['TableCell'] = ParagraphStyle(
            'TableCell',
            parent=self._styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor(self.colors.text),
            alignment=TA_CENTER,
        )

        # Page header
        self._custom_styles['PageHeader'] = ParagraphStyle(
            'PageHeader',
            parent=self._styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            leading=10,
            textColor=colors.HexColor(self.colors.text_secondary),
            alignment=TA_LEFT,
        )

        # Page footer
        self._custom_styles['PageFooter'] = ParagraphStyle(
            'PageFooter',
            parent=self._styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            leading=10,
            textColor=colors.HexColor(self.colors.text_secondary),
            alignment=TA_CENTER,
        )

        # Metric value (large)
        self._custom_styles['MetricValue'] = ParagraphStyle(
            'MetricValue',
            parent=self._styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=24,
            leading=28,
            textColor=colors.HexColor(self.colors.highlight),
            alignment=TA_CENTER,
        )

        # Metric label
        self._custom_styles['MetricLabel'] = ParagraphStyle(
            'MetricLabel',
            parent=self._styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=12,
            textColor=colors.HexColor(self.colors.text_secondary),
            alignment=TA_CENTER,
        )

        # Verdict styles
        self._custom_styles['VerdictRecommend'] = ParagraphStyle(
            'VerdictRecommend',
            parent=self._styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=36,
            leading=42,
            textColor=colors.HexColor(self.colors.positive),
            alignment=TA_CENTER,
        )

        self._custom_styles['VerdictMonitor'] = ParagraphStyle(
            'VerdictMonitor',
            parent=self._styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=36,
            leading=42,
            textColor=colors.HexColor(self.colors.highlight),
            alignment=TA_CENTER,
        )

        self._custom_styles['VerdictPass'] = ParagraphStyle(
            'VerdictPass',
            parent=self._styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=36,
            leading=42,
            textColor=colors.HexColor(self.colors.negative),
            alignment=TA_CENTER,
        )

    def get_style(self, style_name: str) -> ParagraphStyle:
        """Get a paragraph style by name."""
        if style_name in self._custom_styles:
            return self._custom_styles[style_name]
        return self._styles[style_name]

    def get_table_style(self) -> list:
        """Get default table style commands."""
        return [
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors.accent)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor(self.colors.white)),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),

            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor(self.colors.secondary)),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor(self.colors.text)),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),

            # Alternating rows
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor(self.colors.secondary),
              colors.HexColor(self.colors.primary)]),

            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(self.colors.neutral)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]

    def get_percentile_table_style(self) -> list:
        """Get table style with percentile coloring."""
        base_style = self.get_table_style()
        # Additional styling for percentile columns would be applied per-cell
        return base_style

    @property
    def background_color(self) -> colors.Color:
        """Get the page background color."""
        return colors.HexColor(self.colors.primary)
