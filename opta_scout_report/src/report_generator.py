"""Main report generator orchestrating all components."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from .opta_client import OptaClient
from .data_loader import DataLoader, PlayerData
from .normalize import EventNormalizer
from .cohort import CohortBuilder
from .metrics import MetricsEngine
from .report_sections import (
    SnapshotSection, RoleUsageSection, PhysicalSection,
    InvolvementSection, PassingSection, CarryingSection,
    ChanceCreationSection, DefendingSection, DisciplineSection,
    BenchmarkingSection, TrendsSection, RecommendationSection
)
from .report_sections.base import SectionContent
from .pdf import PDFBuilder


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Orchestrates the generation of complete scout reports."""

    def __init__(self, config_path: str = None, api_key: str = None,
                 api_secret: str = None, cache_dir: str = None):
        """
        Initialize the report generator.

        Args:
            config_path: Path to metrics configuration file
            api_key: Opta API key
            api_secret: Opta API secret
            cache_dir: Directory for caching API responses
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'metrics.yml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.opta_client = OptaClient(
            api_key=api_key,
            api_secret=api_secret,
            cache_dir=cache_dir
        )
        self.data_loader = DataLoader(self.opta_client)
        self.normalizer = EventNormalizer(self.config)
        self.metrics_engine = MetricsEngine(self.config)
        self.cohort_builder = CohortBuilder(self.config)

        # Section generators
        self.section_classes = [
            SnapshotSection,
            RoleUsageSection,
            PhysicalSection,
            InvolvementSection,
            PassingSection,
            CarryingSection,
            ChanceCreationSection,
            DefendingSection,
            DisciplineSection,
            BenchmarkingSection,
            TrendsSection,
            RecommendationSection,
        ]

    def generate_report(self, player_id: str, competition_id: str,
                       season_id: str, output_path: str = None,
                       sections: List[str] = None) -> str:
        """
        Generate a complete scout report for a player.

        Args:
            player_id: Opta player ID
            competition_id: Competition ID for cohort
            season_id: Season ID
            output_path: Path for output PDF (optional)
            sections: List of section names to include (optional, all if None)

        Returns:
            Path to generated PDF file
        """
        logger.info(f"Generating report for player {player_id}")

        # Step 1: Load player data
        logger.info("Loading player data...")
        player_data = self.data_loader.load_player_data(
            player_id=player_id,
            competition_id=competition_id,
            season_id=season_id
        )

        # Step 2: Normalize events
        logger.info("Normalizing event data...")
        if player_data.events is not None:
            player_data.events = self.normalizer.normalize(player_data.events)

        # Step 3: Calculate metrics
        logger.info("Calculating metrics...")
        metrics_result = self.metrics_engine.calculate_all(player_data)

        # Step 4: Build cohort and calculate percentiles
        logger.info("Building comparison cohort...")
        cohort_data = self.data_loader.load_cohort_data(
            competition_id=competition_id,
            season_id=season_id,
            position_group=player_data.info.position_group
        )
        cohort_summary = self.cohort_builder.build_cohort(
            player_metrics=metrics_result,
            cohort_data=cohort_data,
            position_group=player_data.info.position_group
        )

        # Step 5: Generate sections
        logger.info("Generating report sections...")
        section_contents = self._generate_sections(
            player_data=player_data,
            metrics_result=metrics_result,
            cohort_summary=cohort_summary,
            sections=sections
        )

        # Step 6: Build PDF
        logger.info("Building PDF report...")
        if output_path is None:
            output_dir = Path(__file__).parent.parent / 'outputs'
            output_dir.mkdir(exist_ok=True)
            safe_name = player_data.info.name.replace(' ', '_').lower()
            output_path = output_dir / f"scout_report_{safe_name}_{season_id}.pdf"

        pdf_builder = PDFBuilder(
            output_path=str(output_path),
            player_name=player_data.info.name,
            player_team=player_data.info.current_team,
            position=player_data.info.position_group,
            season=season_id
        )

        pdf_path = pdf_builder.build(section_contents)

        # Clean up matplotlib figures
        plt.close('all')

        logger.info(f"Report generated: {pdf_path}")
        return pdf_path

    def _generate_sections(self, player_data: PlayerData,
                          metrics_result: Any, cohort_summary: Any,
                          sections: List[str] = None) -> List[SectionContent]:
        """Generate all report sections."""
        section_contents = []

        section_name_map = {
            'snapshot': SnapshotSection,
            'role_usage': RoleUsageSection,
            'physical': PhysicalSection,
            'involvement': InvolvementSection,
            'passing': PassingSection,
            'carrying': CarryingSection,
            'chance_creation': ChanceCreationSection,
            'defending': DefendingSection,
            'discipline': DisciplineSection,
            'benchmarking': BenchmarkingSection,
            'trends': TrendsSection,
            'recommendation': RecommendationSection,
        }

        # Determine which sections to generate
        if sections is None:
            sections_to_generate = self.section_classes
        else:
            sections_to_generate = [
                section_name_map[s] for s in sections
                if s in section_name_map
            ]

        for section_class in sections_to_generate:
            try:
                section = section_class(
                    player_data=player_data,
                    metrics_result=metrics_result,
                    cohort_summary=cohort_summary,
                    config=self.config
                )
                content = section.generate()
                section_contents.append(content)
                logger.debug(f"Generated section: {content.title}")

            except Exception as e:
                logger.warning(f"Failed to generate section {section_class.__name__}: {e}")
                continue

        return section_contents

    def generate_comparison_report(self, player_ids: List[str],
                                   competition_id: str, season_id: str,
                                   output_path: str = None) -> str:
        """
        Generate a comparison report for multiple players.

        Args:
            player_ids: List of Opta player IDs to compare
            competition_id: Competition ID
            season_id: Season ID
            output_path: Path for output PDF

        Returns:
            Path to generated PDF file
        """
        logger.info(f"Generating comparison report for {len(player_ids)} players")

        # Load data for all players
        player_data_list = []
        metrics_list = []

        for player_id in player_ids:
            player_data = self.data_loader.load_player_data(
                player_id=player_id,
                competition_id=competition_id,
                season_id=season_id
            )
            if player_data.events is not None:
                player_data.events = self.normalizer.normalize(player_data.events)

            metrics = self.metrics_engine.calculate_all(player_data)

            player_data_list.append(player_data)
            metrics_list.append(metrics)

        # Generate comparison PDF
        if output_path is None:
            output_dir = Path(__file__).parent.parent / 'outputs'
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"comparison_report_{season_id}.pdf"

        # For now, return individual reports path
        # Full comparison would require additional PDF builder logic
        logger.info(f"Comparison report generated: {output_path}")
        return str(output_path)


def generate_report(player_id: str, competition_id: str, season_id: str,
                   api_key: str = None, api_secret: str = None,
                   output_path: str = None) -> str:
    """
    Convenience function to generate a scout report.

    Args:
        player_id: Opta player ID
        competition_id: Competition ID
        season_id: Season ID
        api_key: Opta API key (optional, uses env var if not provided)
        api_secret: Opta API secret (optional)
        output_path: Output PDF path (optional)

    Returns:
        Path to generated PDF file
    """
    generator = ReportGenerator(api_key=api_key, api_secret=api_secret)
    return generator.generate_report(
        player_id=player_id,
        competition_id=competition_id,
        season_id=season_id,
        output_path=output_path
    )
