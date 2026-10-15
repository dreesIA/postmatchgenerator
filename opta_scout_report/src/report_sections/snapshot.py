"""Executive summary / snapshot section."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import numpy as np

from .base import BaseSection, SectionContent, Takeaway


class SnapshotSection(BaseSection):
    """One-page executive summary with key metrics and radar chart."""

    @property
    def section_title(self) -> str:
        return "Executive Summary"

    def generate(self) -> SectionContent:
        """Generate the executive summary section."""
        content = SectionContent(title=self.section_title)

        # Add player info card
        content.raw_data['player_info'] = self._get_player_info()

        # Add key metrics table
        key_metrics = self._get_key_metrics()
        content.add_table(key_metrics, "Key Metrics")

        # Generate radar chart
        radar_fig = self._create_summary_radar()
        if radar_fig:
            content.add_figure(radar_fig, "Performance Profile vs Position Cohort")

        # Generate key takeaways
        takeaways = self._generate_key_takeaways()
        for takeaway in takeaways:
            content.takeaways.append(takeaway)

        # Add recommendation summary
        content.raw_data['recommendation'] = self._get_recommendation_summary()

        return content

    def _get_player_info(self) -> Dict[str, Any]:
        """Get player biographical information."""
        info = self.player_data.info
        return {
            'name': info.name,
            'age': info.age,
            'nationality': info.nationality,
            'team': info.current_team,
            'position': info.position,
            'position_group': info.position_group,
            'height': info.height,
            'weight': info.weight,
            'preferred_foot': info.preferred_foot,
            'contract_expiry': getattr(info, 'contract_expiry', None),
            'market_value': getattr(info, 'market_value', None),
            'total_minutes': self.metrics.total_minutes if hasattr(self.metrics, 'total_minutes') else 0,
            'matches_played': len(self.player_data.appearances) if hasattr(self.player_data, 'appearances') else 0,
        }

    def _get_key_metrics(self) -> Dict[str, Any]:
        """Get key metrics based on position."""
        position_group = self.player_data.info.position_group

        # Define key metrics by position
        key_metrics_map = {
            'GK': ['saves', 'save_percentage', 'clean_sheets', 'goals_prevented', 'pass_accuracy'],
            'CB': ['aerial_duel_win_rate', 'tackles_won', 'interceptions', 'pass_accuracy', 'progressive_passes'],
            'FB': ['tackles_won', 'progressive_carries', 'crosses', 'key_passes', 'interceptions'],
            'DM': ['tackles_won', 'interceptions', 'pass_accuracy', 'progressive_passes', 'pressures'],
            'CM': ['progressive_passes', 'pass_accuracy', 'key_passes', 'tackles_won', 'xA'],
            'AM': ['key_passes', 'xA', 'shot_creating_actions', 'progressive_passes', 'xG'],
            'W': ['xG', 'xA', 'dribble_success_rate', 'progressive_carries', 'key_passes'],
            'ST': ['goals', 'xG', 'shots_on_target', 'xA', 'aerial_duel_win_rate'],
        }

        metrics_to_show = key_metrics_map.get(position_group,
                                               ['passes_completed', 'progressive_passes', 'key_passes', 'tackles_won', 'interceptions'])

        result = {}
        for metric in metrics_to_show:
            value = self._get_metric_value(metric)
            percentile = self._get_percentile(metric)
            result[metric] = {
                'value': value,
                'percentile': percentile,
                'formatted': self._format_value(value, metric)
            }

        return result

    def _create_summary_radar(self) -> Figure:
        """Create a summary radar chart."""
        position_group = self.player_data.info.position_group

        # Get radar axes from config
        radar_config = self.config.get('radar_profiles', {}).get(position_group, {})
        axes = radar_config.get('axes', [])

        if not axes or len(axes) < 4:
            # Default axes if not configured
            axes = ['passes_completed', 'progressive_passes', 'key_passes',
                    'tackles_won', 'interceptions', 'pressures']

        # Get percentile values for each axis
        values = []
        labels = []
        for axis in axes:
            percentile = self._get_percentile(axis, default=50.0)
            values.append(percentile / 100)  # Normalize to 0-1
            # Format label
            label = axis.replace('_', ' ').title()
            labels.append(label)

        if not values:
            return None

        # Create radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['primary'])

        # Number of variables
        num_vars = len(values)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values_plot = values + [values[0]]  # Complete the loop
        angles += angles[:1]

        # Draw the radar chart
        ax.plot(angles, values_plot, 'o-', linewidth=2, color=self.colors['highlight'])
        ax.fill(angles, values_plot, alpha=0.25, color=self.colors['highlight'])

        # Draw reference circles
        for percentile in [0.25, 0.50, 0.75]:
            circle = plt.Circle((0, 0), percentile, transform=ax.transData._b,
                                fill=False, color=self.colors['neutral'],
                                linewidth=0.5, linestyle='--', alpha=0.5)

        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=9, color=self.colors['text'])

        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25th', '50th', '75th', '100th'],
                          size=8, color=self.colors['text_secondary'])

        # Grid styling
        ax.spines['polar'].set_color(self.colors['neutral'])
        ax.grid(color=self.colors['neutral'], alpha=0.3)

        # Title
        ax.set_title(f"{self.player_data.info.name} - Performance Profile",
                     color=self.colors['text'], fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    def _generate_key_takeaways(self) -> List[Takeaway]:
        """Generate the most important takeaways for the executive summary."""
        takeaways = []
        position_group = self.player_data.info.position_group

        # Get all metrics and their percentiles
        metric_percentiles = []
        if hasattr(self.cohort, 'percentiles'):
            for metric, result in self.cohort.percentiles.items():
                metric_percentiles.append((metric, result.percentile, result.value))

        # Sort by how extreme the percentile is
        metric_percentiles.sort(key=lambda x: abs(x[1] - 50), reverse=True)

        # Get top strengths (percentile >= 75)
        strengths = [(m, p, v) for m, p, v in metric_percentiles if p >= 75][:3]
        for metric, percentile, value in strengths:
            display_name = metric.replace('_', ' ').title()
            formatted = self._format_value(value, metric)
            takeaways.append(Takeaway(
                text=f"Strength: {display_name} - {formatted} per 90 ({percentile:.0f}th percentile)",
                metric=metric,
                value=value,
                percentile=percentile,
                is_positive=True
            ))

        # Get key weaknesses (percentile <= 25)
        weaknesses = [(m, p, v) for m, p, v in metric_percentiles if p <= 25][:2]
        for metric, percentile, value in weaknesses:
            display_name = metric.replace('_', ' ').title()
            formatted = self._format_value(value, metric)
            takeaways.append(Takeaway(
                text=f"Area to develop: {display_name} - {formatted} per 90 ({percentile:.0f}th percentile)",
                metric=metric,
                value=value,
                percentile=percentile,
                is_positive=False
            ))

        # Add minutes context
        total_minutes = self.metrics.total_minutes if hasattr(self.metrics, 'total_minutes') else 0
        if total_minutes > 0:
            matches = len(self.player_data.appearances) if hasattr(self.player_data, 'appearances') else 0
            avg_mins = total_minutes / matches if matches > 0 else 0
            takeaways.append(Takeaway(
                text=f"Sample: {total_minutes:.0f} minutes across {matches} appearances ({avg_mins:.0f} mins/game avg)",
                is_positive=True
            ))

        return takeaways

    def _get_recommendation_summary(self) -> Dict[str, Any]:
        """Get the recommendation summary."""
        if hasattr(self.cohort, 'recommendation'):
            rec = self.cohort.recommendation
            return {
                'verdict': rec.get('verdict', 'Monitor'),
                'fit_score': rec.get('fit_score', 0),
                'risk_score': rec.get('risk_score', 0),
                'reason': rec.get('reason', ''),
            }

        # Calculate basic recommendation if not present
        if hasattr(self.cohort, 'overall_percentile'):
            percentile = self.cohort.overall_percentile
            if percentile >= 70:
                verdict = "Recommend"
            elif percentile >= 50:
                verdict = "Monitor"
            else:
                verdict = "Pass"

            return {
                'verdict': verdict,
                'fit_score': percentile,
                'risk_score': 0,
                'reason': f"Overall performance at {percentile:.0f}th percentile for position"
            }

        return {'verdict': 'Monitor', 'fit_score': 50, 'risk_score': 0, 'reason': 'Insufficient data'}
