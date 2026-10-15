"""Physical performance section."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .base import BaseSection, SectionContent, Takeaway


class PhysicalSection(BaseSection):
    """Analysis of physical performance metrics (if available)."""

    @property
    def section_title(self) -> str:
        return "Physical Performance"

    def generate(self) -> SectionContent:
        """Generate the physical performance section."""
        content = SectionContent(title=self.section_title)

        # Check if physical data is available
        has_physical = self._has_physical_data()

        if not has_physical:
            content.takeaways.append(Takeaway(
                text="Physical tracking data not available for this player/competition",
                is_positive=True  # Neutral observation
            ))
            return content

        # Create physical metrics overview
        overview_fig = self._create_physical_overview()
        if overview_fig:
            content.add_figure(overview_fig, "Physical Metrics Overview")

        # Create distance/sprint chart
        distance_fig = self._create_distance_chart()
        if distance_fig:
            content.add_figure(distance_fig, "Distance & Sprint Metrics by Match")

        # Create intensity comparison
        intensity_fig = self._create_intensity_comparison()
        if intensity_fig:
            content.add_figure(intensity_fig, "Intensity Metrics vs Position Average")

        # Generate takeaways
        takeaways = self._generate_takeaways()
        for takeaway in takeaways:
            content.takeaways.append(takeaway)

        # Add raw physical stats
        content.raw_data['physical_stats'] = self._get_physical_stats()

        return content

    def _has_physical_data(self) -> bool:
        """Check if physical tracking data is available."""
        physical_metrics = ['distance_covered', 'sprints', 'high_speed_runs',
                           'sprint_distance', 'top_speed']

        for metric in physical_metrics:
            value = self._get_metric_value(metric, default=None)
            if value is not None and value > 0:
                return True

        return False

    def _create_physical_overview(self) -> Figure:
        """Create physical metrics overview chart."""
        metrics = {
            'Distance (km)': self._get_metric_value('distance_covered'),
            'Sprints': self._get_metric_value('sprints'),
            'High-Speed Runs': self._get_metric_value('high_speed_runs'),
            'Sprint Distance (m)': self._get_metric_value('sprint_distance'),
        }

        # Filter out zero values
        metrics = {k: v for k, v in metrics.items() if v > 0}

        if not metrics:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.patch.set_facecolor(self.colors['primary'])

        for ax, (metric, value) in zip(axes.flat, metrics.items()):
            ax.set_facecolor(self.colors['secondary'])

            # Get percentile for this metric
            metric_key = metric.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '')
            percentile = self._get_percentile(metric_key, default=50)

            # Create gauge-like visualization
            self._draw_gauge(ax, value, percentile, metric)

        # Hide unused axes
        for i in range(len(metrics), 4):
            axes.flat[i].set_visible(False)

        plt.suptitle('Physical Metrics Overview (Per 90)',
                    color=self.colors['text'], fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def _draw_gauge(self, ax, value: float, percentile: float, label: str) -> None:
        """Draw a gauge visualization for a metric."""
        # Draw arc background
        theta = np.linspace(0, np.pi, 100)
        r = 1

        # Background arc
        ax.fill_between(theta, 0, r, alpha=0.2, color=self.colors['neutral'])

        # Value arc
        value_theta = np.linspace(0, np.pi * percentile / 100, 50)
        if percentile >= 75:
            color = self.colors['positive']
        elif percentile >= 50:
            color = self.colors['highlight']
        elif percentile >= 25:
            color = self.colors['neutral']
        else:
            color = self.colors['negative']

        ax.fill_between(value_theta, 0, r, alpha=0.6, color=color)

        # Value text
        ax.text(np.pi/2, 0.3, f'{value:.1f}', ha='center', va='center',
               fontsize=18, fontweight='bold', color=self.colors['text'])
        ax.text(np.pi/2, 0.05, f'{percentile:.0f}th %ile', ha='center', va='center',
               fontsize=10, color=self.colors['text_secondary'])

        # Label
        ax.set_title(label, color=self.colors['text'], fontsize=11, fontweight='bold')

        ax.set_xlim(0, np.pi)
        ax.set_ylim(-0.2, 1.1)
        ax.axis('off')

    def _create_distance_chart(self) -> Figure:
        """Create distance/sprint metrics by match."""
        # Try to get match-by-match physical data
        match_metrics = getattr(self.metrics, 'match_by_match', None)

        if match_metrics is None or not match_metrics:
            return None

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        x = range(len(match_metrics))

        # Try to extract physical data
        distances = []
        sprints = []

        for match in match_metrics:
            if isinstance(match, dict):
                distances.append(match.get('distance_covered', 0))
                sprints.append(match.get('sprints', 0))
            else:
                distances.append(getattr(match, 'distance_covered', 0))
                sprints.append(getattr(match, 'sprints', 0))

        if sum(distances) == 0 and sum(sprints) == 0:
            plt.close(fig)
            return None

        # Plot distance
        ax2 = ax.twinx()

        if sum(distances) > 0:
            ax.bar(x, distances, color=self.colors['highlight'], alpha=0.7,
                  label='Distance (km)', width=0.4, align='edge')

        if sum(sprints) > 0:
            ax2.plot(x, sprints, color=self.colors['positive'], marker='o',
                    linewidth=2, label='Sprints')

        ax.set_xlabel('Match', color=self.colors['text'])
        ax.set_ylabel('Distance (km)', color=self.colors['highlight'])
        ax2.set_ylabel('Sprints', color=self.colors['positive'])

        ax.tick_params(colors=self.colors['text'])
        ax2.tick_params(colors=self.colors['text'])

        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['highlight'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_color(self.colors['positive'])

        ax.set_title('Physical Output by Match', color=self.colors['text'],
                    fontsize=12, fontweight='bold')

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                 facecolor=self.colors['secondary'], edgecolor=self.colors['neutral'],
                 labelcolor=self.colors['text'])

        plt.tight_layout()
        return fig

    def _create_intensity_comparison(self) -> Figure:
        """Create intensity comparison vs position average."""
        physical_metrics = {
            'Sprints': ('sprints', self._get_metric_value('sprints')),
            'High-Speed Runs': ('high_speed_runs', self._get_metric_value('high_speed_runs')),
            'Top Speed': ('top_speed', self._get_metric_value('top_speed')),
            'Accelerations': ('accelerations', self._get_metric_value('accelerations')),
        }

        # Filter out zeros
        physical_metrics = {k: v for k, v in physical_metrics.items() if v[1] > 0}

        if not physical_metrics:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        labels = list(physical_metrics.keys())
        values = [v[1] for v in physical_metrics.values()]
        percentiles = [self._get_percentile(v[0]) for v in physical_metrics.values()]

        x = np.arange(len(labels))
        width = 0.6

        # Color by percentile
        colors = []
        for pct in percentiles:
            if pct >= 75:
                colors.append(self.colors['positive'])
            elif pct >= 50:
                colors.append(self.colors['highlight'])
            elif pct >= 25:
                colors.append(self.colors['neutral'])
            else:
                colors.append(self.colors['negative'])

        bars = ax.bar(x, values, width, color=colors, alpha=0.8, edgecolor='white')

        # Add percentile annotations
        for bar, pct in zip(bars, percentiles):
            height = bar.get_height()
            ax.annotate(f'{pct:.0f}th',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom',
                       color=self.colors['text'], fontsize=10, fontweight='bold')

        ax.set_ylabel('Value per 90', color=self.colors['text'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=self.colors['text'])

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title('Intensity Metrics with Percentile Rankings',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _generate_takeaways(self) -> List[Takeaway]:
        """Generate physical performance takeaways."""
        takeaways = []

        # Distance analysis
        distance = self._get_metric_value('distance_covered')
        distance_pct = self._get_percentile('distance_covered')
        if distance > 0:
            descriptor = self._percentile_descriptor(distance_pct)
            takeaways.append(Takeaway(
                text=f"Distance coverage: {distance:.1f}km per 90 ({descriptor}, {distance_pct:.0f}th percentile)",
                metric='distance_covered',
                value=distance,
                percentile=distance_pct,
                is_positive=distance_pct >= 50
            ))

        # Sprint analysis
        sprints = self._get_metric_value('sprints')
        sprints_pct = self._get_percentile('sprints')
        if sprints > 0:
            descriptor = self._percentile_descriptor(sprints_pct)
            takeaways.append(Takeaway(
                text=f"Sprint frequency: {sprints:.1f} sprints per 90 ({descriptor}, {sprints_pct:.0f}th percentile)",
                metric='sprints',
                value=sprints,
                percentile=sprints_pct,
                is_positive=sprints_pct >= 50
            ))

        # Top speed
        top_speed = self._get_metric_value('top_speed')
        if top_speed > 0:
            takeaways.append(Takeaway(
                text=f"Top speed: {top_speed:.1f} m/s",
                metric='top_speed',
                value=top_speed,
                is_positive=True
            ))

        return takeaways

    def _get_physical_stats(self) -> Dict[str, Any]:
        """Get all physical statistics."""
        return {
            'distance_covered': self._get_metric_value('distance_covered'),
            'sprints': self._get_metric_value('sprints'),
            'high_speed_runs': self._get_metric_value('high_speed_runs'),
            'sprint_distance': self._get_metric_value('sprint_distance'),
            'accelerations': self._get_metric_value('accelerations'),
            'decelerations': self._get_metric_value('decelerations'),
            'top_speed': self._get_metric_value('top_speed'),
        }
