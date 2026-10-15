"""Benchmarking against position cohort section."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .base import BaseSection, SectionContent, Takeaway


class BenchmarkingSection(BaseSection):
    """Percentile rankings and comparison against position cohort."""

    @property
    def section_title(self) -> str:
        return "Benchmarking"

    def generate(self) -> SectionContent:
        """Generate the benchmarking section."""
        content = SectionContent(title=self.section_title)

        # Create full percentile bar chart
        percentile_fig = self._create_percentile_chart()
        if percentile_fig:
            content.add_figure(percentile_fig, "Full Percentile Rankings")

        # Create position radar comparison
        radar_fig = self._create_position_radar()
        if radar_fig:
            content.add_figure(radar_fig, "Position-Specific Profile")

        # Create scatter comparison
        scatter_fig = self._create_benchmark_scatter()
        if scatter_fig:
            content.add_figure(scatter_fig, "Key Metrics Scatter")

        # Create percentile distribution
        dist_fig = self._create_percentile_distribution()
        if dist_fig:
            content.add_figure(dist_fig, "Percentile Distribution")

        # Generate takeaways
        takeaways = self._generate_takeaways()
        for takeaway in takeaways:
            content.takeaways.append(takeaway)

        # Add cohort info
        content.raw_data['cohort_info'] = self._get_cohort_info()

        return content

    def _create_percentile_chart(self) -> Figure:
        """Create comprehensive percentile bar chart."""
        if not hasattr(self.cohort, 'percentiles') or not self.cohort.percentiles:
            return None

        # Get all percentiles
        all_percentiles = {}
        for metric, result in self.cohort.percentiles.items():
            all_percentiles[metric] = result.percentile

        # Sort by percentile
        sorted_metrics = sorted(all_percentiles.items(), key=lambda x: x[1], reverse=True)

        # Take top/bottom metrics for display
        if len(sorted_metrics) > 20:
            top_10 = sorted_metrics[:10]
            bottom_10 = sorted_metrics[-10:]
            display_metrics = top_10 + bottom_10
        else:
            display_metrics = sorted_metrics

        fig, ax = plt.subplots(figsize=(12, max(8, len(display_metrics) * 0.4)))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        labels = [m[0].replace('_', ' ').title() for m in display_metrics]
        values = [m[1] for m in display_metrics]

        y_pos = np.arange(len(labels))

        # Color by percentile
        colors = []
        for pct in values:
            if pct >= 75:
                colors.append(self.colors['positive'])
            elif pct >= 50:
                colors.append(self.colors['highlight'])
            elif pct >= 25:
                colors.append(self.colors['neutral'])
            else:
                colors.append(self.colors['negative'])

        bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='white', height=0.7)

        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{val:.0f}', va='center', color=self.colors['text'], fontsize=9)

        # Reference lines
        ax.axvline(x=25, color=self.colors['negative'], linestyle=':', alpha=0.5)
        ax.axvline(x=50, color=self.colors['text'], linestyle='--', alpha=0.5)
        ax.axvline(x=75, color=self.colors['positive'], linestyle=':', alpha=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color=self.colors['text'], fontsize=9)
        ax.set_xlabel('Percentile', color=self.colors['text'])
        ax.set_xlim(0, 105)

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.invert_yaxis()

        position = self.player_data.info.position_group
        ax.set_title(f'Percentile Rankings vs {position} Cohort',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_position_radar(self) -> Figure:
        """Create position-specific radar chart."""
        position_group = self.player_data.info.position_group
        radar_config = self.config.get('radar_profiles', {}).get(position_group, {})
        axes = radar_config.get('axes', [])

        if not axes or len(axes) < 4:
            return None

        # Get percentile values
        values = []
        labels = []
        for axis in axes:
            pct = self._get_percentile(axis, default=50.0)
            values.append(pct / 100)
            labels.append(axis.replace('_', ' ').title())

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['primary'])

        num_vars = len(values)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles += angles[:1]

        # Player profile
        ax.plot(angles, values_plot, 'o-', linewidth=2, color=self.colors['highlight'],
               label=self.player_data.info.name)
        ax.fill(angles, values_plot, alpha=0.25, color=self.colors['highlight'])

        # Add median reference (50th percentile)
        median_values = [0.5] * (num_vars + 1)
        ax.plot(angles, median_values, '--', linewidth=1, color=self.colors['neutral'],
               alpha=0.7, label='Median')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=9, color=self.colors['text'])

        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25th', '50th', '75th', '100th'],
                          size=8, color=self.colors['text_secondary'])

        ax.spines['polar'].set_color(self.colors['neutral'])
        ax.grid(color=self.colors['neutral'], alpha=0.3)

        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1),
                 facecolor=self.colors['secondary'],
                 edgecolor=self.colors['neutral'], labelcolor=self.colors['text'])

        ax.set_title(f'{position_group} Profile - Key Metrics',
                    color=self.colors['text'], fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    def _create_benchmark_scatter(self) -> Figure:
        """Create scatter plot of key metrics with cohort context."""
        position_group = self.player_data.info.position_group

        # Select key metrics based on position
        if position_group in ['ST', 'W', 'AM']:
            x_metric = 'xG'
            y_metric = 'xA'
            x_label = 'xG per 90'
            y_label = 'xA per 90'
        elif position_group in ['CM', 'DM']:
            x_metric = 'progressive_passes'
            y_metric = 'tackles_won'
            x_label = 'Progressive Passes per 90'
            y_label = 'Tackles Won per 90'
        else:  # CB, FB
            x_metric = 'interceptions'
            y_metric = 'pass_accuracy'
            x_label = 'Interceptions per 90'
            y_label = 'Pass Accuracy %'

        player_x = self._get_metric_value(x_metric)
        player_y = self._get_metric_value(y_metric)

        # Get cohort values if available
        cohort_x = []
        cohort_y = []
        if hasattr(self.cohort, 'cohort_data') and self.cohort.cohort_data is not None:
            df = self.cohort.cohort_data
            if x_metric in df.columns and y_metric in df.columns:
                cohort_x = df[x_metric].dropna().values
                cohort_y = df[y_metric].dropna().values

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        # Plot cohort (if available)
        if len(cohort_x) > 0 and len(cohort_y) > 0:
            ax.scatter(cohort_x, cohort_y, s=50, c=self.colors['neutral'],
                      alpha=0.4, label='Cohort')

        # Plot player
        ax.scatter(player_x, player_y, s=200, c=self.colors['highlight'],
                  marker='*', edgecolor='white', linewidth=2,
                  label=self.player_data.info.name, zorder=10)

        # Add reference lines at median
        if len(cohort_x) > 0:
            ax.axvline(x=np.median(cohort_x), color=self.colors['neutral'],
                      linestyle='--', alpha=0.5)
        if len(cohort_y) > 0:
            ax.axhline(y=np.median(cohort_y), color=self.colors['neutral'],
                      linestyle='--', alpha=0.5)

        ax.set_xlabel(x_label, color=self.colors['text'])
        ax.set_ylabel(y_label, color=self.colors['text'])

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='upper right', facecolor=self.colors['secondary'],
                 edgecolor=self.colors['neutral'], labelcolor=self.colors['text'])

        ax.set_title(f'{x_label} vs {y_label}',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_percentile_distribution(self) -> Figure:
        """Create histogram of player's percentile distribution."""
        if not hasattr(self.cohort, 'percentiles') or not self.cohort.percentiles:
            return None

        percentiles = [r.percentile for r in self.cohort.percentiles.values()]

        if not percentiles:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        # Create histogram
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        counts, edges, patches = ax.hist(percentiles, bins=bins, color=self.colors['highlight'],
                                         alpha=0.8, edgecolor='white')

        # Color bars by range
        for i, patch in enumerate(patches):
            if edges[i] < 25:
                patch.set_facecolor(self.colors['negative'])
            elif edges[i] < 50:
                patch.set_facecolor(self.colors['neutral'])
            elif edges[i] < 75:
                patch.set_facecolor(self.colors['highlight'])
            else:
                patch.set_facecolor(self.colors['positive'])

        # Add average line
        avg_pct = np.mean(percentiles)
        ax.axvline(x=avg_pct, color=self.colors['text'], linestyle='--',
                  linewidth=2, label=f'Average: {avg_pct:.0f}th')

        ax.set_xlabel('Percentile', color=self.colors['text'])
        ax.set_ylabel('Number of Metrics', color=self.colors['text'])
        ax.set_xlim(0, 100)

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='upper right', facecolor=self.colors['secondary'],
                 edgecolor=self.colors['neutral'], labelcolor=self.colors['text'])

        ax.set_title('Distribution of Percentile Rankings',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _generate_takeaways(self) -> List[Takeaway]:
        """Generate benchmarking takeaways."""
        takeaways = []

        if not hasattr(self.cohort, 'percentiles') or not self.cohort.percentiles:
            return takeaways

        percentiles = [r.percentile for r in self.cohort.percentiles.values()]

        if not percentiles:
            return takeaways

        # Overall assessment
        avg_pct = np.mean(percentiles)
        takeaways.append(Takeaway(
            text=f"Overall profile: Average {avg_pct:.0f}th percentile across all metrics",
            is_positive=avg_pct >= 50
        ))

        # Count elite/poor metrics
        elite_count = sum(1 for p in percentiles if p >= 75)
        poor_count = sum(1 for p in percentiles if p <= 25)

        if elite_count > 0:
            takeaways.append(Takeaway(
                text=f"Elite in {elite_count} metrics (75th+ percentile)",
                is_positive=True
            ))

        if poor_count > 0:
            takeaways.append(Takeaway(
                text=f"Below average in {poor_count} metrics (25th- percentile)",
                is_positive=False
            ))

        # Consistency
        std_pct = np.std(percentiles)
        if std_pct < 15:
            takeaways.append(Takeaway(
                text="Consistent profile: Performance is similar across metrics",
                is_positive=True
            ))
        elif std_pct > 25:
            takeaways.append(Takeaway(
                text="Varied profile: Significant strengths and weaknesses",
                is_positive=True
            ))

        # Top strengths
        sorted_metrics = sorted(self.cohort.percentiles.items(),
                               key=lambda x: x[1].percentile, reverse=True)

        if sorted_metrics:
            top = sorted_metrics[0]
            takeaways.append(Takeaway(
                text=f"Biggest strength: {top[0].replace('_', ' ').title()} ({top[1].percentile:.0f}th percentile)",
                metric=top[0],
                percentile=top[1].percentile,
                is_positive=True
            ))

        return takeaways

    def _get_cohort_info(self) -> Dict[str, Any]:
        """Get information about the comparison cohort."""
        return {
            'position_group': self.player_data.info.position_group,
            'cohort_size': getattr(self.cohort, 'cohort_size', 'Unknown'),
            'min_minutes': self.config.get('position_groups', {}).get(
                self.player_data.info.position_group, {}).get('min_minutes', 450),
            'competition': getattr(self.cohort, 'competition', 'League'),
            'season': getattr(self.cohort, 'season', 'Current'),
        }
