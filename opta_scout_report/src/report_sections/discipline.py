"""Discipline and risk analysis section."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .base import BaseSection, SectionContent, Takeaway


class DisciplineSection(BaseSection):
    """Analysis of fouls, cards, turnovers, and errors."""

    @property
    def section_title(self) -> str:
        return "Discipline & Risk"

    def generate(self) -> SectionContent:
        """Generate the discipline section."""
        content = SectionContent(title=self.section_title)

        # Create discipline overview
        discipline_fig = self._create_discipline_overview()
        if discipline_fig:
            content.add_figure(discipline_fig, "Discipline Metrics")

        # Create foul locations map
        foul_map_fig = self._create_foul_map()
        if foul_map_fig:
            content.add_figure(foul_map_fig, "Foul Locations")

        # Create turnover analysis
        turnover_fig = self._create_turnover_analysis()
        if turnover_fig:
            content.add_figure(turnover_fig, "Turnover Analysis")

        # Create risk profile
        risk_fig = self._create_risk_profile()
        if risk_fig:
            content.add_figure(risk_fig, "Risk Profile")

        # Generate takeaways
        takeaways = self._generate_takeaways()
        for takeaway in takeaways:
            content.takeaways.append(takeaway)

        # Add discipline table
        table = self._get_discipline_table()
        content.add_table(table, "Discipline & Risk Metrics")

        return content

    def _create_discipline_overview(self) -> Figure:
        """Create discipline metrics overview."""
        metrics = {
            'Fouls Committed': ('fouls_committed', False),  # Lower is better
            'Fouls Won': ('fouls_won', True),  # Higher is better
            'Yellow Cards': ('yellow_cards', False),
            'Red Cards': ('red_cards', False),
        }

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.patch.set_facecolor(self.colors['primary'])

        for ax, (display_name, (metric_key, higher_better)) in zip(axes.flat, metrics.items()):
            ax.set_facecolor(self.colors['secondary'])

            value = self._get_metric_value(metric_key)
            pct = self._get_percentile(metric_key)

            # For metrics where lower is better, high percentile is bad
            if higher_better:
                is_good = pct >= 50
            else:
                is_good = pct <= 50

            color = self.colors['positive'] if is_good else self.colors['negative']

            # Create simple bar visualization
            ax.barh([0], [100], color=self.colors['neutral'], alpha=0.2, height=0.5)
            ax.barh([0], [pct], color=color, alpha=0.8, height=0.5)

            # Reference line
            ax.axvline(x=50, color=self.colors['text'], linestyle='--', alpha=0.5)

            # Labels
            ax.text(50, -0.4, f'{value:.2f} p90', ha='center', va='top',
                   color=self.colors['text'], fontsize=12, fontweight='bold')
            ax.text(50, 0.4, f'{pct:.0f}th percentile', ha='center', va='bottom',
                   color=self.colors['text_secondary'], fontsize=9)

            ax.set_xlim(0, 100)
            ax.set_ylim(-0.6, 0.6)
            ax.axis('off')
            ax.set_title(display_name, color=self.colors['text'], fontsize=11, fontweight='bold')

        plt.suptitle('Discipline Overview', color=self.colors['text'],
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def _create_foul_map(self) -> Figure:
        """Create foul locations map."""
        events_df = self.player_data.events if hasattr(self.player_data, 'events') else None

        if events_df is None or events_df.empty:
            return None

        # Filter to fouls committed and fouls won
        fouls_committed = events_df[
            (events_df.get('event_type', '') == 'foul') |
            (events_df.get('type_id', 0) == 4)
        ].copy() if 'event_type' in events_df.columns or 'type_id' in events_df.columns else None

        fouls_won = events_df[
            events_df.get('is_foul_won', False) == True
        ].copy() if 'is_foul_won' in events_df.columns else None

        if (fouls_committed is None or fouls_committed.empty) and (fouls_won is None or fouls_won.empty):
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor('#0a4d0a')

        self._draw_pitch(ax)

        # Plot fouls committed
        if fouls_committed is not None and not fouls_committed.empty:
            x_comm = fouls_committed['x'].dropna().values
            y_comm = fouls_committed['y'].dropna().values
            if len(x_comm) > 0:
                ax.scatter(x_comm, y_comm, s=80, c=self.colors['negative'],
                          marker='x', alpha=0.7, linewidth=2, label='Fouls Committed')

        # Plot fouls won
        if fouls_won is not None and not fouls_won.empty:
            x_won = fouls_won['x'].dropna().values
            y_won = fouls_won['y'].dropna().values
            if len(x_won) > 0:
                ax.scatter(x_won, y_won, s=80, c=self.colors['positive'],
                          marker='o', alpha=0.7, edgecolor='white', label='Fouls Won')

        ax.legend(loc='upper left', facecolor=self.colors['secondary'],
                 edgecolor=self.colors['neutral'], labelcolor=self.colors['text'])

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.axis('off')

        n_committed = len(fouls_committed) if fouls_committed is not None else 0
        n_won = len(fouls_won) if fouls_won is not None else 0
        ax.set_title(f"Foul Locations (Committed: {n_committed}, Won: {n_won})",
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _draw_pitch(self, ax) -> None:
        """Draw pitch markings."""
        ax.plot([0, 100], [0, 0], 'white', linewidth=1)
        ax.plot([0, 100], [100, 100], 'white', linewidth=1)
        ax.plot([0, 0], [0, 100], 'white', linewidth=1)
        ax.plot([100, 100], [0, 100], 'white', linewidth=1)
        ax.plot([50, 50], [0, 100], 'white', linewidth=1)

        circle = plt.Circle((50, 50), 9.15, fill=False, color='white', linewidth=1)
        ax.add_patch(circle)

        ax.plot([0, 16.5], [21.1, 21.1], 'white', linewidth=1)
        ax.plot([0, 16.5], [78.9, 78.9], 'white', linewidth=1)
        ax.plot([16.5, 16.5], [21.1, 78.9], 'white', linewidth=1)

        ax.plot([83.5, 100], [21.1, 21.1], 'white', linewidth=1)
        ax.plot([83.5, 100], [78.9, 78.9], 'white', linewidth=1)
        ax.plot([83.5, 83.5], [21.1, 78.9], 'white', linewidth=1)

    def _create_turnover_analysis(self) -> Figure:
        """Create turnover analysis chart."""
        metrics = {
            'Dispossessions': 'dispossessions',
            'Miscontrols': 'miscontrols',
            'Turnovers': 'turnovers',
            'Errors → Shot': 'errors_leading_to_shot',
            'Errors → Goal': 'errors_leading_to_goal',
        }

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        labels = []
        values = []
        percentiles = []

        for display_name, metric_key in metrics.items():
            value = self._get_metric_value(metric_key)
            pct = self._get_percentile(metric_key)
            labels.append(display_name)
            values.append(value)
            percentiles.append(pct)

        x = np.arange(len(labels))
        width = 0.6

        # For turnovers, lower percentile is better
        colors = []
        for pct in percentiles:
            if pct <= 25:
                colors.append(self.colors['positive'])  # Low turnovers = good
            elif pct <= 50:
                colors.append(self.colors['highlight'])
            elif pct <= 75:
                colors.append(self.colors['neutral'])
            else:
                colors.append(self.colors['negative'])  # High turnovers = bad

        bars = ax.bar(x, values, width, color=colors, alpha=0.8, edgecolor='white')

        for bar, val, pct in zip(bars, values, percentiles):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}\n({pct:.0f}th)',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom',
                       color=self.colors['text'], fontsize=9)

        ax.set_ylabel('Per 90', color=self.colors['text'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=self.colors['text'], rotation=15, ha='right')

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title('Turnover Metrics (Lower = Better)',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_risk_profile(self) -> Figure:
        """Create overall risk profile radar."""
        risk_metrics = {
            'Fouls p90': self._get_metric_value('fouls_committed'),
            'Cards p90': self._get_metric_value('yellow_cards') + self._get_metric_value('red_cards') * 2,
            'Turnovers p90': self._get_metric_value('turnovers'),
            'Dispossessions p90': self._get_metric_value('dispossessions'),
            'Errors p90': self._get_metric_value('errors_leading_to_shot'),
        }

        # Get percentiles (higher = more risky)
        risk_percentiles = {
            'Fouls p90': self._get_percentile('fouls_committed'),
            'Cards p90': self._get_percentile('yellow_cards'),
            'Turnovers p90': self._get_percentile('turnovers'),
            'Dispossessions p90': self._get_percentile('dispossessions'),
            'Errors p90': self._get_percentile('errors_leading_to_shot'),
        }

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['primary'])

        labels = list(risk_percentiles.keys())
        values = [risk_percentiles[k] / 100 for k in labels]  # Normalize to 0-1

        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles += angles[:1]

        # Draw the radar
        ax.plot(angles, values_plot, 'o-', linewidth=2, color=self.colors['negative'])
        ax.fill(angles, values_plot, alpha=0.25, color=self.colors['negative'])

        # Add a "safe zone" reference
        safe_zone = [0.5] * (num_vars + 1)
        ax.plot(angles, safe_zone, '--', linewidth=1, color=self.colors['neutral'], alpha=0.5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=9, color=self.colors['text'])

        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['Low', 'Avg', 'High', 'V.High'],
                          size=8, color=self.colors['text_secondary'])

        ax.spines['polar'].set_color(self.colors['neutral'])
        ax.grid(color=self.colors['neutral'], alpha=0.3)

        # Calculate overall risk score
        avg_risk = np.mean(values) * 100
        risk_label = "Low" if avg_risk < 40 else "Medium" if avg_risk < 60 else "High"

        ax.set_title(f"Risk Profile (Overall: {risk_label}, {avg_risk:.0f}th %ile avg)",
                    color=self.colors['text'], fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    def _generate_takeaways(self) -> List[Takeaway]:
        """Generate discipline takeaways."""
        takeaways = []

        # Fouls committed
        fouls = self._get_metric_value('fouls_committed')
        fouls_pct = self._get_percentile('fouls_committed')
        if fouls > 0:
            if fouls_pct >= 75:
                takeaways.append(Takeaway(
                    text=f"Foul-prone: {fouls:.1f} fouls committed per 90 ({fouls_pct:.0f}th percentile)",
                    metric='fouls_committed',
                    value=fouls,
                    percentile=fouls_pct,
                    is_positive=False
                ))
            elif fouls_pct <= 25:
                takeaways.append(Takeaway(
                    text=f"Clean defender: Only {fouls:.1f} fouls committed per 90",
                    metric='fouls_committed',
                    value=fouls,
                    percentile=fouls_pct,
                    is_positive=True
                ))

        # Fouls won
        fouls_won = self._get_metric_value('fouls_won')
        won_pct = self._get_percentile('fouls_won')
        if fouls_won > 0 and won_pct >= 75:
            takeaways.append(Takeaway(
                text=f"Draws fouls effectively: {fouls_won:.1f} fouls won per 90",
                metric='fouls_won',
                value=fouls_won,
                percentile=won_pct,
                is_positive=True
            ))

        # Cards
        yellows = self._get_metric_value('yellow_cards')
        if yellows > 0.2:  # More than 1 every 5 games
            takeaways.append(Takeaway(
                text=f"Card risk: {yellows:.2f} yellow cards per 90",
                metric='yellow_cards',
                value=yellows,
                is_positive=False
            ))

        # Turnovers
        turnovers = self._get_metric_value('turnovers')
        turnover_pct = self._get_percentile('turnovers')
        if turnovers > 0 and turnover_pct >= 75:
            takeaways.append(Takeaway(
                text=f"Possession security risk: {turnovers:.1f} turnovers per 90 ({turnover_pct:.0f}th percentile)",
                metric='turnovers',
                value=turnovers,
                percentile=turnover_pct,
                is_positive=False
            ))

        # Errors
        errors = self._get_metric_value('errors_leading_to_shot')
        if errors > 0.1:
            takeaways.append(Takeaway(
                text=f"Error-prone: {errors:.2f} errors leading to shots per 90",
                metric='errors_leading_to_shot',
                value=errors,
                is_positive=False
            ))

        # Foul differential
        if fouls > 0 and fouls_won > 0:
            diff = fouls_won - fouls
            if diff > 0.5:
                takeaways.append(Takeaway(
                    text=f"Positive foul differential: Wins {diff:.1f} more fouls than commits per 90",
                    is_positive=True
                ))

        return takeaways

    def _get_discipline_table(self) -> Dict[str, Any]:
        """Get discipline metrics table."""
        metrics = [
            'fouls_committed', 'fouls_won', 'yellow_cards', 'red_cards',
            'turnovers', 'dispossessions', 'miscontrols',
            'errors_leading_to_shot', 'errors_leading_to_goal', 'offsides'
        ]

        result = {}
        for metric in metrics:
            result[metric] = {
                'value': self._get_metric_value(metric),
                'percentile': self._get_percentile(metric),
            }
        return result
