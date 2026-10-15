"""Final recommendation section."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .base import BaseSection, SectionContent, Takeaway


class RecommendationSection(BaseSection):
    """Final recommendation with fit score and risk assessment."""

    @property
    def section_title(self) -> str:
        return "Recommendation"

    def generate(self) -> SectionContent:
        """Generate the recommendation section."""
        content = SectionContent(title=self.section_title)

        # Get recommendation data
        recommendation = self._calculate_recommendation()
        content.raw_data['recommendation'] = recommendation

        # Create recommendation visual
        rec_fig = self._create_recommendation_visual(recommendation)
        if rec_fig:
            content.add_figure(rec_fig, "Recommendation Summary")

        # Create fit score breakdown
        fit_fig = self._create_fit_breakdown(recommendation)
        if fit_fig:
            content.add_figure(fit_fig, "Fit Score Breakdown")

        # Create risk assessment
        risk_fig = self._create_risk_assessment(recommendation)
        if risk_fig:
            content.add_figure(risk_fig, "Risk Assessment")

        # Create strengths/weaknesses summary
        summary_fig = self._create_summary_visual()
        if summary_fig:
            content.add_figure(summary_fig, "Strengths & Areas to Develop")

        # Generate takeaways
        takeaways = self._generate_takeaways(recommendation)
        for takeaway in takeaways:
            content.takeaways.append(takeaway)

        return content

    def _calculate_recommendation(self) -> Dict[str, Any]:
        """Calculate the overall recommendation."""
        # Get fit score
        if hasattr(self.cohort, 'fit_score'):
            fit_score = self.cohort.fit_score
        else:
            # Calculate from percentiles
            if hasattr(self.cohort, 'percentiles') and self.cohort.percentiles:
                percentiles = [r.percentile for r in self.cohort.percentiles.values()]
                fit_score = np.mean(percentiles) if percentiles else 50
            else:
                fit_score = 50

        # Calculate risk score
        risk_factors = self._calculate_risk_factors()
        risk_score = np.mean(list(risk_factors.values())) if risk_factors else 30

        # Determine verdict
        thresholds = self.config.get('recommendation', {}).get('thresholds', {})
        recommend_threshold = thresholds.get('recommend', 70)
        monitor_threshold = thresholds.get('monitor', 50)

        # Adjust fit score by risk
        adjusted_score = fit_score - (risk_score * 0.2)

        if adjusted_score >= recommend_threshold:
            verdict = "RECOMMEND"
            verdict_color = self.colors['positive']
        elif adjusted_score >= monitor_threshold:
            verdict = "MONITOR"
            verdict_color = self.colors['highlight']
        else:
            verdict = "PASS"
            verdict_color = self.colors['negative']

        # Generate reason
        reason = self._generate_recommendation_reason(verdict, fit_score, risk_score)

        return {
            'verdict': verdict,
            'verdict_color': verdict_color,
            'fit_score': fit_score,
            'risk_score': risk_score,
            'adjusted_score': adjusted_score,
            'risk_factors': risk_factors,
            'reason': reason,
        }

    def _calculate_risk_factors(self) -> Dict[str, float]:
        """Calculate individual risk factors."""
        risk_factors = {}

        # Injury history (placeholder - would need external data)
        risk_factors['injury_history'] = 20  # Default low

        # Card rate
        yellows = self._get_metric_value('yellow_cards')
        reds = self._get_metric_value('red_cards')
        card_rate = (yellows + reds * 2) * 10  # Per 90 * 10
        risk_factors['discipline'] = min(100, card_rate * 10)

        # Turnover rate
        turnovers = self._get_metric_value('turnovers')
        turnover_pct = self._get_percentile('turnovers')
        risk_factors['possession_security'] = turnover_pct

        # Consistency (variance in performance)
        match_metrics = getattr(self.metrics, 'match_by_match', None)
        if match_metrics and len(match_metrics) >= 5:
            key_values = []
            for match in match_metrics:
                if isinstance(match, dict):
                    key_values.append(match.get('touches', 0))
                else:
                    key_values.append(getattr(match, 'touches', 0))

            if key_values and np.mean(key_values) > 0:
                cv = np.std(key_values) / np.mean(key_values) * 100
                risk_factors['consistency'] = min(100, cv * 2)
            else:
                risk_factors['consistency'] = 50
        else:
            risk_factors['consistency'] = 50

        # Age curve (simplified)
        age = self.player_data.info.age if hasattr(self.player_data.info, 'age') else 25
        if age < 23:
            risk_factors['age_curve'] = 20  # Young, high upside
        elif age < 28:
            risk_factors['age_curve'] = 10  # Prime
        elif age < 32:
            risk_factors['age_curve'] = 40  # Post-prime
        else:
            risk_factors['age_curve'] = 70  # Declining

        return risk_factors

    def _generate_recommendation_reason(self, verdict: str, fit_score: float,
                                        risk_score: float) -> str:
        """Generate explanation for the recommendation."""
        position = self.player_data.info.position_group
        name = self.player_data.info.name

        if verdict == "RECOMMEND":
            return (f"{name} profiles as an above-average {position} "
                    f"(fit score: {fit_score:.0f}) with manageable risk factors "
                    f"(risk score: {risk_score:.0f}). Performance metrics suggest "
                    f"they would be a valuable addition.")

        elif verdict == "MONITOR":
            return (f"{name} shows potential but has either moderate performance "
                    f"(fit score: {fit_score:.0f}) or elevated risk factors "
                    f"(risk score: {risk_score:.0f}). Continue monitoring for "
                    f"sustained improvement or reduced risk.")

        else:  # PASS
            return (f"{name}'s profile does not meet requirements for the {position} "
                    f"position (fit score: {fit_score:.0f}). Either performance "
                    f"is below threshold or risk factors are too high "
                    f"(risk score: {risk_score:.0f}).")

    def _create_recommendation_visual(self, recommendation: Dict[str, Any]) -> Figure:
        """Create main recommendation visual."""
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['primary'])

        # Large verdict text
        verdict = recommendation['verdict']
        verdict_color = recommendation['verdict_color']

        ax.text(0.5, 0.7, verdict, ha='center', va='center',
               fontsize=48, fontweight='bold', color=verdict_color,
               transform=ax.transAxes)

        # Scores
        fit_score = recommendation['fit_score']
        risk_score = recommendation['risk_score']

        ax.text(0.3, 0.4, f"Fit Score\n{fit_score:.0f}/100",
               ha='center', va='center', fontsize=16, color=self.colors['text'],
               transform=ax.transAxes)

        ax.text(0.7, 0.4, f"Risk Score\n{risk_score:.0f}/100",
               ha='center', va='center', fontsize=16, color=self.colors['text'],
               transform=ax.transAxes)

        # Reason
        reason = recommendation['reason']
        ax.text(0.5, 0.15, reason, ha='center', va='center',
               fontsize=10, color=self.colors['text_secondary'],
               wrap=True, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor=self.colors['secondary'],
                        edgecolor=self.colors['neutral']))

        ax.axis('off')

        ax.set_title(f"Final Recommendation - {self.player_data.info.name}",
                    color=self.colors['text'], fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_fit_breakdown(self, recommendation: Dict[str, Any]) -> Figure:
        """Create fit score breakdown."""
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        # Get top metrics contributing to fit
        if hasattr(self.cohort, 'percentiles') and self.cohort.percentiles:
            sorted_metrics = sorted(self.cohort.percentiles.items(),
                                   key=lambda x: x[1].percentile, reverse=True)

            # Top 5 strengths and bottom 5 weaknesses
            top_5 = sorted_metrics[:5]
            bottom_5 = sorted_metrics[-5:]

            all_metrics = top_5 + bottom_5
            labels = [m[0].replace('_', ' ').title()[:20] for m in all_metrics]
            values = [m[1].percentile for m in all_metrics]

            y_pos = np.arange(len(labels))

            colors = []
            for val in values:
                if val >= 75:
                    colors.append(self.colors['positive'])
                elif val >= 50:
                    colors.append(self.colors['highlight'])
                elif val >= 25:
                    colors.append(self.colors['neutral'])
                else:
                    colors.append(self.colors['negative'])

            bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='white')

            ax.axvline(x=50, color=self.colors['text'], linestyle='--', alpha=0.5)
            ax.axvline(x=75, color=self.colors['positive'], linestyle=':', alpha=0.3)
            ax.axvline(x=25, color=self.colors['negative'], linestyle=':', alpha=0.3)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, color=self.colors['text'], fontsize=9)
            ax.set_xlabel('Percentile', color=self.colors['text'])
            ax.set_xlim(0, 100)

            ax.invert_yaxis()

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title('Key Metrics Breakdown', color=self.colors['text'],
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_risk_assessment(self, recommendation: Dict[str, Any]) -> Figure:
        """Create risk assessment visualization."""
        risk_factors = recommendation.get('risk_factors', {})

        if not risk_factors:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(self.colors['primary'])
        ax.set_facecolor(self.colors['secondary'])

        labels = [k.replace('_', ' ').title() for k in risk_factors.keys()]
        values = list(risk_factors.values())

        y_pos = np.arange(len(labels))

        colors = []
        for val in values:
            if val <= 30:
                colors.append(self.colors['positive'])
            elif val <= 50:
                colors.append(self.colors['highlight'])
            elif val <= 70:
                colors.append(self.colors['neutral'])
            else:
                colors.append(self.colors['negative'])

        bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='white')

        # Risk thresholds
        ax.axvline(x=30, color=self.colors['positive'], linestyle=':', alpha=0.5,
                  label='Low Risk')
        ax.axvline(x=70, color=self.colors['negative'], linestyle=':', alpha=0.5,
                  label='High Risk')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color=self.colors['text'])
        ax.set_xlabel('Risk Level', color=self.colors['text'])
        ax.set_xlim(0, 100)

        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['neutral'])
        ax.spines['left'].set_color(self.colors['neutral'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='lower right', facecolor=self.colors['secondary'],
                 edgecolor=self.colors['neutral'], labelcolor=self.colors['text'])

        overall_risk = np.mean(values)
        ax.set_title(f'Risk Assessment (Overall: {overall_risk:.0f}/100)',
                    color=self.colors['text'], fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_summary_visual(self) -> Figure:
        """Create strengths and weaknesses summary."""
        if not hasattr(self.cohort, 'percentiles') or not self.cohort.percentiles:
            return None

        sorted_metrics = sorted(self.cohort.percentiles.items(),
                               key=lambda x: x[1].percentile, reverse=True)

        strengths = [(m, p.percentile) for m, p in sorted_metrics if p.percentile >= 70][:5]
        weaknesses = [(m, p.percentile) for m, p in sorted_metrics if p.percentile <= 30][:5]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor(self.colors['primary'])

        # Strengths
        ax1.set_facecolor(self.colors['secondary'])
        if strengths:
            labels = [m.replace('_', ' ').title()[:18] for m, _ in strengths]
            values = [p for _, p in strengths]

            y_pos = np.arange(len(labels))
            ax1.barh(y_pos, values, color=self.colors['positive'], alpha=0.8, edgecolor='white')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(labels, color=self.colors['text'])
            ax1.axvline(x=70, color=self.colors['text'], linestyle='--', alpha=0.5)
            ax1.set_xlim(0, 100)
            ax1.invert_yaxis()
        else:
            ax1.text(0.5, 0.5, 'No standout strengths', ha='center', va='center',
                    color=self.colors['text_secondary'], transform=ax1.transAxes)

        ax1.set_title('Strengths (70th+ %ile)', color=self.colors['positive'],
                     fontsize=11, fontweight='bold')
        ax1.tick_params(colors=self.colors['text'])
        ax1.spines['bottom'].set_color(self.colors['neutral'])
        ax1.spines['left'].set_color(self.colors['neutral'])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Weaknesses
        ax2.set_facecolor(self.colors['secondary'])
        if weaknesses:
            labels = [m.replace('_', ' ').title()[:18] for m, _ in weaknesses]
            values = [p for _, p in weaknesses]

            y_pos = np.arange(len(labels))
            ax2.barh(y_pos, values, color=self.colors['negative'], alpha=0.8, edgecolor='white')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels, color=self.colors['text'])
            ax2.axvline(x=30, color=self.colors['text'], linestyle='--', alpha=0.5)
            ax2.set_xlim(0, 100)
            ax2.invert_yaxis()
        else:
            ax2.text(0.5, 0.5, 'No major weaknesses', ha='center', va='center',
                    color=self.colors['text_secondary'], transform=ax2.transAxes)

        ax2.set_title('Areas to Develop (30th- %ile)', color=self.colors['negative'],
                     fontsize=11, fontweight='bold')
        ax2.tick_params(colors=self.colors['text'])
        ax2.spines['bottom'].set_color(self.colors['neutral'])
        ax2.spines['left'].set_color(self.colors['neutral'])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.suptitle('Profile Summary', color=self.colors['text'],
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig

    def _generate_takeaways(self, recommendation: Dict[str, Any]) -> List[Takeaway]:
        """Generate final recommendation takeaways."""
        takeaways = []

        verdict = recommendation['verdict']
        fit_score = recommendation['fit_score']
        risk_score = recommendation['risk_score']

        takeaways.append(Takeaway(
            text=f"Verdict: {verdict} (Fit: {fit_score:.0f}, Risk: {risk_score:.0f})",
            is_positive=verdict == "RECOMMEND"
        ))

        # Key strengths
        if hasattr(self.cohort, 'percentiles') and self.cohort.percentiles:
            sorted_metrics = sorted(self.cohort.percentiles.items(),
                                   key=lambda x: x[1].percentile, reverse=True)

            if sorted_metrics:
                top = sorted_metrics[0]
                takeaways.append(Takeaway(
                    text=f"Primary strength: {top[0].replace('_', ' ').title()} ({top[1].percentile:.0f}th %ile)",
                    is_positive=True
                ))

            # Key weakness
            bottom = sorted_metrics[-1]
            if bottom[1].percentile <= 30:
                takeaways.append(Takeaway(
                    text=f"Key development area: {bottom[0].replace('_', ' ').title()} ({bottom[1].percentile:.0f}th %ile)",
                    is_positive=False
                ))

        # Risk summary
        risk_factors = recommendation.get('risk_factors', {})
        if risk_factors:
            high_risks = [k for k, v in risk_factors.items() if v >= 60]
            if high_risks:
                takeaways.append(Takeaway(
                    text=f"Risk concerns: {', '.join(r.replace('_', ' ').title() for r in high_risks)}",
                    is_positive=False
                ))
            else:
                takeaways.append(Takeaway(
                    text="No significant risk factors identified",
                    is_positive=True
                ))

        return takeaways
