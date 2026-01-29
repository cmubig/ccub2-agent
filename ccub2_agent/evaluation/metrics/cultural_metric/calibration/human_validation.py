"""Human Validation Protocol (C5).

Validates cultural metric against human judgments with:
- Krippendorff's alpha (inter-rater reliability, interval data)
- ICC(2,1) — two-way random, single measures
- ECE — Expected Calibration Error
- Per-dimension Pearson/Spearman correlations
- Insider vs outsider analysis
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Any, List, Optional, Sequence, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import json
import logging
import math
from datetime import datetime

import numpy as np
from scipy.stats import pearsonr, spearmanr

from ..taxonomy import CulturalDimension

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HumanJudgment:
    """A single human judgment."""
    image_id: str
    judge_id: str
    cultural_score: float  # 0-10
    prompt_score: float  # 0-10
    failure_modes: List[str]
    comments: str = ""
    timestamp: str = ""


@dataclass
class DimensionalHumanJudgment:
    """Human judgment with per-dimension scores (C5)."""
    image_id: str
    judge_id: str
    dimension_scores: Dict[str, float]  # dim_name -> 0-1
    overall_cultural_score: float  # 0-10
    prompt_score: float  # 0-10
    failure_modes: List[str]
    judge_country: str = ""  # judge's country of origin
    judge_expertise: str = "general"  # "insider", "outsider", "expert"
    comments: str = ""
    timestamp: str = ""


@dataclass
class MetricPrediction:
    """A metric prediction."""
    image_id: str
    cultural_score: float  # 0-10
    prompt_score: float  # 0-10
    failure_modes: List[str]
    confidence: float = 0.0
    cultscore: float = 0.0
    cultscore_confidence: float = 0.0
    dimension_scores: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core validation protocol
# ---------------------------------------------------------------------------

class HumanValidationProtocol:
    """Human validation protocol for metric calibration.

    Compares metric predictions with human judgments.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.human_judgments: List[HumanJudgment] = []
        self.metric_predictions: List[MetricPrediction] = []

        logger.info(f"HumanValidationProtocol initialized: output_dir={output_dir}")

    def add_human_judgment(
        self,
        image_id: str,
        judge_id: str,
        cultural_score: float,
        prompt_score: float,
        failure_modes: List[str],
        comments: str = "",
    ):
        judgment = HumanJudgment(
            image_id=image_id,
            judge_id=judge_id,
            cultural_score=cultural_score,
            prompt_score=prompt_score,
            failure_modes=failure_modes,
            comments=comments,
            timestamp=datetime.now().isoformat(),
        )
        self.human_judgments.append(judgment)
        logger.debug(f"Human judgment added: image_id={image_id}, judge_id={judge_id}")

    def add_metric_prediction(
        self,
        image_id: str,
        cultural_score: float,
        prompt_score: float,
        failure_modes: List[str],
        confidence: float = 0.0,
    ):
        prediction = MetricPrediction(
            image_id=image_id,
            cultural_score=cultural_score,
            prompt_score=prompt_score,
            failure_modes=failure_modes,
            confidence=confidence,
        )
        self.metric_predictions.append(prediction)
        logger.debug(f"Metric prediction added: image_id={image_id}")

    def compute_correlation(self) -> Dict[str, Any]:
        """Compute correlation between human judgments and metric predictions."""
        matched_pairs = []
        for judgment in self.human_judgments:
            prediction = next(
                (p for p in self.metric_predictions if p.image_id == judgment.image_id),
                None,
            )
            if prediction is not None:
                matched_pairs.append((judgment, prediction))

        if len(matched_pairs) < 2:
            logger.warning("Not enough matched pairs for correlation")
            return {
                "num_pairs": len(matched_pairs),
                "cultural_pearson": None,
                "cultural_spearman": None,
                "prompt_pearson": None,
                "prompt_spearman": None,
            }

        human_cultural = [j.cultural_score for j, p in matched_pairs]
        metric_cultural = [p.cultural_score for j, p in matched_pairs]
        human_prompt = [j.prompt_score for j, p in matched_pairs]
        metric_prompt = [p.prompt_score for j, p in matched_pairs]

        cultural_pearson, cultural_pearson_p = pearsonr(human_cultural, metric_cultural)
        cultural_spearman, cultural_spearman_p = spearmanr(human_cultural, metric_cultural)
        prompt_pearson, prompt_pearson_p = pearsonr(human_prompt, metric_prompt)
        prompt_spearman, prompt_spearman_p = spearmanr(human_prompt, metric_prompt)

        return {
            "num_pairs": len(matched_pairs),
            "cultural_pearson": float(cultural_pearson),
            "cultural_pearson_p": float(cultural_pearson_p),
            "cultural_spearman": float(cultural_spearman),
            "cultural_spearman_p": float(cultural_spearman_p),
            "prompt_pearson": float(prompt_pearson),
            "prompt_pearson_p": float(prompt_pearson_p),
            "prompt_spearman": float(prompt_spearman),
            "prompt_spearman_p": float(prompt_spearman_p),
        }

    def save_results(self, filename: Optional[str] = None) -> Path:
        if filename is None:
            filename = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        results_file = self.output_dir / filename
        correlation = self.compute_correlation()

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "num_judgments": len(self.human_judgments),
            "num_predictions": len(self.metric_predictions),
            "correlation": correlation,
            "judgments": [
                {
                    "image_id": j.image_id,
                    "judge_id": j.judge_id,
                    "cultural_score": j.cultural_score,
                    "prompt_score": j.prompt_score,
                    "failure_modes": j.failure_modes,
                }
                for j in self.human_judgments
            ],
            "predictions": [
                {
                    "image_id": p.image_id,
                    "cultural_score": p.cultural_score,
                    "prompt_score": p.prompt_score,
                    "failure_modes": p.failure_modes,
                    "confidence": p.confidence,
                }
                for p in self.metric_predictions
            ],
        }

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Validation results saved: {results_file}")
        return results_file


# ---------------------------------------------------------------------------
# C5. Enhanced Validation Protocol
# ---------------------------------------------------------------------------

class EnhancedHumanValidationProtocol:
    """Extended validation protocol with Krippendorff's alpha, ICC, ECE,
    per-dimension correlation, and insider vs outsider analysis.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.judgments: List[DimensionalHumanJudgment] = []
        self.predictions: List[MetricPrediction] = []
        logger.info(f"EnhancedHumanValidationProtocol initialized: {output_dir}")

    def add_judgment(self, judgment: DimensionalHumanJudgment) -> None:
        self.judgments.append(judgment)

    def add_prediction(self, prediction: MetricPrediction) -> None:
        self.predictions.append(prediction)

    # -- Krippendorff's alpha (interval data) --

    def compute_krippendorff_alpha(
        self,
        score_key: str = "overall_cultural_score",
    ) -> float:
        """Compute Krippendorff's alpha for inter-rater reliability.

        alpha = 1 - D_observed / D_expected

        Uses interval-level distance function: d(c,k) = (c - k)^2.
        """
        # Build reliability matrix: judges x items
        items = sorted(set(j.image_id for j in self.judgments))
        judges = sorted(set(j.judge_id for j in self.judgments))

        if len(items) < 2 or len(judges) < 2:
            return float("nan")

        # value_map[image_id][judge_id] = score
        value_map: Dict[str, Dict[str, float]] = defaultdict(dict)
        for j in self.judgments:
            score = j.overall_cultural_score if score_key == "overall_cultural_score" else j.prompt_score
            value_map[j.image_id][j.judge_id] = score

        # Collect all paired values per item
        n_total = 0
        d_observed = 0.0
        all_values: List[float] = []

        for item_id in items:
            item_vals = list(value_map[item_id].values())
            if len(item_vals) < 2:
                continue
            m_u = len(item_vals)
            n_total += m_u
            all_values.extend(item_vals)
            # Within-unit disagreement
            for i in range(m_u):
                for k in range(i + 1, m_u):
                    d_observed += (item_vals[i] - item_vals[k]) ** 2

        if n_total < 2:
            return float("nan")

        # Normalize observed disagreement
        # D_o = sum of squared diffs / (n * (n-1) / 2) across items
        n_pairs_observed = sum(
            len(list(value_map[item_id].values())) * (len(list(value_map[item_id].values())) - 1) / 2
            for item_id in items
            if len(list(value_map[item_id].values())) >= 2
        )
        if n_pairs_observed == 0:
            return float("nan")
        d_observed /= n_pairs_observed

        # Expected disagreement: all possible pairs across all values
        n_all = len(all_values)
        if n_all < 2:
            return float("nan")

        d_expected = 0.0
        for i in range(n_all):
            for k in range(i + 1, n_all):
                d_expected += (all_values[i] - all_values[k]) ** 2
        d_expected /= n_all * (n_all - 1) / 2

        if d_expected == 0:
            return 1.0  # perfect agreement

        alpha = 1.0 - d_observed / d_expected
        return float(alpha)

    # -- ICC(2,1) two-way random, single measures --

    def compute_icc(
        self,
        score_key: str = "overall_cultural_score",
    ) -> Tuple[float, float, float]:
        """Compute ICC(2,1) with 95% CI approximation.

        Returns:
            (icc, ci_lower, ci_upper)
        """
        items = sorted(set(j.image_id for j in self.judgments))
        judges = sorted(set(j.judge_id for j in self.judgments))

        n = len(items)
        k = len(judges)

        if n < 2 or k < 2:
            return float("nan"), float("nan"), float("nan")

        # Build n x k matrix (NaN for missing)
        matrix = np.full((n, k), np.nan)
        item_idx = {item: i for i, item in enumerate(items)}
        judge_idx = {judge: i for i, judge in enumerate(judges)}

        for j in self.judgments:
            r = item_idx.get(j.image_id)
            c = judge_idx.get(j.judge_id)
            if r is not None and c is not None:
                score = j.overall_cultural_score if score_key == "overall_cultural_score" else j.prompt_score
                matrix[r, c] = score

        # Filter rows with full data only
        valid_mask = ~np.isnan(matrix).any(axis=1)
        matrix = matrix[valid_mask]
        n = matrix.shape[0]

        if n < 2:
            return float("nan"), float("nan"), float("nan")

        grand_mean = np.mean(matrix)
        row_means = np.mean(matrix, axis=1)
        col_means = np.mean(matrix, axis=0)

        # Sum of squares
        ss_total = np.sum((matrix - grand_mean) ** 2)
        ss_rows = k * np.sum((row_means - grand_mean) ** 2)  # between subjects
        ss_cols = n * np.sum((col_means - grand_mean) ** 2)  # between raters
        ss_error = ss_total - ss_rows - ss_cols  # residual

        # Mean squares
        ms_rows = ss_rows / (n - 1) if n > 1 else 0
        ms_cols = ss_cols / (k - 1) if k > 1 else 0
        ms_error = ss_error / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0

        # ICC(2,1)
        denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
        if denom == 0:
            return float("nan"), float("nan"), float("nan")

        icc = (ms_rows - ms_error) / denom

        # Approximate 95% CI using F distribution approximation
        # Simplified Fleiss (1986) bounds
        f_val = ms_rows / ms_error if ms_error > 0 else float("inf")
        if f_val == float("inf") or n < 3:
            return float(icc), float("nan"), float("nan")

        # F critical values approximated as 1.96-based intervals
        fl = f_val / (1 + 1.96 * math.sqrt(2.0 / (k * (n - 1))))
        fu = f_val * (1 + 1.96 * math.sqrt(2.0 / (k * (n - 1))))

        ci_lower = (fl - 1) / (fl + k - 1)
        ci_upper = (fu - 1) / (fu + k - 1)

        return float(icc), float(ci_lower), float(ci_upper)

    # -- ECE: Expected Calibration Error --

    def compute_ece(
        self,
        num_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error between predicted confidence
        and actual accuracy (metric vs human agreement).

        Bins predictions by confidence, computes per-bin accuracy,
        then weighted average of |accuracy - confidence| per bin.
        """
        # Match predictions to human judgments
        human_map: Dict[str, List[float]] = defaultdict(list)
        for j in self.judgments:
            human_map[j.image_id].append(j.overall_cultural_score)

        bins: List[List[Tuple[float, float]]] = [[] for _ in range(num_bins)]

        for p in self.predictions:
            human_scores = human_map.get(p.image_id)
            if not human_scores:
                continue

            mean_human = sum(human_scores) / len(human_scores)
            # "Accuracy" as 1 - |normalized difference|
            if p.cultscore > 0:
                pred_on_10_scale = p.cultscore * 10
            else:
                pred_on_10_scale = p.cultural_score
            accuracy = 1.0 - abs(mean_human - pred_on_10_scale) / 10.0
            accuracy = max(0.0, min(1.0, accuracy))

            confidence = max(0.0, min(1.0, p.cultscore_confidence if p.cultscore_confidence > 0 else p.confidence))
            bin_idx = min(int(confidence * num_bins), num_bins - 1)
            bins[bin_idx].append((accuracy, confidence))

        total = sum(len(b) for b in bins)
        if total == 0:
            return float("nan")

        ece = 0.0
        for b in bins:
            if not b:
                continue
            avg_acc = sum(a for a, c in b) / len(b)
            avg_conf = sum(c for a, c in b) / len(b)
            ece += len(b) / total * abs(avg_acc - avg_conf)

        return float(ece)

    # -- Per-dimension correlation --

    def compute_per_dimension_correlation(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Compute Pearson and Spearman per cultural dimension.

        Returns dict mapping dimension name -> {pearson, spearman, pearson_p, spearman_p}.
        """
        pred_map: Dict[str, MetricPrediction] = {
            p.image_id: p for p in self.predictions
        }

        # Aggregate human dimension scores per image (average across judges)
        human_dim_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for j in self.judgments:
            for dim_name, score in j.dimension_scores.items():
                human_dim_scores[j.image_id][dim_name].append(score)

        results: Dict[str, Dict[str, Optional[float]]] = {}

        all_dims = set()
        for j in self.judgments:
            all_dims.update(j.dimension_scores.keys())

        for dim_name in sorted(all_dims):
            human_vals = []
            metric_vals = []
            for image_id, dim_dict in human_dim_scores.items():
                if dim_name not in dim_dict:
                    continue
                pred = pred_map.get(image_id)
                if pred is None or dim_name not in pred.dimension_scores:
                    continue
                human_vals.append(np.mean(dim_dict[dim_name]))
                metric_vals.append(pred.dimension_scores[dim_name])

            if len(human_vals) < 3:
                results[dim_name] = {
                    "pearson": None, "spearman": None,
                    "pearson_p": None, "spearman_p": None,
                    "n": len(human_vals),
                }
                continue

            pr, pp = pearsonr(human_vals, metric_vals)
            sr, sp = spearmanr(human_vals, metric_vals)
            results[dim_name] = {
                "pearson": float(pr),
                "spearman": float(sr),
                "pearson_p": float(pp),
                "spearman_p": float(sp),
                "n": len(human_vals),
            }

        return results

    # -- Insider vs outsider analysis --

    def insider_vs_outsider_analysis(
        self,
        target_country: str,
    ) -> Dict[str, Any]:
        """Compare insider (same-country) vs outsider evaluations.

        Returns:
            Dict with insider/outsider means, stds, and t-test p-value.
        """
        insider_scores: List[float] = []
        outsider_scores: List[float] = []

        for j in self.judgments:
            if j.judge_country.lower() == target_country.lower():
                insider_scores.append(j.overall_cultural_score)
            else:
                outsider_scores.append(j.overall_cultural_score)

        result: Dict[str, Any] = {
            "target_country": target_country,
            "n_insider": len(insider_scores),
            "n_outsider": len(outsider_scores),
        }

        if insider_scores:
            result["insider_mean"] = float(np.mean(insider_scores))
            result["insider_std"] = float(np.std(insider_scores, ddof=1)) if len(insider_scores) > 1 else 0.0
        else:
            result["insider_mean"] = None
            result["insider_std"] = None

        if outsider_scores:
            result["outsider_mean"] = float(np.mean(outsider_scores))
            result["outsider_std"] = float(np.std(outsider_scores, ddof=1)) if len(outsider_scores) > 1 else 0.0
        else:
            result["outsider_mean"] = None
            result["outsider_std"] = None

        # Welch's t-test if both groups have enough data
        if len(insider_scores) >= 2 and len(outsider_scores) >= 2:
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(insider_scores, outsider_scores, equal_var=False)
            result["t_statistic"] = float(t_stat)
            result["p_value"] = float(p_val)
            result["significant_05"] = p_val < 0.05
        else:
            result["t_statistic"] = None
            result["p_value"] = None
            result["significant_05"] = None

        return result

    # -- Full validation report --

    def generate_validation_report(
        self,
        target_countries: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive validation report for paper inclusion."""
        report: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "n_judgments": len(self.judgments),
            "n_predictions": len(self.predictions),
            "n_judges": len(set(j.judge_id for j in self.judgments)),
            "n_items": len(set(j.image_id for j in self.judgments)),
        }

        # Inter-rater reliability
        report["krippendorff_alpha_cultural"] = self.compute_krippendorff_alpha("overall_cultural_score")
        report["krippendorff_alpha_prompt"] = self.compute_krippendorff_alpha("prompt_score")

        # ICC
        icc, ci_lo, ci_hi = self.compute_icc("overall_cultural_score")
        report["icc_cultural"] = {"icc": icc, "ci_lower": ci_lo, "ci_upper": ci_hi}
        icc_p, ci_lo_p, ci_hi_p = self.compute_icc("prompt_score")
        report["icc_prompt"] = {"icc": icc_p, "ci_lower": ci_lo_p, "ci_upper": ci_hi_p}

        # ECE
        report["ece"] = self.compute_ece()

        # Per-dimension correlation
        report["dimension_correlations"] = self.compute_per_dimension_correlation()

        # Insider vs outsider
        if target_countries:
            report["insider_outsider"] = {
                country: self.insider_vs_outsider_analysis(country)
                for country in target_countries
            }

        return report

    def save_report(self, filename: Optional[str] = None) -> Path:
        """Save validation report to JSON."""
        if filename is None:
            filename = f"enhanced_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.output_dir / filename
        report = self.generate_validation_report()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Enhanced validation report saved: {path}")
        return path


# ---------------------------------------------------------------------------
# Utility functions (backward compatible)
# ---------------------------------------------------------------------------

def run_validation_study(
    human_judgments: List[HumanJudgment],
    metric_predictions: List[MetricPrediction],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run a validation study (legacy interface)."""
    protocol = HumanValidationProtocol(output_dir)

    for judgment in human_judgments:
        protocol.add_human_judgment(
            image_id=judgment.image_id,
            judge_id=judgment.judge_id,
            cultural_score=judgment.cultural_score,
            prompt_score=judgment.prompt_score,
            failure_modes=judgment.failure_modes,
            comments=judgment.comments,
        )

    for prediction in metric_predictions:
        protocol.add_metric_prediction(
            image_id=prediction.image_id,
            cultural_score=prediction.cultural_score,
            prompt_score=prediction.prompt_score,
            failure_modes=prediction.failure_modes,
            confidence=prediction.confidence,
        )

    correlation = protocol.compute_correlation()
    protocol.save_results()
    return correlation


def compute_correlation(
    human_scores: List[float],
    metric_scores: List[float],
) -> Dict[str, float]:
    """Compute correlation between human and metric scores."""
    if len(human_scores) != len(metric_scores):
        raise ValueError("Human and metric scores must have same length")

    pearson_r, pearson_p = pearsonr(human_scores, metric_scores)
    spearman_r, spearman_p = spearmanr(human_scores, metric_scores)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }
