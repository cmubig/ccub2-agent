"""
Benchmark Splits

Standardized splits for CultureBench-Global evaluation.
Ensures exact reproducibility across runs.
"""

from typing import Dict, List, Optional
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from datetime import datetime

import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSplit:
    """
    A benchmark split definition.
    
    Contains exact image IDs and metadata for reproducibility.
    """
    split_name: str  # e.g., "task_a_train", "task_b_test"
    task: str  # "fidelity", "degradation", "transfer", "portrait", "contrastive"
    split_type: str  # "train", "val", "test"
    image_ids: List[str]  # Exact image IDs in this split
    countries: List[str]  # Countries included
    categories: List[str]  # Categories included
    seed: int  # Random seed used for split
    created_at: str = ""
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


def get_default_splits() -> Dict[str, BenchmarkSplit]:
    """
    Get default benchmark splits.
    
    Returns:
        Dictionary mapping split names to BenchmarkSplit objects
    """
    splits = {}
    
    # Task A: Cultural Fidelity Scoring
    splits["task_a_train"] = BenchmarkSplit(
        split_name="task_a_train",
        task="fidelity",
        split_type="train",
        image_ids=[],  # Will be populated from actual data
        countries=["korea", "japan", "china", "india", "thailand"],
        categories=["traditional_clothing", "food", "architecture"],
        seed=42,
    )
    
    splits["task_a_test"] = BenchmarkSplit(
        split_name="task_a_test",
        task="fidelity",
        split_type="test",
        image_ids=[],
        countries=["korea", "japan", "china", "india", "thailand"],
        categories=["traditional_clothing", "food", "architecture"],
        seed=42,
    )
    
    # Task B: Iterative I2I Degradation
    splits["task_b_test"] = BenchmarkSplit(
        split_name="task_b_test",
        task="degradation",
        split_type="test",
        image_ids=[],
        countries=["korea", "japan"],
        categories=["traditional_clothing"],
        seed=42,
    )
    
    # Task C: Cross-Country Restylization
    splits["task_c_test"] = BenchmarkSplit(
        split_name="task_c_test",
        task="transfer",
        split_type="test",
        image_ids=[],
        countries=["korea", "japan", "china"],
        categories=["traditional_clothing", "food"],
        seed=42,
    )
    
    # Task D: Contrastive Evaluation
    splits["task_d_test"] = BenchmarkSplit(
        split_name="task_d_test",
        task="contrastive",
        split_type="test",
        image_ids=[],
        countries=["korea", "japan", "china", "india"],
        categories=["traditional_clothing"],
        seed=42,
    )
    
    return splits


def save_benchmark_split(split: BenchmarkSplit, split_file: Path):
    """
    Save a benchmark split to JSON file.
    
    Args:
        split: BenchmarkSplit instance
        split_file: Path to save JSON file
    """
    split_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split.to_dict(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"Benchmark split saved: {split_file}")


def load_benchmark_split(split_file: Path) -> BenchmarkSplit:
    """
    Load a benchmark split from JSON file.
    
    Args:
        split_file: Path to JSON file
        
    Returns:
        BenchmarkSplit instance
    """
    with open(split_file, "r", encoding="utf-8") as f:
        split_dict = json.load(f)
    
    split = BenchmarkSplit(**split_dict)
    
    logger.info(f"Benchmark split loaded: {split_file}")
    return split
