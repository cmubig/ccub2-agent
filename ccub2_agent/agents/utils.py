"""
Utility functions for WorldCCUB agents.
"""

from typing import Dict, Any, List
from pathlib import Path
import json


def load_country_config(country: str) -> Dict[str, Any]:
    """Load country-specific configuration."""
    config_file = Path(__file__).parent.parent.parent.parent / "data" / "country_configs" / f"{country}.json"
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Return default config
    return {
        "country": country,
        "categories": [
            "traditional_clothing",
            "architecture",
            "food",
            "festivals",
            "art",
            "music_dance",
            "religious_cultural",
            "daily_life"
        ],
        "supported_languages": ["en", "ko", "zh", "ja"]
    }


def save_agent_log(agent_name: str, result: Dict[str, Any], output_dir: Path):
    """Save agent execution log."""
    log_file = output_dir / f"{agent_name}_log.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
