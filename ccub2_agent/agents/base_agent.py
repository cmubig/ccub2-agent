"""
Base Agent class for WorldCCUB Multi-Agent Loop.

All agents inherit from this base class to ensure consistent interface
and shared functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    country: str
    category: Optional[str] = None
    output_dir: Optional[Path] = None
    verbose: bool = False


@dataclass
class AgentResult:
    """Result from agent execution."""
    success: bool
    data: Dict[str, Any]
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """
    Base class for all WorldCCUB agents.
    
    Provides common functionality:
    - Configuration management
    - Logging
    - Result formatting
    - Error handling
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize agent with configuration.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        if config.verbose:
            self.logger.setLevel(logging.DEBUG)
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the agent's main task.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            AgentResult with execution results
        """
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data before execution.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        return True
    
    def log_execution(self, result: AgentResult):
        """Log execution result."""
        if result.success:
            self.logger.info(f"Execution successful: {result.message}")
        else:
            self.logger.error(f"Execution failed: {result.message}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(country={self.config.country})"
