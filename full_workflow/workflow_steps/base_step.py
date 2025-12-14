#!/usr/bin/env python3
"""
Base class for workflow steps.

Provides common functionality and access to runner context.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
from pathlib import Path
from datetime import datetime
import subprocess


class BaseStep(ABC):
    """Abstract base class for workflow steps."""
    
    def __init__(self, runner):
        """
        Initialize step with runner context.
        
        Args:
            runner: WorkflowRunner instance providing config, dirs, and utilities
        """
        self.runner = runner
    
    @abstractmethod
    def execute(self) -> bool:
        """
        Execute the step.
        
        Returns:
            True if step completed successfully, False otherwise
        """
        pass
    
    @property
    def config(self):
        """Access to configuration dictionary."""
        return self.runner.config
    
    @property
    def dirs(self):
        """Access to directory paths dictionary."""
        return self.runner.dirs
    
    def run_command(self, cmd: list, step_name: str, log_file: Optional[Union[str, Path]] = None) -> int:
        """Run a command and log output (delegates to runner)."""
        return self.runner.run_command(cmd, step_name, log_file)
    
    def should_process_gt(self) -> bool:
        """Check if GT should be processed (delegates to runner)."""
        return self.runner.should_process_gt()
    
    def should_process_pt(self) -> bool:
        """Check if PT should be processed (delegates to runner)."""
        return self.runner.should_process_pt()
    
    def get_process_mode(self) -> str:
        """Get process mode (delegates to runner)."""
        return self.runner.get_process_mode()


