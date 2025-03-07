"""
Automation Package for Lottery Prediction System

This package contains all components needed to run the automated prediction cycle.
"""

import os
import sys
import logging
from datetime import datetime

# Define project paths relatively rather than modifying sys.path
# This avoids potential conflicts with other path manipulations
automation_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(automation_dir)

# Set up package-level logging
log_file = os.path.join(automation_dir, 'automation_status.txt')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Package metadata
__version__ = '1.0.0'
__author__ = 'Mihai-Edward'

# Initialize logger
logger = logging.getLogger('lottery_automation')
logger.info(f"Automation package initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")