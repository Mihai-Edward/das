# File: automation/cycle_manager.py
import os
import sys
import time
from datetime import datetime
import pytz
import logging
from pathlib import Path

# Add the project root and src directories to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from src.data_collector_selenium import KinoDataCollector
from src.draw_handler import DrawHandler
from src.prediction_evaluator import PredictionEvaluator
from automation.scheduler import DrawScheduler
from automation.operations import fetch_latest_draws, perform_analysis, generate_prediction, evaluate_prediction

class PredictionCycleManager:
    def __init__(self):
        """Initialize the cycle manager with default settings."""
        # Set up logging
        self._setup_logging()
        
        # Core configuration
        self.consecutive_failures = 0
        self.max_failures = 5
        self.retry_delay = 30  # seconds
        self.fetch_retries = 3
        
        # Initialize components
        self.collector = KinoDataCollector(user_login="Mihai-Edward")
        self.handler = DrawHandler()
        self.evaluator = PredictionEvaluator()
        self.scheduler = DrawScheduler()
        
        # State management
        self.current_state = 'initializing'
        self.cycle_count = 0
        self.failures = 0
        self.last_successful_cycle = None
        self.last_collection_time = None
        self.last_prediction_time = None
        
        # Initialize status tracking
        self.status_file = Path(current_dir) / 'automation_status.txt'
        self.update_status("Initialized")
        
        self.logger.info("PredictionCycleManager initialized")
        self.logger.info(f"Current UTC time: {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')}")

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path(current_dir) / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f'automation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)

    def update_status(self, status_message):
        """Update automation status."""
        try:
            timestamp = datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')
            with open(self.status_file, 'a') as f:
                f.write(f"{timestamp} UTC: {status_message}\n")
            self.current_state = status_message
            self.logger.info(status_message)
        except Exception as e:
            self.logger.error(f"Failed to write status: {e}")

    def run_cycle(self):
        """Run the complete automation cycle."""
        while True:
            try:
                # Get current time and next draw time in UTC
                current_time = datetime.now(pytz.UTC)
                next_draw = self.scheduler.get_next_draw_time(current_time)
                
                # Update cycle status
                self.update_status(f"Starting cycle {self.cycle_count + 1}")
                self.logger.info(f"Starting cycle {self.cycle_count + 1} at {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                self.logger.info(f"Next draw scheduled for: {next_draw.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                
                try:
                    # 1. Collect Data
                    self.update_status("Collecting latest draw data...")
                    success, result = fetch_latest_draws(max_retries=self.fetch_retries)
                    if not success:
                        raise Exception(f"Data collection failed: {result}")
                    
                    # 2. Analyze Data
                    self.update_status("Analyzing collected data...")
                    success, result = perform_analysis(result)
                    if not success:
                        raise Exception(f"Analysis failed: {result}")
                    
                    # 3. Generate Prediction
                    self.update_status("Generating prediction...")
                    success, result = generate_prediction(next_draw)
                    if not success:
                        raise Exception(f"Prediction failed: {result}")
                    
                    predictions, probabilities, analysis = result
                    
                    # 4. Wait for draw time
                    wait_time = (next_draw - current_time).total_seconds()
                    if wait_time > 0:
                        self.update_status(f"Waiting {wait_time:.0f} seconds until next draw...")
                        self.logger.info(f"Waiting {wait_time:.0f} seconds until draw at {next_draw.strftime('%H:%M:%S UTC')}")
                        
                        # Progress updates every minute for long waits
                        remaining_time = wait_time
                        while remaining_time > 0:
                            time.sleep(min(60, remaining_time))
                            remaining_time -= 60
                            if remaining_time > 0:
                                self.logger.info(f"Remaining wait time: {remaining_time:.0f} seconds")
                    
                    # 5. Wait post-draw interval
                    post_draw_msg = f"Waiting {self.scheduler.post_draw_wait_seconds} seconds for draw results..."
                    self.update_status(post_draw_msg)
                    time.sleep(self.scheduler.post_draw_wait_seconds)
                    
                    # 6. Evaluate Results
                    self.update_status("Evaluating prediction results...")
                    success, eval_result = evaluate_prediction(next_draw)
                    if not success:
                        self.logger.warning(f"Evaluation warning: {eval_result}")
                    
                    # Reset failure count and update success metrics
                    self.failures = 0
                    self.cycle_count += 1
                    self.last_successful_cycle = current_time
                    
                    success_msg = f"Completed cycle {self.cycle_count} successfully"
                    self.update_status(success_msg)
                    
                except Exception as cycle_error:
                    self.failures += 1
                    error_msg = f"Cycle error: {str(cycle_error)}"
                    self.logger.error(error_msg)
                    self.update_status(f"Cycle failed: {str(cycle_error)}")
                    
                    if self.failures >= self.max_failures:
                        raise Exception(f"Maximum failures ({self.max_failures}) reached")
                    
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    
            except Exception as fatal_error:
                error_msg = f"Fatal error in automation cycle: {str(fatal_error)}"
                self.logger.critical(error_msg)
                self.update_status("Automation stopped due to fatal error")
                raise

    def get_status(self):
        """Get current automation status."""
        return {
            'current_state': self.current_state,
            'cycle_count': self.cycle_count,
            'failures': self.failures,
            'last_collection': self.last_collection_time,
            'last_prediction': self.last_prediction_time,
            'last_successful_cycle': self.last_successful_cycle,
            'next_draw': self.scheduler.get_next_draw_time().strftime('%Y-%m-%d %H:%M:%S UTC')
        }

if __name__ == "__main__":
    # Test the cycle manager
    print("Testing cycle manager...")
    manager = PredictionCycleManager()
    
    # Print scheduler details for next draw
    next_draw = manager.scheduler.get_next_draw_time()
    eval_time = manager.scheduler.get_evaluation_time(next_draw)
    
    print(f"Current time UTC: {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Next draw at: {next_draw.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Evaluate at: {eval_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    print("\nTest complete. Import this module and call run_cycle() to start automation.")