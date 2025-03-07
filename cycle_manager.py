
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

# Add both project root and src to Python path
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now try importing with different possible paths
try:
    # Try importing from src directory first
    from src.data_collector_selenium import KinoDataCollector
    from src.draw_handler import DrawHandler, perform_complete_analysis, train_and_predict
    from src.prediction_evaluator import PredictionEvaluator
except ImportError:
    try:
        # Try importing directly
        from data_collector_selenium import KinoDataCollector
        from draw_handler import DrawHandler, perform_complete_analysis, train_and_predict
        from prediction_evaluator import PredictionEvaluator
    except ImportError as e:
        print("Error importing required modules!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        print(f"Project root: {project_root}")
        print(f"Src directory: {src_dir}")
        raise ImportError(f"Failed to import required modules: {str(e)}")

# Import from automation package
from automation.scheduler import DrawScheduler

class PredictionCycleManager:
    def __init__(self):
        """Initialize the cycle manager with default settings."""
        # Core configuration
        self.consecutive_failures = 0
        self.max_failures = 5
        self.retry_delay = 30  # seconds
        self.fetch_retries = 3
        
        # Initialize components
        self.collector = KinoDataCollector()
        self.handler = DrawHandler()
        self.evaluator = PredictionEvaluator()
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.current_state = 'initializing'
        self.cycle_count = 0
        self.failures = 0
        self.last_successful_cycle = None
        self.last_collection_time = None
        self.last_prediction_time = None
        
        # Initialize scheduler
        self.scheduler = DrawScheduler()
        
        # Initialize status tracking
        self.status_file = Path(__file__).parent / 'automation_status.txt'
        self.update_status("Initialized")

    def update_status(self, status_message):
        """Update automation status."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.status_file, 'a') as f:
                f.write(f"{timestamp}: {status_message}\n")
            self.current_state = status_message
            self.logger.info(status_message)
        except Exception as e:
            self.logger.error(f"Failed to write status: {e}")

    def collect_data(self):
        """Collect latest draw data."""
        try:
            self.update_status("Collecting latest draw data...")
            success = self.collector.fetch_latest_draws(num_draws=1)
            if success:
                self.last_collection_time = datetime.now(pytz.UTC)
                self.update_status("Data collection successful")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Data collection error: {e}")
            return False

    def generate_prediction(self):
        """Generate predictions for next draw."""
        try:
            self.update_status("Generating predictions...")
            predictions, probabilities, analysis = train_and_predict()
            if predictions is not None:
                self.last_prediction_time = datetime.now(pytz.UTC)
                self.update_status(f"Generated prediction: {sorted(predictions)}")
                return predictions, probabilities, analysis
            return None, None, None
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return None, None, None

    def evaluate_results(self):
        """Evaluate prediction results."""
        try:
            self.update_status("Evaluating results...")
            success = self.evaluator.evaluate_latest_prediction()
            if success:
                self.update_status("Evaluation completed successfully")
            return success
        except Exception as e:
            self.logger.error(f"Evaluation error: {e}")
            return False

    def run_cycle(self):
        """Run the complete automation cycle"""
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
                    if not self.collect_data():
                        self.failures += 1
                        error_msg = f"Data collection failed (Attempt {self.failures}/{self.max_failures})"
                        self.logger.error(error_msg)
                        self.update_status(error_msg)
                        if self.failures >= self.max_failures:
                            raise Exception("Maximum failures reached in data collection")
                        time.sleep(self.retry_delay)
                        continue
                    
                    # 2. Generate Prediction
                    self.update_status("Generating prediction...")
                    predictions, probabilities, analysis = self.generate_prediction()
                    if predictions is None:
                        self.failures += 1
                        error_msg = f"Prediction generation failed (Attempt {self.failures}/{self.max_failures})"
                        self.logger.error(error_msg)
                        self.update_status(error_msg)
                        if self.failures >= self.max_failures:
                            raise Exception("Maximum failures reached in prediction")
                        time.sleep(self.retry_delay)
                        continue
                    
                    # Log prediction details
                    self.logger.info(f"Generated prediction: {sorted(predictions)}")
                    self.logger.info(f"Prediction probabilities: {probabilities}")
                    
                    # 3. Wait for draw time
                    wait_time = (next_draw - current_time).total_seconds()
                    if wait_time > 0:
                        self.update_status(f"Waiting {wait_time:.0f} seconds until next draw...")
                        self.logger.info(f"Waiting {wait_time:.0f} seconds until next draw at {next_draw.strftime('%H:%M:%S')}")
                        
                        # Progress updates for long waits
                        remaining_time = wait_time
                        while remaining_time > 0:
                            time.sleep(min(60, remaining_time))
                            remaining_time -= 60
                            if remaining_time > 0:
                                self.logger.info(f"Remaining wait time: {remaining_time:.0f} seconds")
                    
                    # 4. Wait post-draw interval
                    post_draw_msg = f"Waiting {self.scheduler.post_draw_wait_seconds} seconds for draw results..."
                    self.update_status(post_draw_msg)
                    self.logger.info(post_draw_msg)
                    time.sleep(self.scheduler.post_draw_wait_seconds)
                    
                    # 5. Evaluate Results
                    self.update_status("Evaluating prediction results...")
                    if not self.evaluate_results():
                        self.logger.warning("Evaluation completed with warnings")
                    
                    # Reset failure count and update success metrics
                    self.failures = 0
                    self.cycle_count += 1
                    self.last_successful_cycle = current_time
                    
                    success_msg = f"Completed cycle {self.cycle_count} successfully"
                    self.update_status(success_msg)
                    self.logger.info(success_msg)
                    
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
        """Get current automation status"""
        return {
            'current_state': self.current_state,
            'cycle_count': self.cycle_count,
            'failures': self.failures,
            'last_collection': self.last_collection_time,
            'last_prediction': self.last_prediction_time,
            'last_successful_cycle': self.last_successful_cycle
        }
if __name__ == "__main__":
    # Simple test of the cycle manager
    print("Testing cycle manager...")
    manager = PredictionCycleManager()
    
    # Print scheduler details for next draw
    next_draw = manager.scheduler.get_next_draw_time()
    eval_time = manager.scheduler.get_evaluation_time(next_draw)
    
    print(f"Current time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Next draw at: {next_draw.strftime('%H:%M:%S')}")
    print(f"Evaluate at: {eval_time.strftime('%H:%M:%S')}")
    
    # Don't actually run the cycle in test mode
    print("Test complete. Import this module and call run_cycle() to start automation.")