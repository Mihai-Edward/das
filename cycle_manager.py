# automation/cycle_manager.py

import os
import sys
import time
from datetime import datetime, timedelta

# Add project root to path if necessary
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary modules
from automation.scheduler import DrawScheduler
from config.paths import PATHS, ensure_directories

class PredictionCycleManager:
    def __init__(self):
        """Initialize the cycle manager with default settings."""
        # Status tracking
        self.active = True
        self.status = "initializing"
        self.last_success = None
        self.last_error = None
        self.consecutive_failures = 0
        
        # Configuration
        self.max_failures = 5
        self.retry_delay = 30  # seconds
        self.fetch_retries = 3
        self.continuous_learning_enabled = False  # Default to disabled
        
        # Initialize scheduler
        self.scheduler = DrawScheduler()
        
        # Log initialization
        self._log_status("Prediction cycle manager initialized")
        
    def get_status(self):
        """Get the current status of the cycle manager."""
        return {
            'active': self.active,
            'status': self.status,
            'last_success': self.last_success,
            'last_error': self.last_error,
            'consecutive_failures': self.consecutive_failures,
            'next_draw': self.scheduler.get_next_draw_time(),
            'next_eval': self.scheduler.get_evaluation_time(self.scheduler.get_next_draw_time())
        }
        
    def _log_status(self, message):
        """Log a status message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}")
        
    def _handle_failure(self, stage, message):
        """Handle a failure during operation."""
        self.consecutive_failures += 1
        self.last_error = {'stage': stage, 'message': message, 'time': datetime.now()}
        self._log_status(f"Failure in {stage}: {message}")
        self._log_status(f"Consecutive failures: {self.consecutive_failures}/{self.max_failures}")
        
        # Exit if max failures reached
        if self.consecutive_failures >= self.max_failures:
            self._log_status("Maximum consecutive failures reached, stopping cycle")
            self.active = False
        
        # Add delay before retrying
        time.sleep(self.retry_delay)
        
    def collect_data(self):
        """Collect data for prediction."""
        from automation.operations import collect_data_operation
        success, message = collect_data_operation(num_draws=self.fetch_retries * 3)
        return success, message
        
    def analyze_data(self):
        """Analyze collected data."""
        from automation.operations import analyze_data_operation
        success, message = analyze_data_operation()
        return success, message
        
    def generate_prediction(self, for_draw_time=None):
        """Generate prediction for next draw."""
        from automation.operations import generate_prediction_operation
        success, message = generate_prediction_operation(for_draw_time=for_draw_time)
        return success, message
        
    def evaluate_prediction(self):
        """Evaluate past predictions."""
        from automation.operations import evaluate_prediction_operation
        success, message = evaluate_prediction_operation()
        
        # Track status and return results
        if success:
            self.last_success = datetime.now()
            self.consecutive_failures = 0
        return success, message
        
    def run_continuous_learning(self):
        """Run the continuous learning cycle after evaluation."""
        from automation.operations import run_continuous_learning
        success, message = run_continuous_learning()
        
        # Track status and return results
        if success:
            self.last_success = datetime.now()
            self.consecutive_failures = 0
            return True, "Continuous learning completed successfully"
        else:
            # Don't increment failure counter for learning failures
            # as this is not a critical operation
            return False, f"Continuous learning failed: {message}"
        
    def run_cycle(self):
        """Run the main prediction cycle."""
        while self.active and self.consecutive_failures < self.max_failures:
            try:
                # Get the next draw time
                next_draw_time = self.scheduler.get_next_draw_time()
                evaluation_time = self.scheduler.get_evaluation_time(next_draw_time)
                
                # Calculate wait time
                now = datetime.now()
                
                # 1. Check if we should do prediction for next draw
                if now < next_draw_time - timedelta(minutes=10):  # Predict at least 10 min before draw
                    self.status = "predicting"
                    self._log_status(f"Generating prediction for draw at {next_draw_time.strftime('%H:%M:%S')}")
                    
                    # A) Collect latest draw data
                    success, message = self.collect_data()
                    if not success:
                        self._handle_failure("data_collection", message)
                        continue
                        
                    # B) Analyze collected data
                    success, message = self.analyze_data()
                    if not success:
                        self._handle_failure("analysis", message)
                        continue
                    
                    # C) Generate prediction
                    success, message = self.generate_prediction(for_draw_time=next_draw_time)
                    if not success:
                        self._handle_failure("prediction", message)
                        continue
                        
                    # D) Run continuous learning if enabled
                    if self.continuous_learning_enabled:
                        self._log_status("Running continuous learning...")
                        success, message = self.run_continuous_learning()
                        if success:
                            self._log_status("Continuous learning completed successfully")
                        else:
                            self._log_status(f"Note: Continuous learning skipped or failed: {message}")
                    
                    # Success! Record and log
                    self.last_success = now
                    self.consecutive_failures = 0
                    self._log_status(f"Prediction complete, waiting for draw at {next_draw_time.strftime('%H:%M:%S')}")
                    
                    # Wait until just after the draw time
                    next_check_time = evaluation_time
                    
                # 2. Check if we should evaluate a recent draw
                elif now >= evaluation_time and now < next_draw_time:
                    self.status = "evaluating"
                    self._log_status(f"Evaluating draw at {next_draw_time.strftime('%H:%M:%S')}")
                    
                    # Run evaluation
                    success, message = self.evaluate_prediction()
                    if not success:
                        self._handle_failure("evaluation", message)
                        next_check_time = next_draw_time
                    else:
                        self._log_status("Evaluation complete")
                        next_check_time = next_draw_time
                
                # 3. Otherwise, wait for the next appropriate time
                else:
                    self.status = "waiting"
                    next_check_time = min(next_draw_time - timedelta(minutes=10), evaluation_time)
                    if now < next_check_time:
                        wait_time = (next_check_time - now).total_seconds()
                        self._log_status(f"Waiting {int(wait_time/60)} minutes {int(wait_time%60)} seconds until next action")
                        
                # Sleep for a bit before checking again  
                time.sleep(30)
                
            except Exception as e:
                self._handle_failure("cycle", str(e))
                time.sleep(self.retry_delay)