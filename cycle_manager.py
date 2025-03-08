import os
import sys
import time
from datetime import datetime
from enum import Enum
import logging
from typing import Optional, Dict, Any, Tuple

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scheduler import DrawScheduler
from config.paths import PATHS, ensure_directories

class PredictionState(Enum):
    """States for the prediction cycle state machine"""
    STARTUP = "startup"
    FETCHING = "fetching"
    ANALYZING = "analyzing"
    PREDICTING = "predicting"
    LEARNING = "learning"
    WAITING = "waiting"
    EVALUATING = "evaluating"
    ERROR = "error"

class CycleManager:
    def __init__(self, 
                 max_failures: int = 5,
                 retry_delay: int = 30,
                 enable_learning: bool = False):
        """
        Initialize the cycle manager with configuration parameters
        """
        # Initialize scheduler
        self.scheduler = DrawScheduler()
        
        # Configuration
        self.max_failures = max_failures
        self.retry_delay = retry_delay
        self.continuous_learning_enabled = enable_learning
        
        # State tracking
        self.state = PredictionState.STARTUP
        self.active = True
        self.consecutive_failures = 0
        
        # Performance tracking
        self.cycle_metrics = {
            'last_success': None,
            'last_error': None,
            'cycle_count': 0,
            'state_transitions': [],
            'operation_durations': {}
        }
        
        # Set up logging
        self.logger = logging.getLogger('CycleManager')
        self._setup_logging()
        
        # Initialize paths
        ensure_directories()
        
        self.logger.info("Cycle Manager initialized")
        self._log_configuration()

    def _setup_logging(self):
        """Configure logging for the cycle manager"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _log_configuration(self):
        """Log current configuration settings"""
        self.logger.info(
            f"\nConfiguration:"
            f"\n- Max failures: {self.max_failures}"
            f"\n- Retry delay: {self.retry_delay}s"
            f"\n- Continuous learning: {'Enabled' if self.continuous_learning_enabled else 'Disabled'}"
            f"\n- Initial state: {self.state.value}"
        )

    def _track_state_transition(self, from_state: PredictionState, to_state: PredictionState):
        """Track state transitions for monitoring"""
        transition_time = self.scheduler.get_current_time()
        self.state_transitions.append({
            'from': from_state.value,
            'to': to_state.value,
            'time': transition_time
        })
        self.logger.info(f"State transition: {from_state.value} -> {to_state.value}")

    def _handle_failure(self, operation: str, error: str) -> bool:
        """
        Handle operation failures with retry logic
        Returns: True if should retry, False if should stop
        """
        self.consecutive_failures += 1
        self.last_error = {
            'operation': operation,
            'error': error,
            'time': self.scheduler.get_current_time()
        }
        
        self.logger.error(
            f"Failure in {operation}: {error}\n"
            f"Consecutive failures: {self.consecutive_failures}/{self.max_failures}"
        )
        
        if self.consecutive_failures >= self.max_failures:
            self.logger.critical("Maximum consecutive failures reached, stopping cycle")
            self.active = False
            return False
            
        time.sleep(self.retry_delay)
        return True

    def _transition_state(self, new_state: PredictionState):
        """Safely transition to a new state with logging"""
        old_state = self.state
        self.state = new_state
        self._track_state_transition(old_state, new_state)

    async def _handle_startup(self) -> bool:
        """Handle system startup and initialization"""
        try:
            self.logger.info("System startup initiated")
            
            # Determine current time and next draw
            current_time = self.scheduler.get_current_time()
            next_draw = self.scheduler.get_next_draw_time()
            
            self.logger.info(
                f"Startup complete:"
                f"\n- Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
                f"\n- Next draw: {next_draw.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # Transition to fetching state
            self._transition_state(PredictionState.FETCHING)
            return True
            
        except Exception as e:
            self.logger.error(f"Startup failed: {str(e)}")
            self._handle_failure("startup", str(e))
            return False

    async def _handle_fetching(self) -> bool:
        """Handle data fetching state"""
        try:
            from data_collector_selenium import KinoDataCollector
            
            self.logger.info("Starting data collection")
            collector = KinoDataCollector()
            
            # Execute fetch with retry capability
            start_time = self.scheduler.get_current_time()
            draws = collector.fetch_latest_draws()
            
            if not draws:
                raise Exception("No draws collected")
                
            # Validate timing
            self.scheduler.validate_and_track_timing("Data collection", start_time)
            
            # On success, transition to analyzing
            self._transition_state(PredictionState.ANALYZING)
            self.consecutive_failures = 0
            return True
            
        except Exception as e:
            return self._handle_failure("fetching", str(e))

    async def _handle_analyzing(self) -> bool:
        """Handle data analysis state"""
        try:
            from data_analysis import DataAnalysis
            
            self.logger.info("Starting data analysis")
            start_time = self.scheduler.get_current_time()
            
            # Execute analysis with collected data
            analysis = DataAnalysis(self.collected_draws)
            analysis_results = analysis.perform_complete_analysis()
            
            if not analysis_results:
                raise Exception("Analysis failed to produce results")
                
            # Validate timing
            self.scheduler.validate_and_track_timing("Data analysis", start_time)
            
            # On success, transition to predicting
            self._transition_state(PredictionState.PREDICTING)
            self.consecutive_failures = 0
            return True
            
        except Exception as e:
            return self._handle_failure("analysis", str(e))

    async def _handle_predicting(self) -> bool:
        """Handle prediction generation state"""
        try:
            from lottery_predictor import LotteryPredictor
            
            self.logger.info("Generating predictions")
            start_time = self.scheduler.get_current_time()
            
            predictor = LotteryPredictor()
            predictions = predictor.generate_predictions()
            
            if not predictions:
                raise Exception("Failed to generate predictions")
                
            # Validate timing
            self.scheduler.validate_and_track_timing("Prediction generation", start_time)
            
            # On success, transition to learning or waiting based on configuration
            if self.continuous_learning_enabled:
                self._transition_state(PredictionState.LEARNING)
            else:
                self._transition_state(PredictionState.WAITING)
                
            self.consecutive_failures = 0
            return True
            
        except Exception as e:
            return self._handle_failure("prediction", str(e))

    async def _handle_learning(self) -> bool:
        """Handle continuous learning state"""
        try:
            if not self.continuous_learning_enabled:
                self.logger.info("Continuous learning disabled, skipping")
                self._transition_state(PredictionState.WAITING)
                return True
                
            self.logger.info("Executing continuous learning cycle")
            start_time = self.scheduler.get_current_time()
            
            # Execute learning cycle
            from draw_handler import DrawHandler
            handler = DrawHandler()
            success = handler.run_continuous_learning_cycle()
            
            # Validate timing
            self.scheduler.validate_and_track_timing("Continuous learning", start_time)
            
            # Always transition to waiting, regardless of success
            self._transition_state(PredictionState.WAITING)
            
            # Learning failures don't count towards consecutive failures
            if not success:
                self.logger.warning("Learning cycle completed with warnings")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Learning error (non-critical): {str(e)}")
            self._transition_state(PredictionState.WAITING)
            return True

    async def _handle_waiting(self) -> bool:
        """Handle waiting state with precise timing"""
        try:
            next_draw = self.scheduler.get_next_draw_time()
            eval_time = self.scheduler.get_evaluation_time(next_draw)
            
            while self.active:
                current_time = self.scheduler.get_current_time()
                
                # Check if it's time to evaluate
                if self.scheduler.is_within_tolerance(current_time, eval_time):
                    self._transition_state(PredictionState.EVALUATING)
                    return True
                    
                # Calculate and log waiting time
                time_to_eval = (eval_time - current_time).total_seconds()
                if time_to_eval > 60:
                    self.logger.info(f"Waiting {int(time_to_eval/60)}m {int(time_to_eval%60)}s until evaluation")
                    time.sleep(30)  # Longer sleep for longer waits
                else:
                    self.logger.info(f"Waiting {int(time_to_eval)}s until evaluation")
                    time.sleep(5)  # Shorter sleep for final countdown
                    
            return True
            
        except Exception as e:
            return self._handle_failure("waiting", str(e))

    async def _handle_evaluating(self) -> bool:
        """Handle evaluation state"""
        try:
            from prediction_evaluator import PredictionEvaluator
            
            self.logger.info("Starting prediction evaluation")
            start_time = self.scheduler.get_current_time()
            
            evaluator = PredictionEvaluator()
            success = evaluator.evaluate_past_predictions()
            
            # Validate timing
            self.scheduler.validate_and_track_timing("Evaluation", start_time)
            
            if success:
                self.consecutive_failures = 0
                self.last_success = self.scheduler.get_current_time()
            
            # Always transition back to fetching
            self._transition_state(PredictionState.FETCHING)
            return True
            
        except Exception as e:
            self.logger.error(f"Evaluation error: {str(e)}")
            self._transition_state(PredictionState.FETCHING)
            return True

    async def run_cycle(self):
        """Main cycle execution loop"""
        state_handlers = {
            PredictionState.STARTUP: self._handle_startup,
            PredictionState.FETCHING: self._handle_fetching,
            PredictionState.ANALYZING: self._handle_analyzing,
            PredictionState.PREDICTING: self._handle_predicting,
            PredictionState.LEARNING: self._handle_learning,
            PredictionState.WAITING: self._handle_waiting,
            PredictionState.EVALUATING: self._handle_evaluating
        }
        
        self.logger.info("Starting prediction cycle")
        
        while self.active:
            try:
                handler = state_handlers.get(self.state)
                if handler:
                    if not await handler():
                        break
                else:
                    self.logger.error(f"No handler for state: {self.state}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Critical error in cycle: {str(e)}")
                if not self._handle_failure("cycle", str(e)):
                    break
                    
        self.logger.info("Prediction cycle stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'active': self.active,
            'state': self.state.value,
            'consecutive_failures': self.consecutive_failures,
            'last_success': self.last_success,
            'last_error': self.last_error,
            'cycle_count': self.cycle_metrics['cycle_count'],
            'next_draw': self.scheduler.next_draw_time,
            'next_evaluation': self.scheduler.get_evaluation_time(self.scheduler.next_draw_time)
        }

if __name__ == "__main__":
    # Test the cycle manager
    manager = CycleManager(enable_learning=True)
    import asyncio
    asyncio.run(manager.run_cycle())