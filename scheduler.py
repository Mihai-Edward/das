import os
import sys
from datetime import datetime, timedelta
import pytz
import logging
from typing import Optional, Tuple

class DrawScheduler:
    """
    Handles all timing-related operations for the lottery prediction system.
    Ensures precise timing for draw cycles and state transitions.
    """
    def __init__(self, 
                 draw_interval_minutes: int = 5,
                 post_draw_wait_seconds: int = 50,
                 timing_tolerance_seconds: int = 5):
        # Core timing settings
        self.draw_interval_minutes = draw_interval_minutes
        self.post_draw_wait_seconds = post_draw_wait_seconds
        self.timing_tolerance = timing_tolerance_seconds
        
        # Timezone setting (standardized to Bucharest)
        self.timezone = pytz.timezone('Europe/Bucharest')
        
        # Initialize timing tracking
        self.last_draw_time = None
        self.next_draw_time = None
        self.last_evaluation_time = None
        
        # Set up logging
        self.logger = logging.getLogger('DrawScheduler')
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for the scheduler"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def get_current_time(self) -> datetime:
        """Get current time in Bucharest timezone"""
        return datetime.now(self.timezone)

    def get_next_draw_time(self, reference_time: Optional[datetime] = None) -> datetime:
        """Calculate the next draw time based on the reference time"""
        if reference_time is None:
            reference_time = self.get_current_time()
            
        # Calculate minutes until next draw
        minutes_past = reference_time.minute % self.draw_interval_minutes
        minutes_until = self.draw_interval_minutes - minutes_past
        
        # Calculate next draw time
        next_draw = reference_time + timedelta(minutes=minutes_until)
        next_draw = next_draw.replace(second=0, microsecond=0)
        
        # Update internal tracking
        self.next_draw_time = next_draw
        
        self.logger.debug(f"Next draw time calculated: {next_draw.strftime('%Y-%m-%d %H:%M:%S')}")
        return next_draw

    def get_evaluation_time(self, draw_time: datetime) -> datetime:
        """Calculate when to evaluate results for a given draw time"""
        return draw_time + timedelta(seconds=self.post_draw_wait_seconds)

    def is_within_tolerance(self, time1: datetime, time2: datetime, 
                          custom_tolerance: Optional[int] = None) -> bool:
        """Check if two times are within the tolerance window"""
        tolerance = custom_tolerance if custom_tolerance is not None else self.timing_tolerance
        delta = abs((time1 - time2).total_seconds())
        return delta <= tolerance

    def get_wait_times(self) -> Tuple[float, float, float]:
        """
        Calculate various wait times from current moment
        Returns: (seconds_to_next_draw, seconds_to_evaluation, seconds_to_fetch)
        """
        current_time = self.get_current_time()
        next_draw = self.get_next_draw_time(current_time)
        eval_time = self.get_evaluation_time(next_draw)
        
        time_to_draw = (next_draw - current_time).total_seconds()
        time_to_eval = (eval_time - current_time).total_seconds()
        
        # Calculate optimal fetch time (typically 50 seconds after last draw)
        last_draw = next_draw - timedelta(minutes=self.draw_interval_minutes)
        fetch_time = last_draw + timedelta(seconds=self.post_draw_wait_seconds)
        time_to_fetch = (fetch_time - current_time).total_seconds()
        
        return time_to_draw, time_to_eval, time_to_fetch

    def should_transition_state(self, current_state: str, 
                              target_time: datetime,
                              tolerance: Optional[int] = None) -> bool:
        """
        Determine if it's time to transition from the current state
        based on target time and tolerance
        """
        current_time = self.get_current_time()
        return self.is_within_tolerance(current_time, target_time, tolerance)

    def get_formatted_wait_time(self, seconds: float) -> str:
        """Format wait time in a human-readable format"""
        if seconds <= 0:
            return "0s"
            
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        
        if minutes > 0:
            return f"{minutes}m {remaining_seconds}s"
        return f"{remaining_seconds}s"

    def validate_and_track_timing(self, operation_name: str, 
                                start_time: datetime,
                                max_duration_seconds: int = 30) -> bool:
        """
        Validate operation timing and track performance
        Returns True if operation completed within expected timeframe
        """
        end_time = self.get_current_time()
        duration = (end_time - start_time).total_seconds()
        
        self.logger.info(f"{operation_name} completed in {duration:.2f} seconds")
        
        if duration > max_duration_seconds:
            self.logger.warning(
                f"{operation_name} took longer than expected: {duration:.2f}s > {max_duration_seconds}s"
            )
            return False
        return True

    def is_draw_time(self, tolerance: Optional[int] = None) -> bool:
        """Check if current time is a draw time within tolerance"""
        current_time = self.get_current_time()
        minutes_past = current_time.minute % self.draw_interval_minutes
        return abs(minutes_past) <= (tolerance or self.timing_tolerance)

    def is_evaluation_time(self, tolerance: Optional[int] = None) -> bool:
        """Check if it's time to evaluate based on last draw"""
        if self.next_draw_time is None:
            return False
            
        current_time = self.get_current_time()
        eval_time = self.get_evaluation_time(self.next_draw_time)
        return self.is_within_tolerance(current_time, eval_time, tolerance)

    def log_timing_status(self):
        """Log current timing status for monitoring"""
        time_to_draw, time_to_eval, time_to_fetch = self.get_wait_times()
        
        self.logger.info(
            f"Timing Status:\n"
            f"- Next draw in: {self.get_formatted_wait_time(time_to_draw)}\n"
            f"- Next evaluation in: {self.get_formatted_wait_time(time_to_eval)}\n"
            f"- Next fetch in: {self.get_formatted_wait_time(time_to_fetch)}"
        )

if __name__ == "__main__":
    # Test the scheduler
    scheduler = DrawScheduler()
    current_time = scheduler.get_current_time()
    next_draw = scheduler.get_next_draw_time()
    eval_time = scheduler.get_evaluation_time(next_draw)
    
    print(f"Current time (Bucharest): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Next draw at: {next_draw.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Evaluation at: {eval_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test wait times
    scheduler.log_timing_status()