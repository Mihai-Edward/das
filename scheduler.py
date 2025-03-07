# File: automation/scheduler.py
import os
import sys
from datetime import datetime, timedelta
import math

class DrawScheduler:
    """Handles scheduling for draw times and waiting periods."""
    
    def __init__(self, draw_interval_minutes=5, post_draw_wait_seconds=50):
        """
        Initialize the scheduler with specified parameters.
        
        Args:
            draw_interval_minutes: Time between lottery draws in minutes (default 5)
            post_draw_wait_seconds: Time to wait after draw before evaluating (default 50)
        """
        self.draw_interval_minutes = draw_interval_minutes
        self.post_draw_wait_seconds = post_draw_wait_seconds
    
    def get_next_draw_time(self, reference_time=None):
        """
        Calculate the time of the next draw.
        
        Args:
            reference_time: Optional time reference (defaults to current time)
            
        Returns:
            datetime: Time of the next draw
        """
        if reference_time is None:
            reference_time = datetime.now()
            
        # Calculate minutes past the hour
        minutes_past = reference_time.minute % self.draw_interval_minutes
        
        # If we're exactly at a draw time, the next one is interval minutes away
        # Otherwise, we need to calculate the time until the next interval mark
        if minutes_past == 0 and reference_time.second == 0:
            # Exactly at a draw time, next one is interval away
            next_draw = reference_time + timedelta(minutes=self.draw_interval_minutes)
        else:
            # Calculate minutes until next draw
            minutes_until = self.draw_interval_minutes - minutes_past
            
            # Create the next draw time
            next_draw = reference_time + timedelta(minutes=minutes_until)
            
            # Reset seconds and microseconds
            next_draw = next_draw.replace(second=0, microsecond=0)
        
        return next_draw
    
    def get_evaluation_time(self, draw_time):
        """
        Calculate the time to start evaluation after a draw.
        
        Args:
            draw_time: The time of the draw
            
        Returns:
            datetime: Time to begin evaluation
        """
        return draw_time + timedelta(seconds=self.post_draw_wait_seconds)
    
    def seconds_until_time(self, target_time):
        """
        Calculate the number of seconds until a target time.
        
        Args:
            target_time: The target datetime
            
        Returns:
            int: Number of seconds until target time (0 if in the past)
        """
        now = datetime.now()
        if target_time <= now:
            return 0
            
        delta = target_time - now
        return delta.total_seconds()


def get_next_draw_time(reference_time=None):
    """Convenience function to get the next draw time."""
    scheduler = DrawScheduler()
    return scheduler.get_next_draw_time(reference_time)


def get_seconds_until(target_time):
    """Convenience function to get seconds until a target time."""
    now = datetime.now()
    if target_time <= now:
        return 0
        
    delta = target_time - now
    return delta.total_seconds()


def get_formatted_time_remaining(target_time):
    """
    Format the time remaining until target time in a human-readable way.
    
    Args:
        target_time: The target datetime
        
    Returns:
        str: Formatted time remaining (e.g., "2h 30m 15s")
    """
    seconds = get_seconds_until(target_time)
    
    # If time is in the past
    if seconds <= 0:
        return "0s"
        
    # Calculate hours, minutes, seconds
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds = math.floor(seconds % 60)
    
    # Format the string
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
        
    return " ".join(parts)


if __name__ == "__main__":
    # Simple test of the scheduler
    scheduler = DrawScheduler()
    now = datetime.now()
    
    next_draw = scheduler.get_next_draw_time()
    evaluation_time = scheduler.get_evaluation_time(next_draw)
    
    print(f"Current time: {now.strftime('%H:%M:%S')}")
    print(f"Next draw at: {next_draw.strftime('%H:%M:%S')}")
    print(f"Evaluate at: {evaluation_time.strftime('%H:%M:%S')}")
    print(f"Time until next draw: {get_formatted_time_remaining(next_draw)}")
    print(f"Time until evaluation: {get_formatted_time_remaining(evaluation_time)}")