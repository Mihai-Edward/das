"""
Lottery Automation System
Author: Mihai-Edward
Last Updated: 2025-03-08 12:28:48 UTC
Description: Automated system for lottery prediction with timezone handling and task management
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timedelta
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
from logging.handlers import RotatingFileHandler
import pytz

# Import your existing components
from draw_handler import DrawHandler
from data_collector_selenium import KinoDataCollector
from prediction_evaluator import PredictionEvaluator
from config.paths import PATHS, ensure_directories
from data_analysis import DataAnalysis

class TaskQueue:
    """Manages task prioritization and execution"""
    
    def __init__(self):
        self.critical_queue = deque()
        self.background_queue = deque()
        self.results = {}
        
    def add_critical(self, task_name, task_func, *args):
        self.critical_queue.append((task_name, task_func, args))
        
    def add_background(self, task_name, task_func, *args):
        self.background_queue.append((task_name, task_func, args))
        
    def get_critical_task(self):
        return self.critical_queue.popleft() if self.critical_queue else None
        
    def get_background_task(self):
        return self.background_queue.popleft() if self.background_queue else None
        
    def store_result(self, task_name, result):
        self.results[task_name] = {
            'result': result,
            'timestamp': datetime.now()
        }

class LotteryAutomation:
    """Main automation class for lottery prediction system"""
    
    def __init__(self):
        print(f"Initializing automation system...")
        
        # 1. Set up base directories first
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Base directory: {self.base_dir}")
        
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        print(f"Creating logs directory: {self.logs_dir}")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # 2. Set up logging
        print("Setting up logging...")
        self._setup_logging()
        
        # 3. Ensure all required directories exist
        ensure_directories()
        
        # 4. Set up timezone
        print("Configuring timezone...")
        self.timezone = pytz.timezone('Europe/Bucharest')  # UTC+2
        self.utc = pytz.UTC
        current_time = datetime.now(self.utc)
        local_time = current_time.astimezone(self.timezone)
        print(f"Current UTC time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Current local time: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # 5. Initialize components
        print("Initializing components...")
        self.handler = DrawHandler()
        self.collector = KinoDataCollector(user_login="Mihai-Edward")
        self.evaluator = PredictionEvaluator()
        
        # 6. Initialize task management
        self.task_queue = TaskQueue()
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        
        # 7. Initialize execution tracking
        self.last_execution_time = None
        self.execution_stats = {
            'cycles_completed': 0,
            'successful_cycles': 0,
            'failed_cycles': 0,
            'average_execution_time': 0,
            'last_critical_execution_time': 0,
            'last_analysis_time': None
        }
        
        print("Initialization complete!")

    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('LotteryAutomation')
        self.logger.setLevel(logging.INFO)
        
        log_file = os.path.join(self.logs_dir, 'lottery_automation.log')
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10000000,
            backupCount=5
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def get_local_time(self):
        """Get current time in local timezone (UTC+2)"""
        utc_now = datetime.now(self.utc)
        local_time = utc_now.astimezone(self.timezone)
        return local_time

    def _is_critical_window(self):
        """Check if we're in the critical 50-second window"""
        local_now = self.get_local_time()
        seconds_past_five = (local_now.minute % 5) * 60 + local_now.second
        return 50 <= seconds_past_five <= 100

    def _calculate_next_window(self):
        """Calculate time until next execution window"""
        local_now = self.get_local_time()
        minutes_to_next = 5 - (local_now.minute % 5)
        if minutes_to_next == 5 and local_now.second < 50:
            return timedelta(seconds=50 - local_now.second)
        return timedelta(minutes=minutes_to_next - 1, seconds=50)

    async def _execute_critical_tasks(self):
        """Execute time-critical tasks with complete analysis sequence"""
        try:
            start_time = self.get_local_time()
            self.logger.info("Starting critical tasks execution")

            # 1. Fetch latest draw
            print("\nFetching 1 latest draws...")
            draws = await asyncio.get_event_loop().run_in_executor(
                None, self.collector.fetch_latest_draws, 1
            )
            if not draws:
                raise Exception("Failed to fetch latest draw")
            
            # 2. Validate and save draw
            if draws[0]:
                # First validate the draw data
                if await asyncio.get_event_loop().run_in_executor(
                    None, self.collector.validate_draw_data, draws[0][1]
                ):
                    # Then save the draw
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.collector.save_draw, draws[0][0], draws[0][1]
                    )
                    print("\nSuccessfully collected 1 draws")
                    
                    # Sort historical draws
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.collector.sort_historical_draws
                    )

            # 3. Complete Analysis Sequence
            print("\nStarting comprehensive data analysis...")
            try:
                # Create DataAnalysis instance
                analysis = DataAnalysis(draws)
                
                # 3.1 Frequency Analysis
                print("Performing frequency analysis...")
                frequency = analysis.count_frequency()
                top_numbers = analysis.get_top_numbers(20)
                print(f"Top 20 numbers: {', '.join(map(str, top_numbers))}")
                
                # 3.2 Pattern Analysis
                print("Analyzing number patterns...")
                common_pairs = analysis.find_common_pairs()
                consecutive_pairs = analysis.find_consecutive_numbers()
                sequence_patterns = analysis.sequence_pattern_analysis()
                
                # 3.3 Range Analysis
                print("Analyzing number ranges...")
                range_distribution = analysis.number_range_analysis()
                
                # 3.4 Hot and Cold Numbers
                print("Identifying hot and cold numbers...")
                hot_numbers, cold_numbers = analysis.hot_and_cold_numbers()
                print(f"Hot numbers: {[num for num, _ in hot_numbers[:5]]}")
                print(f"Cold numbers: {[num for num, _ in cold_numbers[:5]]}")
                
                # 3.5 Cluster Analysis
                print("Performing cluster analysis...")
                clusters = analysis.cluster_analysis()
                
                # 3.6 Save All Analysis Results
                print("Saving complete analysis results...")
                save_success = analysis.save_to_excel()
                if not save_success:
                    raise Exception("Failed to save analysis results")
                
                print("Complete analysis finished and saved successfully")
                self.execution_stats['last_analysis_time'] = self.get_local_time()
                
            except Exception as e:
                print(f"Error during analysis: {e}")
                raise
            
            # 4. ML Prediction using analyzed data
            predictions, probabilities, analysis_results = await asyncio.get_event_loop().run_in_executor(
                None, self.handler.handle_prediction_pipeline
            )
            
            if predictions:
                next_draw_time = self.get_local_time() + timedelta(minutes=5)
                await asyncio.get_event_loop().run_in_executor(
                    None, self.handler.save_predictions_to_csv,
                    predictions, probabilities, next_draw_time.strftime('%Y-%m-%d %H:%M:%S')
                )
                print(f"\nPredicted numbers: {sorted(predictions)}")
            
            # 5. Run continuous learning cycle
            print("\nStarting continuous learning cycle...")
            await asyncio.get_event_loop().run_in_executor(
                None, self.handler.run_continuous_learning_cycle
            )

            execution_time = (self.get_local_time() - start_time).total_seconds()
            self.execution_stats['last_critical_execution_time'] = execution_time
            self.logger.info(f"Critical tasks completed in {execution_time:.2f} seconds")
            
            return True, draws

        except Exception as e:
            self.logger.error(f"Error in critical tasks: {str(e)}")
            return False, None

    async def _execute_background_tasks(self, draws):
        """Execute non-time-critical background tasks"""
        try:
            self.logger.info("Starting background tasks")
            
            tasks = [
                self.thread_pool.submit(self.evaluator.evaluate_past_predictions)
            ]
            
            for future in tasks:
                await asyncio.get_event_loop().run_in_executor(None, future.result)
            
            self.logger.info("Background tasks completed")
            return True

        except Exception as e:
            self.logger.error(f"Error in background tasks: {str(e)}")
            return False

    def _update_stats(self, success, execution_time):
        """Update execution statistics"""
        self.execution_stats['cycles_completed'] += 1
        if success:
            self.execution_stats['successful_cycles'] += 1
            self.execution_stats['average_execution_time'] = (
                (self.execution_stats['average_execution_time'] * 
                 (self.execution_stats['successful_cycles'] - 1) + 
                 execution_time) / self.execution_stats['successful_cycles']
            )
        else:
            self.execution_stats['failed_cycles'] += 1

    async def run_prediction_cycle(self):
        """Execute one complete prediction cycle"""
        try:
            cycle_start = self.get_local_time()
            self.logger.info(f"Starting prediction cycle at {cycle_start.strftime('%Y-%m-%d %H:%M:%S %Z')}")

            # Execute critical tasks first
            critical_success, draws = await self._execute_critical_tasks()
            if not critical_success:
                raise Exception("Critical tasks failed")

            # Execute background tasks if time permits
            if not self._is_critical_window():
                await self._execute_background_tasks(draws)

            # Calculate execution time and update stats
            execution_time = (self.get_local_time() - cycle_start).total_seconds()
            self._update_stats(True, execution_time)
            
            self.logger.info(f"Cycle completed in {execution_time:.2f} seconds")
            return True

        except Exception as e:
            self.logger.error(f"Error in prediction cycle: {str(e)}")
            self._update_stats(False, 0)
            return False

    def print_stats(self):
        """Print current execution statistics"""
        print("\nExecution Statistics:")
        print(f"Cycles completed: {self.execution_stats['cycles_completed']}")
        print(f"Successful cycles: {self.execution_stats['successful_cycles']}")
        print(f"Failed cycles: {self.execution_stats['failed_cycles']}")
        print(f"Average execution time: {self.execution_stats['average_execution_time']:.2f}s")
        print(f"Last critical execution time: {self.execution_stats['last_critical_execution_time']:.2f}s")
        if self.execution_stats['last_analysis_time']:
            print(f"Last analysis time: {self.execution_stats['last_analysis_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}")

    async def run(self):
        """Main automation loop"""
        self.logger.info("Starting Lottery Automation System")
        print("\nStarting Lottery Automation System...")
        print(f"Current local time: {self.get_local_time().strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print("Waiting for next execution window...")

        while True:
            try:
                local_time = self.get_local_time()
                
                if self._is_critical_window():
                    if self.last_execution_time is None or \
                       (local_time - self.last_execution_time).total_seconds() >= 300:
                        
                        await self.run_prediction_cycle()
                        self.last_execution_time = local_time
                        
                        self.print_stats()
                        
                        next_execution = local_time + timedelta(minutes=5)
                        next_execution = next_execution.replace(second=50)
                        wait_time = (next_execution - self.get_local_time()).total_seconds()
                        
                        print(f"\nNext execution at: {next_execution.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                        print(f"Waiting {wait_time:.0f} seconds...")

                await asyncio.sleep(1)

            except KeyboardInterrupt:
                self.logger.info("Stopping automation")
                print("\nStopping automation...")
                break
            except Exception as e:
                self.logger.error(f"Error in automation loop: {str(e)}")
                await asyncio.sleep(60)

    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown()
        self.logger.info("Automation system shutdown complete")

def test_mode():
    """Run system in test mode"""
    automation = LotteryAutomation()
    try:
        print("\nRunning test cycle...")
        asyncio.run(automation.run_prediction_cycle())
        
        local_time = automation.get_local_time()
        print(f"\nCurrent local time: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Next window: {(automation._calculate_next_window()).total_seconds():.0f} seconds")
        
        automation.print_stats()
        
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        automation.cleanup()

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_mode()
    else:
        automation = LotteryAutomation()
        try:
            asyncio.run(automation.run())
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            automation.cleanup()

if __name__ == "__main__":
    main()