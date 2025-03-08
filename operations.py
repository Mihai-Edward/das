import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.paths import PATHS, ensure_directories
from lottery_predictor import LotteryPredictor
from data_collector_selenium import KinoDataCollector
from data_analysis import DataAnalysis
from prediction_evaluator import PredictionEvaluator

class OperationResult:
    """Standardized operation result container"""
    def __init__(self, success: bool, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error
        self.timestamp = datetime.now()

class Operations:
    """
    Pure operations handler for the lottery prediction system.
    No timing or state management logic - only operations.
    """
    def __init__(self):
        # Initialize logging
        self.logger = logging.getLogger('Operations')
        self._setup_logging()
        
        # Ensure directories exist
        ensure_directories()
        
        # Initialize operation trackers
        self.operation_metrics = {
            'fetch_count': 0,
            'analysis_count': 0,
            'prediction_count': 0,
            'evaluation_count': 0,
            'learning_count': 0
        }
        
        # Cache for operation results
        self.latest_results = {
            'fetch': None,
            'analysis': None,
            'prediction': None,
            'evaluation': None
        }

    def _setup_logging(self):
        """Configure logging"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def fetch_latest_draws(self, num_draws: int = 5) -> OperationResult:
        """Fetch latest draw data from the website"""
        try:
            self.logger.info(f"Fetching {num_draws} latest draws")
            collector = KinoDataCollector()
            draws = collector.fetch_latest_draws(num_draws=num_draws)
            
            if not draws:
                return OperationResult(False, error="No draws collected")
                
            # Validate collected data
            validated_draws = []
            for draw_date, numbers in draws:
                if len(numbers) == 20 and all(1 <= n <= 80 for n in numbers):
                    validated_draws.append((draw_date, sorted(numbers)))
                    
            if not validated_draws:
                return OperationResult(False, error="No valid draws after validation")
                
            self.operation_metrics['fetch_count'] += 1
            self.latest_results['fetch'] = validated_draws
            
            return OperationResult(True, validated_draws)
            
        except Exception as e:
            self.logger.error(f"Fetch operation failed: {str(e)}")
            return OperationResult(False, error=str(e))

    def analyze_draws(self, draws: List[Tuple[str, List[int]]]) -> OperationResult:
        """Analyze collected draw data"""
        try:
            self.logger.info("Analyzing draw data")
            
            if not draws:
                return OperationResult(False, error="No draws provided for analysis")
                
            analyzer = DataAnalysis(draws)
            
            analysis_results = {
                'frequency': analyzer.count_frequency(),
                'top_numbers': analyzer.get_top_numbers(20),
                'common_pairs': analyzer.find_common_pairs(),
                'consecutive_numbers': analyzer.find_consecutive_numbers(),
                'range_analysis': analyzer.number_range_analysis(),
                'hot_cold': analyzer.hot_and_cold_numbers()
            }
            
            # Save analysis results
            analyzer.save_to_excel()
            
            self.operation_metrics['analysis_count'] += 1
            self.latest_results['analysis'] = analysis_results
            
            return OperationResult(True, analysis_results)
            
        except Exception as e:
            self.logger.error(f"Analysis operation failed: {str(e)}")
            return OperationResult(False, error=str(e))

    def generate_prediction(self, historical_data: Optional[pd.DataFrame] = None) -> OperationResult:
        """Generate predictions for next draw"""
        try:
            self.logger.info("Generating prediction")
            predictor = LotteryPredictor()
            
            # Load data if not provided
            if historical_data is None:
                historical_data = predictor.load_data()
                
            if historical_data is None:
                return OperationResult(False, error="Failed to load historical data")
                
            # Generate prediction
            predictions, probabilities, analysis = predictor.train_and_predict(historical_data)
            
            if predictions is None:
                return OperationResult(False, error="Failed to generate predictions")
                
            result = {
                'predictions': sorted(predictions),
                'probabilities': probabilities,
                'analysis_context': analysis
            }
            
            # Save prediction
            predictor.save_prediction_to_csv(predictions, probabilities)
            
            self.operation_metrics['prediction_count'] += 1
            self.latest_results['prediction'] = result
            
            return OperationResult(True, result)
            
        except Exception as e:
            self.logger.error(f"Prediction operation failed: {str(e)}")
            return OperationResult(False, error=str(e))

    def evaluate_predictions(self) -> OperationResult:
        """Evaluate prediction accuracy"""
        try:
            self.logger.info("Evaluating predictions")
            evaluator = PredictionEvaluator()
            
            evaluator.evaluate_past_predictions()
            stats = evaluator.get_performance_stats()
            
            if stats is None:
                return OperationResult(False, error="No evaluation statistics available")
                
            self.operation_metrics['evaluation_count'] += 1
            self.latest_results['evaluation'] = stats
            
            return OperationResult(True, stats)
            
        except Exception as e:
            self.logger.error(f"Evaluation operation failed: {str(e)}")
            return OperationResult(False, error=str(e))

    def run_continuous_learning(self) -> OperationResult:
        """Execute continuous learning cycle"""
        try:
            self.logger.info("Running continuous learning cycle")
            
            # Get evaluation stats first
            eval_result = self.evaluate_predictions()
            if not eval_result.success:
                return OperationResult(False, error="Cannot learn without evaluation data")
                
            # Load predictor and update models
            predictor = LotteryPredictor()
            
            # Use evaluation results to improve model
            learning_results = predictor.apply_learning_from_evaluations(eval_result.data)
            
            if learning_results:
                self.operation_metrics['learning_count'] += 1
                return OperationResult(True, learning_results)
            else:
                return OperationResult(False, error="Learning cycle failed")
                
        except Exception as e:
            self.logger.error(f"Learning operation failed: {str(e)}")
            return OperationResult(False, error=str(e))

    def get_operation_metrics(self) -> Dict[str, Any]:
        """Get metrics for all operations"""
        return {
            'metrics': self.operation_metrics,
            'latest_results': {
                k: (v.timestamp if v else None) 
                for k, v in self.latest_results.items()
            }
        }

    def validate_system_state(self) -> OperationResult:
        """Validate overall system state and data integrity"""
        try:
            validation_results = {
                'files_exist': True,
                'data_valid': True,
                'models_ready': True,
                'issues': []
            }
            
            # Check required files
            required_files = [
                PATHS['HISTORICAL_DATA'],
                PATHS['PREDICTIONS'],
                PATHS['ANALYSIS']
            ]
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    validation_results['files_exist'] = False
                    validation_results['issues'].append(f"Missing file: {file_path}")
            
            # Validate predictor state
            predictor = LotteryPredictor()
            is_valid, message = predictor.validate_model_state()
            
            if not is_valid:
                validation_results['models_ready'] = False
                validation_results['issues'].append(f"Model validation failed: {message}")
            
            return OperationResult(True, validation_results)
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return OperationResult(False, error=str(e))

if __name__ == "__main__":
    # Test operations
    ops = Operations()
    
    print("\nTesting operations...")
    
    # Test fetch
    fetch_result = ops.fetch_latest_draws()
    print(f"\nFetch success: {fetch_result.success}")
    if fetch_result.success:
        # Test analysis
        analysis_result = ops.analyze_draws(fetch_result.data)
        print(f"Analysis success: {analysis_result.success}")
        
        # Test prediction
        prediction_result = ops.generate_prediction()
        print(f"Prediction success: {prediction_result.success}")
        
        # Test evaluation
        eval_result = ops.evaluate_predictions()
        print(f"Evaluation success: {eval_result.success}")
    
    # Get metrics
    metrics = ops.get_operation_metrics()
    print("\nOperation metrics:", metrics)