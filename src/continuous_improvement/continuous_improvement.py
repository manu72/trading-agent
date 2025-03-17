"""
Continuous Improvement Mechanism - Core Module

This module implements the Continuous Improvement Mechanism component of the DeepResearch 4.5 Trading System.
It enables the trading system to learn from its performance and adapt over time.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import threading
import queue
import json
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import joblib

# Import from other components
from ..knowledge_base import get_knowledge_base
from ..ai_trading_agent import get_ai_trading_agent
from ..algorithmic_trading import get_algorithmic_trading_system
from ..data_analysis import get_data_analysis_system

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("continuous_improvement.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ContinuousImprovement")

class ContinuousImprovementSystem:
    """
    Main class for the Continuous Improvement Mechanism component.
    
    The Continuous Improvement Mechanism enables the trading system to learn from its
    performance and adapt over time.
    """
    
    def __init__(self, models_dir: str = None, performance_review_interval: int = 86400):
        """
        Initialize the Continuous Improvement Mechanism component.
        
        Args:
            models_dir: Directory to store trained models
            performance_review_interval: Interval in seconds for performance reviews
        """
        self.kb = get_knowledge_base()
        self.ai_agent = get_ai_trading_agent()
        self.trading_system = get_algorithmic_trading_system()
        self.data_analysis = get_data_analysis_system()
        
        # Set models directory
        if models_dir is None:
            self.models_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'models'
        else:
            self.models_dir = Path(models_dir)
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(exist_ok=True)
        
        # Set performance review interval
        self.performance_review_interval = performance_review_interval
        
        # Initialize learning models
        self.learning_models = {
            'signal_quality': SignalQualityModel(self.models_dir),
            'position_sizing': PositionSizingModel(self.models_dir),
            'exit_timing': ExitTimingModel(self.models_dir),
            'market_regime': MarketRegimeModel(self.models_dir),
            'strategy_selection': StrategySelectionModel(self.models_dir)
        }
        
        # Initialize performance metrics
        self.performance_metrics = PerformanceMetrics()
        
        # Initialize strategy optimizer
        self.strategy_optimizer = StrategyOptimizer()
        
        # Initialize parameter tuner
        self.parameter_tuner = ParameterTuner()
        
        # Initialize feedback loop
        self.feedback_loop = FeedbackLoop()
        
        # Initialize performance review queue and thread
        self.review_queue = queue.Queue()
        self.running = True
        self.review_thread = threading.Thread(target=self._process_review_queue)
        self.review_thread.daemon = True
        self.review_thread.start()
        
        # Schedule initial performance review
        self._schedule_performance_review()
        
        logger.info("Continuous Improvement System initialized")
    
    def evaluate_signal_quality(self, signal: Dict[str, Any]) -> float:
        """
        Evaluate the quality of a trading signal.
        
        Args:
            signal: Trading signal to evaluate
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            # Get signal quality model
            model = self.learning_models['signal_quality']
            
            # Prepare signal features
            features = model.prepare_features(signal)
            
            # Predict signal quality
            quality_score = model.predict(features)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error evaluating signal quality: {str(e)}")
            return 0.5  # Default to neutral quality
    
    def optimize_position_size(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> float:
        """
        Optimize the position size for a trading signal.
        
        Args:
            signal: Trading signal to optimize
            account_info: Account information
            
        Returns:
            Optimized position size as percentage of portfolio
        """
        try:
            # Get position sizing model
            model = self.learning_models['position_sizing']
            
            # Prepare features
            features = model.prepare_features(signal, account_info)
            
            # Predict optimal position size
            position_size = model.predict(features)
            
            # Apply constraints
            position_size = max(0.01, min(0.25, position_size))  # Limit to 1% - 25%
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error optimizing position size: {str(e)}")
            return 0.05  # Default to 5% position size
    
    def recommend_exit_timing(self, position: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Recommend exit timing for a position.
        
        Args:
            position: Position to recommend exit for
            market_data: Market data for the position's symbol
            
        Returns:
            Exit recommendation
        """
        try:
            # Get exit timing model
            model = self.learning_models['exit_timing']
            
            # Prepare features
            features = model.prepare_features(position, market_data)
            
            # Predict exit probability and optimal time
            exit_prob, exit_days = model.predict(features)
            
            # Create recommendation
            recommendation = {
                'symbol': position['symbol'],
                'exit_probability': exit_prob,
                'optimal_exit_days': exit_days,
                'timestamp': datetime.now().isoformat()
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error recommending exit timing: {str(e)}")
            return {
                'symbol': position['symbol'],
                'exit_probability': 0.5,
                'optimal_exit_days': 5,
                'timestamp': datetime.now().isoformat()
            }
    
    def detect_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Detect the current market regime.
        
        Args:
            market_data: Market data for multiple symbols
            
        Returns:
            Market regime information
        """
        try:
            # Get market regime model
            model = self.learning_models['market_regime']
            
            # Prepare features
            features = model.prepare_features(market_data)
            
            # Predict market regime
            regime, confidence = model.predict(features)
            
            # Create regime information
            regime_info = {
                'regime': regime,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            return regime_info
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def select_optimal_strategy(self, symbol: str, market_regime: Dict[str, Any]) -> str:
        """
        Select the optimal trading strategy for a symbol.
        
        Args:
            symbol: Symbol to select strategy for
            market_regime: Current market regime
            
        Returns:
            Selected strategy name
        """
        try:
            # Get strategy selection model
            model = self.learning_models['strategy_selection']
            
            # Prepare features
            features = model.prepare_features(symbol, market_regime)
            
            # Predict optimal strategy
            strategy, confidence = model.predict(features)
            
            # Log selection
            logger.info(f"Selected strategy {strategy} for {symbol} with confidence {confidence:.2f}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error selecting optimal strategy: {str(e)}")
            return "momentum"  # Default to momentum strategy
    
    def calculate_performance_metrics(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """
        Calculate performance metrics for the trading system.
        
        Args:
            start_date: Start date for metrics calculation
            end_date: End date for metrics calculation
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=30)
            
            # Get orders and positions from Knowledge Base
            orders = self.kb.get_orders(start_date, end_date)
            positions = self.kb.get_positions(end_date)
            
            # Calculate metrics
            metrics = self.performance_metrics.calculate_metrics(orders, positions, start_date, end_date)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def optimize_strategy_parameters(self, strategy: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize parameters for a trading strategy.
        
        Args:
            strategy: Strategy to optimize
            performance_data: Performance data for the strategy
            
        Returns:
            Optimized parameters
        """
        try:
            # Get current parameters
            current_params = self.ai_agent.get_strategy_parameters(strategy)
            
            # Optimize parameters
            optimized_params = self.strategy_optimizer.optimize(strategy, current_params, performance_data)
            
            # Update strategy parameters
            self.ai_agent.update_strategy_parameters(strategy, optimized_params)
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {str(e)}")
            return {}
    
    def tune_risk_parameters(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tune risk management parameters.
        
        Args:
            performance_data: Performance data
            
        Returns:
            Tuned risk parameters
        """
        try:
            # Get current risk parameters
            current_params = self.trading_system.get_risk_parameters()
            
            # Tune parameters
            tuned_params = self.parameter_tuner.tune_risk_parameters(current_params, performance_data)
            
            # Update risk parameters
            self.trading_system.update_risk_parameters(tuned_params)
            
            return tuned_params
            
        except Exception as e:
            logger.error(f"Error tuning risk parameters: {str(e)}")
            return {}
    
    def process_trade_feedback(self, trade_result: Dict[str, Any]) -> None:
        """
        Process feedback from a completed trade.
        
        Args:
            trade_result: Result of a completed trade
        """
        try:
            # Extract trade information
            symbol = trade_result.get('symbol', '')
            strategy = trade_result.get('strategy', '')
            entry_price = trade_result.get('entry_price', 0.0)
            exit_price = trade_result.get('exit_price', 0.0)
            entry_time = trade_result.get('entry_time', '')
            exit_time = trade_result.get('exit_time', '')
            position_size = trade_result.get('position_size', 0.0)
            pnl = trade_result.get('pnl', 0.0)
            pnl_pct = trade_result.get('pnl_pct', 0.0)
            
            # Calculate trade duration
            try:
                entry_dt = datetime.fromisoformat(entry_time)
                exit_dt = datetime.fromisoformat(exit_time)
                duration_days = (exit_dt - entry_dt).total_seconds() / 86400
            except:
                duration_days = 0
            
            # Create feedback data
            feedback_data = {
                'symbol': symbol,
                'strategy': strategy,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'position_size': position_size,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'duration_days': duration_days,
                'timestamp': datetime.now().isoformat()
            }
            
            # Process feedback
            self.feedback_loop.process_feedback(feedback_data)
            
            # Update learning models
            self._update_learning_models(feedback_data)
            
            logger.info(f"Processed trade feedback for {symbol} with PnL {pnl_pct:.2%}")
            
        except Exception as e:
            logger.error(f"Error processing trade feedback: {str(e)}")
    
    def perform_performance_review(self) -> Dict[str, Any]:
        """
        Perform a comprehensive performance review of the trading system.
        
        Returns:
            Performance review results
        """
        try:
            # Calculate performance metrics
            daily_metrics = self.calculate_performance_metrics(
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
            
            weekly_metrics = self.calculate_performance_metrics(
                start_date=datetime.now() - timedelta(days=7),
                end_date=datetime.now()
            )
            
            monthly_metrics = self.calculate_performance_metrics(
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now()
            )
            
            # Analyze strategy performance
            strategy_performance = self._analyze_strategy_performance()
            
            # Analyze risk management
            risk_analysis = self._analyze_risk_management()
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                daily_metrics, weekly_metrics, monthly_metrics,
                strategy_performance, risk_analysis
            )
            
            # Create review results
            review_results = {
                'timestamp': datetime.now().isoformat(),
                'daily_metrics': daily_metrics,
                'weekly_metrics': weekly_metrics,
                'monthly_metrics': monthly_metrics,
                'strategy_performance': strategy_performance,
                'risk_analysis': risk_analysis,
                'recommendations': recommendations
            }
            
            # Store review in Knowledge Base
            self.kb.store_performance_review(review_results)
            
            # Apply recommendations
            self._apply_recommendations(recommendations)
            
            # Schedule next review
            self._schedule_performance_review()
            
            return review_results
            
        except Exception as e:
            logger.error(f"Error performing performance review: {str(e)}")
            return {}
    
    def _analyze_strategy_performance(self) -> Dict[str, Any]:
        """
        Analyze the performance of trading strategies.
        
        Returns:
            Strategy performance analysis
        """
        try:
            # Get strategy performance data
            strategies = ['momentum', 'sector_rotation', 'swing_trading', 'mean_reversion']
            
            performance_data = {}
            
            for strategy in strategies:
                # Get strategy trades
                trades = self.kb.get_trades_by_strategy(
                    strategy=strategy,
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now()
                )
                
                # Calculate strategy metrics
                if trades:
                    win_count = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
                    loss_count = sum(1 for trade in trades if trade.get('pnl', 0) <= 0)
                    total_count = len(trades)
                    
                    win_rate = win_count / total_count if total_count > 0 else 0
                    
                    pnl_values = [trade.get('pnl', 0) for trade in trades]
                    total_pnl = sum(pnl_values)
                    avg_pnl = total_pnl / total_count if total_count > 0 else 0
                    
                    win_values = [trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0]
                    loss_values = [trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) <= 0]
                    
                    avg_win = sum(win_values) / len(win_values) if win_values else 0
                    avg_loss = sum(loss_values) / len(loss_values) if loss_values else 0
                    
                    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                    
                    performance_data[strategy] = {
                        'trade_count': total_count,
                        'win_count': win_count,
                        'loss_count': loss_count,
                        'win_rate': win_rate,
                        'total_pnl': total_pnl,
                        'avg_pnl': avg_pnl,
                        'avg_win': avg_win,
                        'avg_loss': avg_loss,
                        'win_loss_ratio': win_loss_ratio
                    }
                else:
                    performance_data[strategy] = {
                        'trade_count': 0,
                        'win_count': 0,
                        'loss_count': 0,
                        'win_rate': 0,
                        'total_pnl': 0,
                        'avg_pnl': 0,
                        'avg_win': 0,
                        'avg_loss': 0,
                        'win_loss_ratio': 0
                    }
            
            # Rank strategies
            ranked_strategies = sorted(
                performance_data.keys(),
                key=lambda s: (
                    performance_data[s]['win_rate'],
                    performance_data[s]['total_pnl'],
                    performance_data[s]['win_loss_ratio']
                ),
                reverse=True
            )
            
            # Create analysis result
            analysis = {
                'strategy_metrics': performance_data,
                'ranked_strategies': ranked_strategies,
                'best_strategy': ranked_strategies[0] if ranked_strategies else None,
                'worst_strategy': ranked_strategies[-1] if ranked_strategies else None
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing strategy performance: {str(e)}")
            return {}
    
    def _analyze_risk_management(self) -> Dict[str, Any]:
        """
        Analyze risk management performance.
        
        Returns:
            Risk management analysis
        """
        try:
            # Get trades
            trades = self.kb.get_trades(
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now()
            )
            
            if not trades:
                return {
                    'max_drawdown': 0,
                    'avg_position_size': 0,
                    'avg_holding_period': 0,
                    'stop_loss_hit_rate': 0,
                    'risk_reward_ratio': 0
                }
            
            # Calculate risk metrics
            pnl_values = [trade.get('pnl_pct', 0) for trade in trades]
            cumulative_returns = np.cumsum(pnl_values)
            
            # Max drawdown
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / (1 + peak)
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            # Average position size
            position_sizes = [trade.get('position_size', 0) for trade in trades]
            avg_position_size = np.mean(position_sizes) if position_sizes else 0
            
            # Average holding period
            holding_periods = []
            for trade in trades:
                try:
                    entry_time = datetime.fromisoformat(trade.get('entry_time', ''))
                    exit_time = datetime.fromisoformat(trade.get('exit_time', ''))
                    holding_period = (exit_time - entry_time).total_seconds() / 86400  # in days
                    holding_periods.append(holding_period)
                except:
                    pass
            
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0
            
            # Stop loss hit rate
            stop_loss_hits = sum(1 for trade in trades if trade.get('exit_reason', '') == 'stop_loss')
            stop_loss_hit_rate = stop_loss_hits / len(trades) if trades else 0
            
            # Risk-reward ratio
            win_values = [trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0]
            loss_values = [abs(trade.get('pnl', 0)) for trade in trades if trade.get('pnl', 0) < 0]
            
            avg_win = np.mean(win_values) if win_values else 0
            avg_loss = np.mean(loss_values) if loss_values else 0
            
            risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            # Create analysis result
            analysis = {
                'max_drawdown': max_drawdown,
                'avg_position_size': avg_position_size,
                'avg_holding_period': avg_holding_period,
                'stop_loss_hit_rate': stop_loss_hit_rate,
                'risk_reward_ratio': risk_reward_ratio
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing risk management: {str(e)}")
            return {}
    
    def _generate_recommendations(self, daily_metrics: Dict[str, Any], weekly_metrics: Dict[str, Any],
                                 monthly_metrics: Dict[str, Any], strategy_performance: Dict[str, Any],
                                 risk_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on performance analysis.
        
        Args:
            daily_metrics: Daily performance metrics
            weekly_metrics: Weekly performance metrics
            monthly_metrics: Monthly performance metrics
            strategy_performance: Strategy performance analysis
            risk_analysis: Risk management analysis
            
        Returns:
            List of recommendations
        """
        try:
            recommendations = []
            
            # Check if any metrics are available
            if not daily_metrics or not weekly_metrics or not monthly_metrics:
                return recommendations
            
            # Check for declining performance
            if (daily_metrics.get('total_return', 0) < 0 and
                weekly_metrics.get('total_return', 0) < 0 and
                monthly_metrics.get('total_return', 0) < 0):
                
                recommendations.append({
                    'type': 'alert',
                    'category': 'performance',
                    'message': 'Declining performance across all timeframes',
                    'action': 'reduce_exposure',
                    'priority': 'high'
                })
            
            # Check for high drawdown
            if risk_analysis.get('max_drawdown', 0) > 0.1:  # 10% drawdown
                recommendations.append({
                    'type': 'alert',
                    'category': 'risk',
                    'message': f"High drawdown detected: {risk_analysis.get('max_drawdown', 0):.2%}",
                    'action': 'reduce_position_size',
                    'priority': 'high'
                })
            
            # Check for poor strategy performance
            for strategy, metrics in strategy_performance.get('strategy_metrics', {}).items():
                if metrics.get('trade_count', 0) >= 5 and metrics.get('win_rate', 0) < 0.4:
                    recommendations.append({
                        'type': 'alert',
                        'category': 'strategy',
                        'message': f"Poor performance for {strategy} strategy: {metrics.get('win_rate', 0):.2%} win rate",
                        'action': 'optimize_strategy',
                        'strategy': strategy,
                        'priority': 'medium'
                    })
            
            # Check for low risk-reward ratio
            if risk_analysis.get('risk_reward_ratio', 0) < 1.5:
                recommendations.append({
                    'type': 'alert',
                    'category': 'risk',
                    'message': f"Low risk-reward ratio: {risk_analysis.get('risk_reward_ratio', 0):.2f}",
                    'action': 'adjust_exit_criteria',
                    'priority': 'medium'
                })
            
            # Check for high stop loss hit rate
            if risk_analysis.get('stop_loss_hit_rate', 0) > 0.3:  # 30% of trades hit stop loss
                recommendations.append({
                    'type': 'alert',
                    'category': 'risk',
                    'message': f"High stop loss hit rate: {risk_analysis.get('stop_loss_hit_rate', 0):.2%}",
                    'action': 'widen_stop_loss',
                    'priority': 'medium'
                })
            
            # Check for best performing strategy
            best_strategy = strategy_performance.get('best_strategy')
            if best_strategy:
                best_metrics = strategy_performance.get('strategy_metrics', {}).get(best_strategy, {})
                if best_metrics.get('trade_count', 0) >= 5 and best_metrics.get('win_rate', 0) > 0.6:
                    recommendations.append({
                        'type': 'opportunity',
                        'category': 'strategy',
                        'message': f"Strong performance for {best_strategy} strategy: {best_metrics.get('win_rate', 0):.2%} win rate",
                        'action': 'increase_allocation',
                        'strategy': best_strategy,
                        'priority': 'medium'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def _apply_recommendations(self, recommendations: List[Dict[str, Any]]) -> None:
        """
        Apply recommendations to the trading system.
        
        Args:
            recommendations: List of recommendations
        """
        try:
            for recommendation in recommendations:
                action = recommendation.get('action', '')
                priority = recommendation.get('priority', 'low')
                
                logger.info(f"Applying recommendation: {action} (priority: {priority})")
                
                if action == 'reduce_exposure':
                    # Reduce overall exposure
                    self.trading_system.update_max_exposure(0.5)  # 50% of normal exposure
                
                elif action == 'reduce_position_size':
                    # Reduce position sizes
                    current_params = self.trading_system.get_risk_parameters()
                    current_size = current_params.get('max_position_size_pct', 0.05)
                    new_size = max(0.01, current_size * 0.7)  # Reduce by 30%
                    
                    self.trading_system.update_risk_parameters({
                        'max_position_size_pct': new_size
                    })
                
                elif action == 'optimize_strategy':
                    # Optimize strategy parameters
                    strategy = recommendation.get('strategy', '')
                    if strategy:
                        performance_data = self.kb.get_strategy_performance(strategy)
                        self.optimize_strategy_parameters(strategy, performance_data)
                
                elif action == 'adjust_exit_criteria':
                    # Adjust exit criteria to improve risk-reward ratio
                    self.ai_agent.update_exit_criteria({
                        'profit_target_multiplier': 2.0,  # Increase profit targets
                        'trailing_stop_activation': 0.5  # Activate trailing stops earlier
                    })
                
                elif action == 'widen_stop_loss':
                    # Widen stop loss to reduce premature exits
                    self.ai_agent.update_exit_criteria({
                        'stop_loss_multiplier': 1.5  # Increase stop loss distance
                    })
                
                elif action == 'increase_allocation':
                    # Increase allocation to well-performing strategy
                    strategy = recommendation.get('strategy', '')
                    if strategy:
                        self.ai_agent.update_strategy_allocation(strategy, 1.5)  # Increase by 50%
                
        except Exception as e:
            logger.error(f"Error applying recommendations: {str(e)}")
    
    def _update_learning_models(self, feedback_data: Dict[str, Any]) -> None:
        """
        Update learning models with new feedback data.
        
        Args:
            feedback_data: Feedback data from a completed trade
        """
        try:
            # Update signal quality model
            self.learning_models['signal_quality'].update(feedback_data)
            
            # Update position sizing model
            self.learning_models['position_sizing'].update(feedback_data)
            
            # Update exit timing model
            self.learning_models['exit_timing'].update(feedback_data)
            
            # Update market regime model
            self.learning_models['market_regime'].update(feedback_data)
            
            # Update strategy selection model
            self.learning_models['strategy_selection'].update(feedback_data)
            
        except Exception as e:
            logger.error(f"Error updating learning models: {str(e)}")
    
    def _schedule_performance_review(self) -> None:
        """Schedule a performance review."""
        try:
            # Create review task
            review_task = {
                'type': 'performance_review',
                'scheduled_time': datetime.now() + timedelta(seconds=self.performance_review_interval)
            }
            
            # Add to review queue
            self.review_queue.put(review_task)
            
            logger.info(f"Scheduled performance review for {review_task['scheduled_time']}")
            
        except Exception as e:
            logger.error(f"Error scheduling performance review: {str(e)}")
    
    def _process_review_queue(self) -> None:
        """Process tasks in the review queue."""
        while self.running:
            try:
                # Get task from queue with timeout
                try:
                    task = self.review_queue.get(timeout=1)
                except queue.Empty:
                    # Check if any scheduled tasks are due
                    continue
                
                # Check if task is due
                if task.get('scheduled_time', datetime.now()) <= datetime.now():
                    # Process task
                    if task.get('type') == 'performance_review':
                        self.perform_performance_review()
                else:
                    # Task is not due yet, put it back in the queue
                    self.review_queue.put(task)
                
                # Sleep to prevent high CPU usage
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in review processing thread: {str(e)}")
                time.sleep(1)
    
    def shutdown(self) -> None:
        """Shutdown the Continuous Improvement Mechanism component."""
        try:
            logger.info("Shutting down Continuous Improvement Mechanism component")
            
            # Stop processing thread
            self.running = False
            if self.review_thread.is_alive():
                self.review_thread.join(timeout=5)
            
            # Save learning models
            for name, model in self.learning_models.items():
                model.save()
            
            logger.info("Continuous Improvement Mechanism component shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")


class LearningModel:
    """Base class for learning models."""
    
    def __init__(self, models_dir: Path, model_name: str):
        """
        Initialize the learning model.
        
        Args:
            models_dir: Directory to store trained models
            model_name: Name of the model
        """
        self.models_dir = models_dir
        self.model_name = model_name
        self.model_path = models_dir / f"{model_name}.joblib"
        self.data_path = models_dir / f"{model_name}_data.pkl"
        self.model = None
        self.scaler = None
        self.training_data = []
        
        # Load existing model and data if available
        self.load()
    
    def prepare_features(self, *args, **kwargs) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            Feature array
        """
        raise NotImplementedError("Subclasses must implement prepare_features()")
    
    def predict(self, features: np.ndarray) -> Any:
        """
        Make a prediction using the model.
        
        Args:
            features: Feature array
            
        Returns:
            Prediction
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def update(self, feedback_data: Dict[str, Any]) -> None:
        """
        Update the model with new feedback data.
        
        Args:
            feedback_data: Feedback data
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def train(self) -> None:
        """Train the model on collected data."""
        raise NotImplementedError("Subclasses must implement train()")
    
    def save(self) -> None:
        """Save the model and training data."""
        try:
            if self.model is not None:
                joblib.dump((self.model, self.scaler), self.model_path)
            
            with open(self.data_path, 'wb') as f:
                pickle.dump(self.training_data, f)
                
            logger.info(f"Saved model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error saving model {self.model_name}: {str(e)}")
    
    def load(self) -> None:
        """Load the model and training data."""
        try:
            if self.model_path.exists():
                self.model, self.scaler = joblib.load(self.model_path)
                logger.info(f"Loaded model {self.model_name}")
            
            if self.data_path.exists():
                with open(self.data_path, 'rb') as f:
                    self.training_data = pickle.load(f)
                logger.info(f"Loaded training data for {self.model_name}")
                
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            # Initialize new model and data
            self.model = None
            self.scaler = StandardScaler()
            self.training_data = []


class SignalQualityModel(LearningModel):
    """Model for evaluating signal quality."""
    
    def __init__(self, models_dir: Path):
        """
        Initialize the signal quality model.
        
        Args:
            models_dir: Directory to store trained models
        """
        super().__init__(models_dir, "signal_quality")
        
        # Initialize model if not loaded
        if self.model is None:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
    
    def prepare_features(self, signal: Dict[str, Any]) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Args:
            signal: Trading signal
            
        Returns:
            Feature array
        """
        try:
            # Extract features from signal
            features = [
                signal.get('strength', 0.5),
                1 if signal.get('direction', '') == 'BUY' else 0,
                self._encode_strategy(signal.get('strategy', '')),
                self._encode_timeframe(signal.get('timeframe', '')),
                signal.get('metadata', {}).get('rsi', 50) / 100,
                signal.get('metadata', {}).get('momentum_score', 0),
                signal.get('metadata', {}).get('volatility', 0.2),
                signal.get('metadata', {}).get('volume_ratio', 1.0)
            ]
            
            # Convert to numpy array
            features = np.array(features).reshape(1, -1)
            
            # Scale features if scaler is fitted
            if hasattr(self.scaler, 'mean_'):
                features = self.scaler.transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features for signal quality model: {str(e)}")
            return np.zeros((1, 8))
    
    def predict(self, features: np.ndarray) -> float:
        """
        Predict signal quality.
        
        Args:
            features: Feature array
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            # If model is not trained, return default quality
            if len(self.training_data) < 10 or not hasattr(self.model, 'classes_'):
                return 0.5
            
            # Predict probability of positive outcome
            proba = self.model.predict_proba(features)[0]
            
            # Return probability of positive class
            return proba[1]
            
        except Exception as e:
            logger.error(f"Error predicting signal quality: {str(e)}")
            return 0.5
    
    def update(self, feedback_data: Dict[str, Any]) -> None:
        """
        Update the model with new feedback data.
        
        Args:
            feedback_data: Feedback data from a completed trade
        """
        try:
            # Extract signal information
            signal = {
                'symbol': feedback_data.get('symbol', ''),
                'strategy': feedback_data.get('strategy', ''),
                'direction': 'BUY' if feedback_data.get('position_size', 0) > 0 else 'SELL',
                'strength': 0.5,  # Default
                'timeframe': 'daily',  # Default
                'metadata': {}
            }
            
            # Get additional signal data from Knowledge Base if available
            kb_signal = self.kb.get_signal_by_id(feedback_data.get('signal_id', ''))
            if kb_signal:
                signal.update(kb_signal)
            
            # Prepare features
            features = self.prepare_features(signal)
            
            # Determine label (1 for profitable trade, 0 for losing trade)
            label = 1 if feedback_data.get('pnl', 0) > 0 else 0
            
            # Add to training data
            self.training_data.append((features[0], label))
            
            # Train model if enough data is available
            if len(self.training_data) >= 10 and len(self.training_data) % 5 == 0:
                self.train()
                
        except Exception as e:
            logger.error(f"Error updating signal quality model: {str(e)}")
    
    def train(self) -> None:
        """Train the model on collected data."""
        try:
            # Extract features and labels
            X = np.array([data[0] for data in self.training_data])
            y = np.array([data[1] for data in self.training_data])
            
            # Fit scaler
            self.scaler.fit(X)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Evaluate model
            if len(X) >= 20:
                cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
                logger.info(f"Signal quality model cross-validation accuracy: {np.mean(cv_scores):.4f}")
                
            logger.info(f"Trained signal quality model on {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training signal quality model: {str(e)}")
    
    def _encode_strategy(self, strategy: str) -> float:
        """
        Encode strategy as a numeric value.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Encoded value
        """
        strategy_map = {
            'momentum': 0.0,
            'sector_rotation': 0.25,
            'swing_trading': 0.5,
            'mean_reversion': 0.75,
            'trend_following': 1.0
        }
        
        return strategy_map.get(strategy.lower(), 0.0)
    
    def _encode_timeframe(self, timeframe: str) -> float:
        """
        Encode timeframe as a numeric value.
        
        Args:
            timeframe: Timeframe name
            
        Returns:
            Encoded value
        """
        timeframe_map = {
            'intraday': 0.0,
            'daily': 0.25,
            'weekly': 0.5,
            'monthly': 0.75,
            'quarterly': 1.0
        }
        
        return timeframe_map.get(timeframe.lower(), 0.25)


class PositionSizingModel(LearningModel):
    """Model for optimizing position sizing."""
    
    def __init__(self, models_dir: Path):
        """
        Initialize the position sizing model.
        
        Args:
            models_dir: Directory to store trained models
        """
        super().__init__(models_dir, "position_sizing")
        
        # Initialize model if not loaded
        if self.model is None:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
    
    def prepare_features(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Args:
            signal: Trading signal
            account_info: Account information
            
        Returns:
            Feature array
        """
        try:
            # Extract features from signal and account info
            features = [
                signal.get('strength', 0.5),
                1 if signal.get('direction', '') == 'BUY' else 0,
                self._encode_strategy(signal.get('strategy', '')),
                self._encode_timeframe(signal.get('timeframe', '')),
                signal.get('metadata', {}).get('rsi', 50) / 100,
                signal.get('metadata', {}).get('momentum_score', 0),
                signal.get('metadata', {}).get('volatility', 0.2),
                signal.get('metadata', {}).get('volume_ratio', 1.0),
                account_info.get('net_liquidation_value', 100000) / 1000000,  # Normalize
                account_info.get('buying_power', 200000) / 1000000,  # Normalize
                account_info.get('cash_balance', 50000) / 1000000,  # Normalize
                account_info.get('day_trades_remaining', 3) / 3,  # Normalize
                account_info.get('leverage', 1.0)
            ]
            
            # Convert to numpy array
            features = np.array(features).reshape(1, -1)
            
            # Scale features if scaler is fitted
            if hasattr(self.scaler, 'mean_'):
                features = self.scaler.transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features for position sizing model: {str(e)}")
            return np.zeros((1, 13))
    
    def predict(self, features: np.ndarray) -> float:
        """
        Predict optimal position size.
        
        Args:
            features: Feature array
            
        Returns:
            Position size as percentage of portfolio (0.01 to 0.25)
        """
        try:
            # If model is not trained, return default position size
            if len(self.training_data) < 10 or not hasattr(self.model, 'feature_importances_'):
                return 0.05
            
            # Predict position size
            position_size = self.model.predict(features)[0]
            
            # Clip to valid range
            position_size = max(0.01, min(0.25, position_size))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error predicting position size: {str(e)}")
            return 0.05
    
    def update(self, feedback_data: Dict[str, Any]) -> None:
        """
        Update the model with new feedback data.
        
        Args:
            feedback_data: Feedback data from a completed trade
        """
        try:
            # Extract signal and account information
            signal = {
                'symbol': feedback_data.get('symbol', ''),
                'strategy': feedback_data.get('strategy', ''),
                'direction': 'BUY' if feedback_data.get('position_size', 0) > 0 else 'SELL',
                'strength': 0.5,  # Default
                'timeframe': 'daily',  # Default
                'metadata': {}
            }
            
            account_info = {
                'net_liquidation_value': 100000,  # Default
                'buying_power': 200000,  # Default
                'cash_balance': 50000,  # Default
                'day_trades_remaining': 3,  # Default
                'leverage': 1.0  # Default
            }
            
            # Get additional signal data from Knowledge Base if available
            kb_signal = self.kb.get_signal_by_id(feedback_data.get('signal_id', ''))
            if kb_signal:
                signal.update(kb_signal)
            
            # Get account info from Knowledge Base if available
            kb_account_info = self.kb.get_account_info(feedback_data.get('entry_time', ''))
            if kb_account_info:
                account_info.update(kb_account_info)
            
            # Prepare features
            features = self.prepare_features(signal, account_info)
            
            # Get position size
            position_size = feedback_data.get('position_size', 0.05)
            
            # Adjust position size based on performance
            pnl_pct = feedback_data.get('pnl_pct', 0)
            adjusted_position_size = position_size
            
            if pnl_pct > 0.05:  # Very good trade
                adjusted_position_size = min(0.25, position_size * 1.2)
            elif pnl_pct > 0.02:  # Good trade
                adjusted_position_size = min(0.25, position_size * 1.1)
            elif pnl_pct < -0.05:  # Very bad trade
                adjusted_position_size = max(0.01, position_size * 0.8)
            elif pnl_pct < -0.02:  # Bad trade
                adjusted_position_size = max(0.01, position_size * 0.9)
            
            # Add to training data
            self.training_data.append((features[0], adjusted_position_size))
            
            # Train model if enough data is available
            if len(self.training_data) >= 10 and len(self.training_data) % 5 == 0:
                self.train()
                
        except Exception as e:
            logger.error(f"Error updating position sizing model: {str(e)}")
    
    def train(self) -> None:
        """Train the model on collected data."""
        try:
            # Extract features and labels
            X = np.array([data[0] for data in self.training_data])
            y = np.array([data[1] for data in self.training_data])
            
            # Fit scaler
            self.scaler.fit(X)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Evaluate model
            if len(X) >= 20:
                cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
                rmse = np.sqrt(-np.mean(cv_scores))
                logger.info(f"Position sizing model cross-validation RMSE: {rmse:.4f}")
                
            logger.info(f"Trained position sizing model on {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training position sizing model: {str(e)}")
    
    def _encode_strategy(self, strategy: str) -> float:
        """
        Encode strategy as a numeric value.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Encoded value
        """
        strategy_map = {
            'momentum': 0.0,
            'sector_rotation': 0.25,
            'swing_trading': 0.5,
            'mean_reversion': 0.75,
            'trend_following': 1.0
        }
        
        return strategy_map.get(strategy.lower(), 0.0)
    
    def _encode_timeframe(self, timeframe: str) -> float:
        """
        Encode timeframe as a numeric value.
        
        Args:
            timeframe: Timeframe name
            
        Returns:
            Encoded value
        """
        timeframe_map = {
            'intraday': 0.0,
            'daily': 0.25,
            'weekly': 0.5,
            'monthly': 0.75,
            'quarterly': 1.0
        }
        
        return timeframe_map.get(timeframe.lower(), 0.25)


class ExitTimingModel(LearningModel):
    """Model for optimizing exit timing."""
    
    def __init__(self, models_dir: Path):
        """
        Initialize the exit timing model.
        
        Args:
            models_dir: Directory to store trained models
        """
        super().__init__(models_dir, "exit_timing")
        
        # Initialize models if not loaded
        if self.model is None:
            self.model = {
                'exit_prob': LogisticRegression(random_state=42),
                'exit_days': LinearRegression()
            }
            self.scaler = StandardScaler()
    
    def prepare_features(self, position: Dict[str, Any], market_data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Args:
            position: Position information
            market_data: Market data for the position's symbol
            
        Returns:
            Feature array
        """
        try:
            # Calculate position metrics
            entry_price = position.get('average_cost', 0)
            current_price = market_data['close'].iloc[-1] if not market_data.empty else entry_price
            unrealized_pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            # Calculate technical indicators
            if not market_data.empty and len(market_data) >= 20:
                # RSI
                delta = market_data['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                # Bollinger Bands
                sma_20 = market_data['close'].rolling(window=20).mean()
                std_20 = market_data['close'].rolling(window=20).std()
                upper_band = sma_20 + 2 * std_20
                lower_band = sma_20 - 2 * std_20
                
                bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1]) if (upper_band.iloc[-1] - lower_band.iloc[-1]) > 0 else 0.5
                
                # MACD
                ema_12 = market_data['close'].ewm(span=12, adjust=False).mean()
                ema_26 = market_data['close'].ewm(span=26, adjust=False).mean()
                macd = ema_12 - ema_26
                macd_signal = macd.ewm(span=9, adjust=False).mean()
                macd_hist = macd - macd_signal
                
                current_macd = macd.iloc[-1]
                current_macd_signal = macd_signal.iloc[-1]
                current_macd_hist = macd_hist.iloc[-1]
                
                # Volatility
                atr = market_data['high'] - market_data['low']
                atr_20 = atr.rolling(window=20).mean()
                volatility = atr_20.iloc[-1] / current_price if current_price > 0 else 0
            else:
                current_rsi = 50
                bb_position = 0.5
                current_macd = 0
                current_macd_signal = 0
                current_macd_hist = 0
                volatility = 0.02
            
            # Extract features
            features = [
                unrealized_pnl_pct,
                position.get('days_held', 0) / 30,  # Normalize
                1 if position.get('direction', '') == 'BUY' else 0,
                self._encode_strategy(position.get('strategy', '')),
                current_rsi / 100,  # Normalize
                bb_position,
                current_macd / current_price if current_price > 0 else 0,
                current_macd_hist / current_price if current_price > 0 else 0,
                volatility,
                position.get('size_pct', 0.05) / 0.25  # Normalize
            ]
            
            # Convert to numpy array
            features = np.array(features).reshape(1, -1)
            
            # Scale features if scaler is fitted
            if hasattr(self.scaler, 'mean_'):
                features = self.scaler.transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features for exit timing model: {str(e)}")
            return np.zeros((1, 10))
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Predict exit probability and optimal exit time.
        
        Args:
            features: Feature array
            
        Returns:
            Tuple of (exit probability, optimal exit days)
        """
        try:
            # If models are not trained, return default values
            if len(self.training_data) < 10:
                return 0.5, 5.0
            
            # Check if exit probability model is trained
            if not hasattr(self.model['exit_prob'], 'classes_'):
                return 0.5, 5.0
            
            # Check if exit days model is trained
            if not hasattr(self.model['exit_days'], 'coef_'):
                return 0.5, 5.0
            
            # Predict exit probability
            exit_prob = self.model['exit_prob'].predict_proba(features)[0][1]
            
            # Predict optimal exit days
            exit_days = max(1.0, self.model['exit_days'].predict(features)[0])
            
            return exit_prob, exit_days
            
        except Exception as e:
            logger.error(f"Error predicting exit timing: {str(e)}")
            return 0.5, 5.0
    
    def update(self, feedback_data: Dict[str, Any]) -> None:
        """
        Update the model with new feedback data.
        
        Args:
            feedback_data: Feedback data from a completed trade
        """
        try:
            # Extract position information
            position = {
                'symbol': feedback_data.get('symbol', ''),
                'strategy': feedback_data.get('strategy', ''),
                'direction': 'BUY' if feedback_data.get('position_size', 0) > 0 else 'SELL',
                'average_cost': feedback_data.get('entry_price', 0),
                'days_held': feedback_data.get('duration_days', 0),
                'size_pct': abs(feedback_data.get('position_size', 0.05))
            }
            
            # Get market data from Knowledge Base
            symbol = position['symbol']
            end_date = datetime.fromisoformat(feedback_data.get('exit_time', datetime.now().isoformat()))
            start_date = end_date - timedelta(days=30)
            
            market_data = self.kb.get_market_data(symbol, start_date, end_date)
            
            # If market data is not available, skip update
            if market_data.empty:
                return
            
            # Prepare features
            features = self.prepare_features(position, market_data)
            
            # Determine labels
            pnl = feedback_data.get('pnl', 0)
            pnl_pct = feedback_data.get('pnl_pct', 0)
            
            # Exit probability label (1 if should exit, 0 if should hold)
            exit_label = 1 if pnl > 0 else 0
            
            # Optimal exit days label
            if pnl_pct > 0.05:  # Very good trade
                optimal_days = position['days_held']
            elif pnl_pct > 0:  # Good trade
                optimal_days = position['days_held']
            elif pnl_pct > -0.02:  # Small loss
                optimal_days = max(1, position['days_held'] * 0.8)
            else:  # Large loss
                optimal_days = max(1, position['days_held'] * 0.5)
            
            # Add to training data
            self.training_data.append((features[0], exit_label, optimal_days))
            
            # Train model if enough data is available
            if len(self.training_data) >= 10 and len(self.training_data) % 5 == 0:
                self.train()
                
        except Exception as e:
            logger.error(f"Error updating exit timing model: {str(e)}")
    
    def train(self) -> None:
        """Train the model on collected data."""
        try:
            # Extract features and labels
            X = np.array([data[0] for data in self.training_data])
            y_exit = np.array([data[1] for data in self.training_data])
            y_days = np.array([data[2] for data in self.training_data])
            
            # Fit scaler
            self.scaler.fit(X)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Train exit probability model
            self.model['exit_prob'].fit(X_scaled, y_exit)
            
            # Train exit days model
            self.model['exit_days'].fit(X_scaled, y_days)
            
            # Evaluate models
            if len(X) >= 20:
                # Evaluate exit probability model
                exit_cv_scores = cross_val_score(self.model['exit_prob'], X_scaled, y_exit, cv=5)
                logger.info(f"Exit probability model cross-validation accuracy: {np.mean(exit_cv_scores):.4f}")
                
                # Evaluate exit days model
                days_cv_scores = cross_val_score(self.model['exit_days'], X_scaled, y_days, cv=5, scoring='neg_mean_squared_error')
                days_rmse = np.sqrt(-np.mean(days_cv_scores))
                logger.info(f"Exit days model cross-validation RMSE: {days_rmse:.4f}")
                
            logger.info(f"Trained exit timing models on {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training exit timing models: {str(e)}")
    
    def _encode_strategy(self, strategy: str) -> float:
        """
        Encode strategy as a numeric value.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Encoded value
        """
        strategy_map = {
            'momentum': 0.0,
            'sector_rotation': 0.25,
            'swing_trading': 0.5,
            'mean_reversion': 0.75,
            'trend_following': 1.0
        }
        
        return strategy_map.get(strategy.lower(), 0.0)


class MarketRegimeModel(LearningModel):
    """Model for detecting market regimes."""
    
    def __init__(self, models_dir: Path):
        """
        Initialize the market regime model.
        
        Args:
            models_dir: Directory to store trained models
        """
        super().__init__(models_dir, "market_regime")
        
        # Initialize model if not loaded
        if self.model is None:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            
        # Define regime types
        self.regime_types = ['bullish', 'bearish', 'sideways', 'volatile']
    
    def prepare_features(self, market_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Args:
            market_data: Market data for multiple symbols
            
        Returns:
            Feature array
        """
        try:
            # Use SPY as market proxy if available
            spy_data = market_data.get('SPY', None)
            
            # If SPY is not available, use the first available symbol
            if spy_data is None or spy_data.empty:
                for symbol, data in market_data.items():
                    if not data.empty:
                        spy_data = data
                        break
            
            # If no data is available, return zeros
            if spy_data is None or spy_data.empty:
                return np.zeros((1, 10))
            
            # Calculate market features
            
            # Returns
            returns = spy_data['close'].pct_change().dropna()
            
            # Volatility
            volatility = returns.rolling(window=20).std().iloc[-1] if len(returns) >= 20 else 0
            
            # Trend
            sma_20 = spy_data['close'].rolling(window=20).mean()
            sma_50 = spy_data['close'].rolling(window=50).mean()
            sma_200 = spy_data['close'].rolling(window=200).mean()
            
            trend_20_50 = sma_20.iloc[-1] / sma_50.iloc[-1] - 1 if len(sma_50) > 0 and sma_50.iloc[-1] > 0 else 0
            trend_50_200 = sma_50.iloc[-1] / sma_200.iloc[-1] - 1 if len(sma_200) > 0 and sma_200.iloc[-1] > 0 else 0
            
            # Momentum
            momentum_20 = spy_data['close'].iloc[-1] / spy_data['close'].iloc[-20] - 1 if len(spy_data) >= 20 else 0
            
            # RSI
            delta = spy_data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
            
            # Volume
            volume_ratio = spy_data['volume'].iloc[-5:].mean() / spy_data['volume'].iloc[-20:-5].mean() if len(spy_data) >= 20 else 1
            
            # Correlation between symbols
            if len(market_data) >= 2:
                # Calculate returns for all symbols
                symbol_returns = {}
                for symbol, data in market_data.items():
                    if not data.empty and len(data) >= 20:
                        symbol_returns[symbol] = data['close'].pct_change().dropna()
                
                # Calculate average correlation
                correlations = []
                symbols = list(symbol_returns.keys())
                for i in range(len(symbols)):
                    for j in range(i+1, len(symbols)):
                        if len(symbol_returns[symbols[i]]) > 0 and len(symbol_returns[symbols[j]]) > 0:
                            corr = symbol_returns[symbols[i]].corr(symbol_returns[symbols[j]])
                            if not np.isnan(corr):
                                correlations.append(corr)
                
                avg_correlation = np.mean(correlations) if correlations else 0
            else:
                avg_correlation = 0
            
            # Extract features
            features = [
                returns.mean() * 100,  # Average daily return (%)
                volatility * 100,  # Daily volatility (%)
                trend_20_50,
                trend_50_200,
                momentum_20,
                current_rsi / 100,  # Normalize
                volume_ratio,
                avg_correlation,
                np.percentile(returns, 10) * 100,  # 10th percentile return (%)
                np.percentile(returns, 90) * 100   # 90th percentile return (%)
            ]
            
            # Convert to numpy array
            features = np.array(features).reshape(1, -1)
            
            # Scale features if scaler is fitted
            if hasattr(self.scaler, 'mean_'):
                features = self.scaler.transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features for market regime model: {str(e)}")
            return np.zeros((1, 10))
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict market regime.
        
        Args:
            features: Feature array
            
        Returns:
            Tuple of (regime type, confidence)
        """
        try:
            # If model is not trained, use heuristic approach
            if len(self.training_data) < 10 or not hasattr(self.model, 'classes_'):
                return self._heuristic_regime_detection(features)
            
            # Predict regime
            regime_idx = self.model.predict(features)[0]
            regime = self.regime_types[regime_idx]
            
            # Get prediction confidence
            proba = self.model.predict_proba(features)[0]
            confidence = proba[regime_idx]
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Error predicting market regime: {str(e)}")
            return 'unknown', 0.0
    
    def update(self, feedback_data: Dict[str, Any]) -> None:
        """
        Update the model with new feedback data.
        
        Args:
            feedback_data: Feedback data from a completed trade
        """
        try:
            # Get market data from Knowledge Base
            end_date = datetime.fromisoformat(feedback_data.get('exit_time', datetime.now().isoformat()))
            start_date = end_date - timedelta(days=30)
            
            market_data = {}
            for symbol in ['SPY', 'QQQ', 'IWM', 'DIA']:
                data = self.kb.get_market_data(symbol, start_date, end_date)
                if not data.empty:
                    market_data[symbol] = data
            
            # If market data is not available, skip update
            if not market_data:
                return
            
            # Prepare features
            features = self.prepare_features(market_data)
            
            # Determine regime label
            pnl = feedback_data.get('pnl', 0)
            strategy = feedback_data.get('strategy', '')
            
            # Infer regime from strategy performance
            if strategy == 'momentum' and pnl > 0:
                regime = 'bullish'
            elif strategy == 'mean_reversion' and pnl > 0:
                regime = 'sideways'
            elif pnl < 0 and feedback_data.get('duration_days', 0) < 5:
                regime = 'volatile'
            elif pnl < 0:
                regime = 'bearish'
            else:
                # Use heuristic approach if no clear signal
                regime, _ = self._heuristic_regime_detection(features)
            
            # Convert regime to index
            regime_idx = self.regime_types.index(regime) if regime in self.regime_types else 0
            
            # Add to training data
            self.training_data.append((features[0], regime_idx))
            
            # Train model if enough data is available
            if len(self.training_data) >= 10 and len(self.training_data) % 5 == 0:
                self.train()
                
        except Exception as e:
            logger.error(f"Error updating market regime model: {str(e)}")
    
    def train(self) -> None:
        """Train the model on collected data."""
        try:
            # Extract features and labels
            X = np.array([data[0] for data in self.training_data])
            y = np.array([data[1] for data in self.training_data])
            
            # Fit scaler
            self.scaler.fit(X)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Evaluate model
            if len(X) >= 20:
                cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
                logger.info(f"Market regime model cross-validation accuracy: {np.mean(cv_scores):.4f}")
                
            logger.info(f"Trained market regime model on {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training market regime model: {str(e)}")
    
    def _heuristic_regime_detection(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Detect market regime using heuristics.
        
        Args:
            features: Feature array
            
        Returns:
            Tuple of (regime type, confidence)
        """
        try:
            # Unscale features if needed
            if hasattr(self.scaler, 'mean_'):
                features = self.scaler.inverse_transform(features)
            
            # Extract key metrics
            avg_return = features[0, 0]
            volatility = features[0, 1]
            trend_20_50 = features[0, 2]
            trend_50_200 = features[0, 3]
            momentum = features[0, 4]
            rsi = features[0, 5] * 100  # Denormalize
            
            # Determine regime
            if avg_return > 0.1 and trend_20_50 > 0.01 and trend_50_200 > 0.01 and momentum > 0.02:
                regime = 'bullish'
                confidence = min(1.0, (avg_return / 0.2 + trend_20_50 / 0.02 + momentum / 0.05) / 3)
            elif avg_return < -0.1 and trend_20_50 < -0.01 and momentum < -0.02:
                regime = 'bearish'
                confidence = min(1.0, (abs(avg_return) / 0.2 + abs(trend_20_50) / 0.02 + abs(momentum) / 0.05) / 3)
            elif volatility > 1.5:
                regime = 'volatile'
                confidence = min(1.0, volatility / 3.0)
            else:
                regime = 'sideways'
                confidence = min(1.0, (1.0 - abs(avg_return) / 0.2) * (1.0 - abs(trend_20_50) / 0.02))
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Error in heuristic regime detection: {str(e)}")
            return 'unknown', 0.0


class StrategySelectionModel(LearningModel):
    """Model for selecting optimal trading strategies."""
    
    def __init__(self, models_dir: Path):
        """
        Initialize the strategy selection model.
        
        Args:
            models_dir: Directory to store trained models
        """
        super().__init__(models_dir, "strategy_selection")
        
        # Initialize model if not loaded
        if self.model is None:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            
        # Define strategy types
        self.strategy_types = ['momentum', 'sector_rotation', 'swing_trading', 'mean_reversion']
    
    def prepare_features(self, symbol: str, market_regime: Dict[str, Any]) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Args:
            symbol: Symbol to select strategy for
            market_regime: Current market regime
            
        Returns:
            Feature array
        """
        try:
            # Get symbol data from Knowledge Base
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            symbol_data = self.kb.get_market_data(symbol, start_date, end_date)
            
            # If symbol data is not available, use default features
            if symbol_data.empty:
                return np.zeros((1, 10))
            
            # Calculate symbol features
            
            # Returns
            returns = symbol_data['close'].pct_change().dropna()
            
            # Volatility
            volatility = returns.rolling(window=20).std().iloc[-1] if len(returns) >= 20 else 0
            
            # Trend
            sma_20 = symbol_data['close'].rolling(window=20).mean()
            sma_50 = symbol_data['close'].rolling(window=50).mean()
            
            trend_20_50 = sma_20.iloc[-1] / sma_50.iloc[-1] - 1 if len(sma_50) > 0 and sma_50.iloc[-1] > 0 else 0
            
            # Momentum
            momentum_20 = symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[-20] - 1 if len(symbol_data) >= 20 else 0
            
            # RSI
            delta = symbol_data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
            
            # Volume
            volume_ratio = symbol_data['volume'].iloc[-5:].mean() / symbol_data['volume'].iloc[-20:-5].mean() if len(symbol_data) >= 20 else 1
            
            # Market regime features
            regime_value = self._encode_regime(market_regime.get('regime', 'unknown'))
            regime_confidence = market_regime.get('confidence', 0.0)
            
            # Extract features
            features = [
                returns.mean() * 100,  # Average daily return (%)
                volatility * 100,  # Daily volatility (%)
                trend_20_50,
                momentum_20,
                current_rsi / 100,  # Normalize
                volume_ratio,
                regime_value,
                regime_confidence,
                np.percentile(returns, 10) * 100 if len(returns) >= 10 else -1,  # 10th percentile return (%)
                np.percentile(returns, 90) * 100 if len(returns) >= 10 else 1    # 90th percentile return (%)
            ]
            
            # Convert to numpy array
            features = np.array(features).reshape(1, -1)
            
            # Scale features if scaler is fitted
            if hasattr(self.scaler, 'mean_'):
                features = self.scaler.transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features for strategy selection model: {str(e)}")
            return np.zeros((1, 10))
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict optimal strategy.
        
        Args:
            features: Feature array
            
        Returns:
            Tuple of (strategy name, confidence)
        """
        try:
            # If model is not trained, use heuristic approach
            if len(self.training_data) < 10 or not hasattr(self.model, 'classes_'):
                return self._heuristic_strategy_selection(features)
            
            # Predict strategy
            strategy_idx = self.model.predict(features)[0]
            strategy = self.strategy_types[strategy_idx]
            
            # Get prediction confidence
            proba = self.model.predict_proba(features)[0]
            confidence = proba[strategy_idx]
            
            return strategy, confidence
            
        except Exception as e:
            logger.error(f"Error predicting optimal strategy: {str(e)}")
            return 'momentum', 0.5
    
    def update(self, feedback_data: Dict[str, Any]) -> None:
        """
        Update the model with new feedback data.
        
        Args:
            feedback_data: Feedback data from a completed trade
        """
        try:
            # Extract trade information
            symbol = feedback_data.get('symbol', '')
            strategy = feedback_data.get('strategy', '')
            pnl = feedback_data.get('pnl', 0)
            
            # Skip update if strategy is not in our list
            if strategy not in self.strategy_types:
                return
            
            # Get market regime at trade time
            exit_time = datetime.fromisoformat(feedback_data.get('exit_time', datetime.now().isoformat()))
            market_regime = self.kb.get_market_regime(exit_time)
            
            # If market regime is not available, use a default
            if not market_regime:
                market_regime = {
                    'regime': 'unknown',
                    'confidence': 0.0
                }
            
            # Prepare features
            features = self.prepare_features(symbol, market_regime)
            
            # Determine if strategy was successful
            strategy_idx = self.strategy_types.index(strategy)
            
            # Add to training data only if trade was profitable
            if pnl > 0:
                self.training_data.append((features[0], strategy_idx))
                
                # Train model if enough data is available
                if len(self.training_data) >= 10 and len(self.training_data) % 5 == 0:
                    self.train()
                
        except Exception as e:
            logger.error(f"Error updating strategy selection model: {str(e)}")
    
    def train(self) -> None:
        """Train the model on collected data."""
        try:
            # Extract features and labels
            X = np.array([data[0] for data in self.training_data])
            y = np.array([data[1] for data in self.training_data])
            
            # Fit scaler
            self.scaler.fit(X)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Evaluate model
            if len(X) >= 20:
                cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
                logger.info(f"Strategy selection model cross-validation accuracy: {np.mean(cv_scores):.4f}")
                
            logger.info(f"Trained strategy selection model on {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training strategy selection model: {str(e)}")
    
    def _heuristic_strategy_selection(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Select strategy using heuristics.
        
        Args:
            features: Feature array
            
        Returns:
            Tuple of (strategy name, confidence)
        """
        try:
            # Unscale features if needed
            if hasattr(self.scaler, 'mean_'):
                features = self.scaler.inverse_transform(features)
            
            # Extract key metrics
            avg_return = features[0, 0]
            volatility = features[0, 1]
            trend = features[0, 2]
            momentum = features[0, 3]
            rsi = features[0, 4] * 100  # Denormalize
            regime_value = features[0, 6]
            
            # Determine strategy
            if momentum > 0.02 and trend > 0.01 and regime_value > 0.6:
                # Strong uptrend in bullish market
                strategy = 'momentum'
                confidence = min(1.0, (momentum / 0.05 + trend / 0.02 + regime_value / 0.8) / 3)
            elif rsi < 30 and trend < -0.01:
                # Oversold in downtrend
                strategy = 'mean_reversion'
                confidence = min(1.0, (1.0 - rsi / 30) * (abs(trend) / 0.02))
            elif rsi > 70 and trend > 0.01:
                # Overbought in uptrend
                strategy = 'swing_trading'
                confidence = min(1.0, (rsi / 70 - 1.0) * (trend / 0.02))
            elif volatility < 1.0 and abs(trend) < 0.005:
                # Low volatility sideways market
                strategy = 'sector_rotation'
                confidence = min(1.0, (1.0 - volatility / 1.0) * (1.0 - abs(trend) / 0.005))
            else:
                # Default to momentum
                strategy = 'momentum'
                confidence = 0.5
            
            return strategy, confidence
            
        except Exception as e:
            logger.error(f"Error in heuristic strategy selection: {str(e)}")
            return 'momentum', 0.5
    
    def _encode_regime(self, regime: str) -> float:
        """
        Encode market regime as a numeric value.
        
        Args:
            regime: Market regime name
            
        Returns:
            Encoded value
        """
        regime_map = {
            'bullish': 1.0,
            'sideways': 0.5,
            'volatile': 0.25,
            'bearish': 0.0,
            'unknown': 0.5
        }
        
        return regime_map.get(regime.lower(), 0.5)


class PerformanceMetrics:
    """Class for calculating performance metrics."""
    
    def calculate_metrics(self, orders: List[Dict[str, Any]], positions: List[Dict[str, Any]],
                         start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            orders: List of orders
            positions: List of positions
            start_date: Start date for metrics calculation
            end_date: End date for metrics calculation
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Filter orders by date
            filtered_orders = [
                order for order in orders
                if start_date <= datetime.fromisoformat(order.get('timestamp', '')) <= end_date
            ]
            
            # Calculate metrics
            
            # Total return
            total_pnl = sum(order.get('pnl', 0) for order in filtered_orders if 'pnl' in order)
            
            # Win rate
            winning_trades = [order for order in filtered_orders if order.get('pnl', 0) > 0]
            losing_trades = [order for order in filtered_orders if order.get('pnl', 0) < 0]
            
            win_rate = len(winning_trades) / len(filtered_orders) if filtered_orders else 0
            
            # Average win and loss
            avg_win = sum(order.get('pnl', 0) for order in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(order.get('pnl', 0) for order in losing_trades) / len(losing_trades) if losing_trades else 0
            
            # Profit factor
            gross_profit = sum(order.get('pnl', 0) for order in winning_trades)
            gross_loss = abs(sum(order.get('pnl', 0) for order in losing_trades))
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Sharpe ratio
            daily_returns = []
            current_date = start_date
            while current_date <= end_date:
                daily_orders = [
                    order for order in filtered_orders
                    if datetime.fromisoformat(order.get('timestamp', '')).date() == current_date.date()
                ]
                
                daily_pnl = sum(order.get('pnl', 0) for order in daily_orders)
                daily_returns.append(daily_pnl)
                
                current_date += timedelta(days=1)
            
            if daily_returns:
                avg_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            cumulative_returns = np.cumsum(daily_returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / (1 + peak)
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            # Current exposure
            total_exposure = sum(position.get('market_value', 0) for position in positions)
            
            # Create metrics dictionary
            metrics = {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'total_trades': len(filtered_orders),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'total_return': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'current_exposure': total_exposure
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}


class StrategyOptimizer:
    """Class for optimizing trading strategy parameters."""
    
    def optimize(self, strategy: str, current_params: Dict[str, Any], 
                performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize parameters for a trading strategy.
        
        Args:
            strategy: Strategy to optimize
            current_params: Current parameters
            performance_data: Performance data for the strategy
            
        Returns:
            Optimized parameters
        """
        try:
            # Clone current parameters
            optimized_params = current_params.copy()
            
            # Optimize based on strategy type
            if strategy == 'momentum':
                optimized_params = self._optimize_momentum_params(current_params, performance_data)
            elif strategy == 'sector_rotation':
                optimized_params = self._optimize_sector_rotation_params(current_params, performance_data)
            elif strategy == 'swing_trading':
                optimized_params = self._optimize_swing_trading_params(current_params, performance_data)
            elif strategy == 'mean_reversion':
                optimized_params = self._optimize_mean_reversion_params(current_params, performance_data)
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {str(e)}")
            return current_params
    
    def _optimize_momentum_params(self, current_params: Dict[str, Any], 
                                 performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize momentum strategy parameters.
        
        Args:
            current_params: Current parameters
            performance_data: Performance data
            
        Returns:
            Optimized parameters
        """
        try:
            # Clone current parameters
            params = current_params.copy()
            
            # Extract performance metrics
            win_rate = performance_data.get('win_rate', 0.5)
            profit_factor = performance_data.get('profit_factor', 1.0)
            
            # Adjust lookback period
            lookback = params.get('lookback_period', 20)
            if win_rate < 0.4:
                # Increase lookback for more stability
                params['lookback_period'] = min(60, lookback + 10)
            elif win_rate > 0.6 and profit_factor > 1.5:
                # Decrease lookback for more responsiveness
                params['lookback_period'] = max(10, lookback - 5)
            
            # Adjust momentum threshold
            threshold = params.get('momentum_threshold', 0.05)
            if win_rate < 0.4:
                # Increase threshold for stronger signals
                params['momentum_threshold'] = min(0.15, threshold + 0.02)
            elif win_rate > 0.6 and profit_factor > 1.5:
                # Decrease threshold for more signals
                params['momentum_threshold'] = max(0.02, threshold - 0.01)
            
            return params
            
        except Exception as e:
            logger.error(f"Error optimizing momentum parameters: {str(e)}")
            return current_params
    
    def _optimize_sector_rotation_params(self, current_params: Dict[str, Any], 
                                        performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize sector rotation strategy parameters.
        
        Args:
            current_params: Current parameters
            performance_data: Performance data
            
        Returns:
            Optimized parameters
        """
        try:
            # Clone current parameters
            params = current_params.copy()
            
            # Extract performance metrics
            win_rate = performance_data.get('win_rate', 0.5)
            profit_factor = performance_data.get('profit_factor', 1.0)
            
            # Adjust top sectors count
            top_count = params.get('top_sectors_count', 3)
            if win_rate < 0.4:
                # Decrease count for more selectivity
                params['top_sectors_count'] = max(1, top_count - 1)
            elif win_rate > 0.6 and profit_factor > 1.5:
                # Increase count for more diversification
                params['top_sectors_count'] = min(5, top_count + 1)
            
            # Adjust rotation period
            period = params.get('rotation_period_days', 30)
            if win_rate < 0.4:
                # Increase period for more stability
                params['rotation_period_days'] = min(90, period + 15)
            elif win_rate > 0.6 and profit_factor > 1.5:
                # Decrease period for more responsiveness
                params['rotation_period_days'] = max(15, period - 7)
            
            return params
            
        except Exception as e:
            logger.error(f"Error optimizing sector rotation parameters: {str(e)}")
            return current_params
    
    def _optimize_swing_trading_params(self, current_params: Dict[str, Any], 
                                      performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize swing trading strategy parameters.
        
        Args:
            current_params: Current parameters
            performance_data: Performance data
            
        Returns:
            Optimized parameters
        """
        try:
            # Clone current parameters
            params = current_params.copy()
            
            # Extract performance metrics
            win_rate = performance_data.get('win_rate', 0.5)
            profit_factor = performance_data.get('profit_factor', 1.0)
            
            # Adjust overbought threshold
            overbought = params.get('overbought_threshold', 70)
            if win_rate < 0.4:
                # Increase threshold for stronger signals
                params['overbought_threshold'] = min(80, overbought + 5)
            elif win_rate > 0.6 and profit_factor > 1.5:
                # Decrease threshold for more signals
                params['overbought_threshold'] = max(65, overbought - 2)
            
            # Adjust oversold threshold
            oversold = params.get('oversold_threshold', 30)
            if win_rate < 0.4:
                # Decrease threshold for stronger signals
                params['oversold_threshold'] = max(20, oversold - 5)
            elif win_rate > 0.6 and profit_factor > 1.5:
                # Increase threshold for more signals
                params['oversold_threshold'] = min(35, oversold + 2)
            
            return params
            
        except Exception as e:
            logger.error(f"Error optimizing swing trading parameters: {str(e)}")
            return current_params
    
    def _optimize_mean_reversion_params(self, current_params: Dict[str, Any], 
                                       performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize mean reversion strategy parameters.
        
        Args:
            current_params: Current parameters
            performance_data: Performance data
            
        Returns:
            Optimized parameters
        """
        try:
            # Clone current parameters
            params = current_params.copy()
            
            # Extract performance metrics
            win_rate = performance_data.get('win_rate', 0.5)
            profit_factor = performance_data.get('profit_factor', 1.0)
            
            # Adjust deviation threshold
            deviation = params.get('deviation_threshold', 2.0)
            if win_rate < 0.4:
                # Increase threshold for stronger signals
                params['deviation_threshold'] = min(3.0, deviation + 0.2)
            elif win_rate > 0.6 and profit_factor > 1.5:
                # Decrease threshold for more signals
                params['deviation_threshold'] = max(1.5, deviation - 0.1)
            
            # Adjust lookback period
            lookback = params.get('lookback_period', 20)
            if win_rate < 0.4:
                # Increase lookback for more stability
                params['lookback_period'] = min(40, lookback + 5)
            elif win_rate > 0.6 and profit_factor > 1.5:
                # Decrease lookback for more responsiveness
                params['lookback_period'] = max(10, lookback - 2)
            
            return params
            
        except Exception as e:
            logger.error(f"Error optimizing mean reversion parameters: {str(e)}")
            return current_params


class ParameterTuner:
    """Class for tuning system parameters."""
    
    def tune_risk_parameters(self, current_params: Dict[str, Any], 
                            performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tune risk management parameters.
        
        Args:
            current_params: Current parameters
            performance_data: Performance data
            
        Returns:
            Tuned parameters
        """
        try:
            # Clone current parameters
            params = current_params.copy()
            
            # Extract performance metrics
            win_rate = performance_data.get('win_rate', 0.5)
            profit_factor = performance_data.get('profit_factor', 1.0)
            max_drawdown = performance_data.get('max_drawdown', 0.1)
            
            # Adjust position size
            position_size = params.get('max_position_size_pct', 0.05)
            if max_drawdown > 0.15:
                # Decrease position size for high drawdown
                params['max_position_size_pct'] = max(0.01, position_size * 0.8)
            elif max_drawdown < 0.05 and win_rate > 0.6 and profit_factor > 1.5:
                # Increase position size for low drawdown and good performance
                params['max_position_size_pct'] = min(0.1, position_size * 1.2)
            
            # Adjust sector exposure
            sector_exposure = params.get('max_sector_exposure_pct', 0.3)
            if max_drawdown > 0.15:
                # Decrease sector exposure for high drawdown
                params['max_sector_exposure_pct'] = max(0.1, sector_exposure * 0.8)
            elif max_drawdown < 0.05 and win_rate > 0.6 and profit_factor > 1.5:
                # Increase sector exposure for low drawdown and good performance
                params['max_sector_exposure_pct'] = min(0.5, sector_exposure * 1.2)
            
            # Adjust leverage
            leverage = params.get('max_leverage', 1.0)
            if max_drawdown > 0.1:
                # Decrease leverage for high drawdown
                params['max_leverage'] = max(1.0, leverage * 0.8)
            elif max_drawdown < 0.05 and win_rate > 0.6 and profit_factor > 2.0:
                # Increase leverage for low drawdown and excellent performance
                params['max_leverage'] = min(1.5, leverage * 1.1)
            
            return params
            
        except Exception as e:
            logger.error(f"Error tuning risk parameters: {str(e)}")
            return current_params


class FeedbackLoop:
    """Class for processing feedback from trades."""
    
    def process_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """
        Process feedback from a completed trade.
        
        Args:
            feedback_data: Feedback data
        """
        try:
            # Extract trade information
            symbol = feedback_data.get('symbol', '')
            strategy = feedback_data.get('strategy', '')
            pnl = feedback_data.get('pnl', 0)
            pnl_pct = feedback_data.get('pnl_pct', 0)
            duration_days = feedback_data.get('duration_days', 0)
            
            # Log feedback
            if pnl > 0:
                logger.info(f"Positive feedback: {symbol} {strategy} trade with {pnl_pct:.2%} return over {duration_days:.1f} days")
            else:
                logger.info(f"Negative feedback: {symbol} {strategy} trade with {pnl_pct:.2%} return over {duration_days:.1f} days")
            
            # Store feedback in Knowledge Base
            self.kb.store_trade_feedback(feedback_data)
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
