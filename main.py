import importlib
import subprocess
import sys
import os
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import traceback

def install_if_missing(package_name, pip_name=None):
    try:
        importlib.import_module(package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or package_name])

# Install required packages
packages = [
    ('yfinance', None),
    ('pandas', None),
    ('numpy', None),
    ('sklearn', 'scikit-learn'),
    ('flask', None),
    ('binance', 'python-binance'),
    ('imbalanced_learn', 'imbalanced-learn'),
    ('requests', None)
]

for pkg, pip_name in packages:
    install_if_missing(pkg, pip_name)

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    RandomOverSampler = None
from flask import Flask, render_template, jsonify, request
from binance.client import Client
import requests
import json
import joblib
import copy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class TradingConfig:
    """Trading configuration and parameters"""
    DEFAULT_PARAMS = {
        'SYMBOL': 'BTCUSDT',
        'INTERVAL': '5m',
        'TRAINING_PERIOD': '60d',
        'BACKTEST_PERIOD': '30d',
        'START_CAPITAL': 10000,
        'MAKER_FEE_RATE': 0.001,
        'TAKER_FEE_RATE': 0.001,
        'STOP_LOSS_PCT': 0.05,
        'TAKE_PROFIT_PCT': 0.008,
        'TRAIL_STOP_PCT': 0.02,
        'BUY_PROB_THRESHOLD': 0.9,
        'MODEL_TYPE': 'HistGradientBoosting',
        'CONFIDENCE_THRESHOLD': 0.7,
        'FUTURE_WINDOW': 24,
        'USE_OVERSAMPLING': True,
        'TEST_SIZE': 0.3,
        'SIGNAL_EXIT_THRESHOLD': 0.25,
        'SIGNAL_EXIT_CONSECUTIVE': 3,
        'MIN_HOLD_CANDLES': 6
    }

class SettingsManager:
    """Manage saving and loading of trading parameters"""

    def __init__(self, settings_dir: str = "settings"):
        self.settings_dir = settings_dir
        os.makedirs(settings_dir, exist_ok=True)

    def save_settings(self, params: Dict, name: str = None) -> str:
        """Save parameters to a JSON file"""
        try:
            if name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"settings_{timestamp}"

            if not name.endswith('.json'):
                name += '.json'

            filepath = os.path.join(self.settings_dir, name)

            # Create a copy and add metadata
            settings_data = copy.deepcopy(params)
            settings_data['_metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'name': name,
                'version': '1.0'
            }

            with open(filepath, 'w') as f:
                json.dump(settings_data, f, indent=2)

            logger.info(f"Settings saved to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            raise e

    def load_settings(self, filepath: str) -> Dict:
        """Load parameters from a JSON file"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Settings file not found: {filepath}")

            with open(filepath, 'r') as f:
                settings_data = json.load(f)

            # Remove metadata before returning
            if '_metadata' in settings_data:
                del settings_data['_metadata']

            logger.info(f"Settings loaded from: {filepath}")
            return settings_data

        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            raise e

    def list_saved_settings(self) -> List[Dict]:
        """List all saved settings with metadata"""
        try:
            settings = []
            if not os.path.exists(self.settings_dir):
                return settings

            for filename in os.listdir(self.settings_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.settings_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)

                        metadata = data.get('_metadata', {})
                        settings.append({
                            'filename': filename,
                            'filepath': filepath,
                            'name': metadata.get('name', filename),
                            'saved_at': metadata.get('saved_at', 'Unknown'),
                            'symbol': data.get('SYMBOL', 'Unknown'),
                            'model_type': data.get('MODEL_TYPE', 'Unknown'),
                            'start_capital': data.get('START_CAPITAL', 0)
                        })
                    except Exception as e:
                        logger.error(f"Error reading settings file {filename}: {e}")
                        continue

            # Sort by saved_at date (newest first)
            settings.sort(key=lambda x: x['saved_at'], reverse=True)
            return settings

        except Exception as e:
            logger.error(f"Error listing settings: {e}")
            return []

    def delete_settings(self, filepath: str) -> bool:
        """Delete a settings file"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Settings deleted: {filepath}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting settings: {e}")
            return False

class DataProvider:
    """Handles data fetching from various sources"""

    @staticmethod
    def get_binance_data(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data from Binance API"""
        try:
            client = Client()
            interval_map = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY
            }

            binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_5MINUTE)
            klines = client.get_historical_klines(symbol, binance_interval, start_date, end_date)

            if not klines:
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)

            return df[['Open', 'High', 'Low', 'Close', 'Volume']]

        except Exception as e:
            logger.error(f"Binance API error: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_yfinance_data(symbol: str, period: str = '60d', interval: str = '5m') -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            yf_symbol = 'BTC-USD' if symbol == 'BTCUSDT' else symbol
            df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
            if df.empty:
                return pd.DataFrame()

            # Flatten column names if MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            # Ensure we have the right columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in required_cols if col in df.columns]

            if len(available_cols) == len(required_cols):
                df = df[required_cols]
            else:
                logger.error(f"Missing columns in yfinance data. Available: {df.columns.tolist()}")
                return pd.DataFrame()

            return df
        except Exception as e:
            logger.error(f"YFinance error: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_training_data(symbol: str, period: str = '60d', interval: str = '5m') -> pd.DataFrame:
        """Get data specifically for model training"""
        logger.info(f"🔧 TRAINING DATA: Using Yahoo Finance")
        return DataProvider.get_yfinance_data(symbol, period, interval)

    @staticmethod
    def get_backtest_data_with_period(symbol: str, period: str = '30d', interval: str = '5m') -> pd.DataFrame:
        """Get data specifically for backtesting with configurable period"""
        # Try Binance first, then fallback to Yahoo Finance
        data = DataProvider.get_binance_data(symbol, interval, 
                                           (datetime.now() - timedelta(days=int(period.rstrip('d')))).strftime('%Y-%m-%d'),
                                           datetime.now().strftime('%Y-%m-%d'))
        if data.empty:
            yf_symbol = 'BTC-USD' if 'BTC' in symbol else symbol
            data = DataProvider.get_yfinance_data(yf_symbol, period, interval)
        return data

    @staticmethod
    def get_live_data(symbol: str, interval: str = '5m') -> pd.DataFrame:
        """Get live data for live trading - uses same source as training"""
        logger.info(f"📡 LIVE TRADING: Getting latest data")

        # Use Yahoo Finance for consistency with training
        try:
            yf_symbol = 'BTC-USD' if 'BTC' in symbol else symbol
            df = DataProvider.get_yfinance_data(yf_symbol, '5d', interval)
            if not df.empty:
                logger.info(f"✅ Live data: {len(df)} candles from Yahoo Finance")
                return df
            else:
                raise Exception("Empty dataframe from Yahoo Finance")

        except Exception as yf_error:
            logger.error(f"❌ Yahoo Finance error: {yf_error}")

        logger.error("❌ No data source available")
        return pd.DataFrame()

class TechnicalAnalysis:
    """Technical analysis and feature extraction"""

    @staticmethod
    def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()
            rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return pd.Series([50] * len(series), index=series.index)

    @staticmethod
    def extract_features(df: pd.DataFrame, future_window: int = 24, is_live_trading: bool = False) -> pd.DataFrame:
        """Extract technical features using simplified approach"""
        try:
            if df.empty or len(df) < 50:
                return pd.DataFrame()

            df = df.copy()

            # Log input data characteristics for live trading
            if is_live_trading:
                logger.info(f"🔧 LIVE FEATURE EXTRACTION:")
                logger.info(f"   Input data shape: {df.shape}")
                logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
                logger.info(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

            # Simplified feature extraction
            df['returns'] = df['Close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=5).std()
            df['rsi'] = TechnicalAnalysis.compute_rsi(df['Close'])

            # Clean data
            df = df.dropna()

            if df.empty:
                logger.error("All data was NaN after feature extraction")
                return pd.DataFrame()

            if is_live_trading:
                logger.info(f"   ✅ Final features shape: {df.shape}")
                latest = df.iloc[-1]
                logger.info(f"   ✅ Latest: returns={latest['returns']:.6f}, volatility={latest['volatility']:.6f}, rsi={latest['rsi']:.2f}")

            return df

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            traceback.print_exc()
            return pd.DataFrame()

class MLModel:
    """Enhanced machine learning model for trading predictions"""

    def __init__(self, model_type: str = 'HistGradientBoosting'):
        self.model_type = model_type
        self.model = self._get_model(model_type)
        self.scaler = StandardScaler()
        self.features = ['returns', 'volatility', 'rsi']  # Simplified feature set
        self.is_trained = False
        self.training_accuracy = 0.0
        self.test_accuracy = 0.0
        self.validation_accuracy = 0.0
        self.hit_rate = 0.0
        self.class_distribution = {}

    def _get_model(self, model_type: str):
        """Get the appropriate model based on type"""
        if model_type == 'HistGradientBoosting':
            return HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'RandomForest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            logger.warning(f"Unknown model type {model_type}, using HistGradientBoosting")
            return HistGradientBoostingClassifier(max_iter=200, random_state=42)

    def train(self, df: pd.DataFrame, future_window: int = 24, take_profit_pct: float = 0.008, 
              use_oversampling: bool = True, test_size: float = 0.3, training_period: str = '60d'):
        """Training method matching your approach"""
        try:
            if df.empty or len(df) < 200:
                logger.error(f"Insufficient data for training: {len(df)} rows")
                return 0.0

            df = df.copy()

            # Create target
            df['target'] = (df['Close'].shift(-future_window) > df['Close'] * (1 + take_profit_pct)).astype(int)
            df.dropna(inplace=True)

            # Log target distribution
            target_dist = df['target'].value_counts(normalize=True)
            self.class_distribution = target_dist.to_dict()
            logger.info(f"Target distribution: {self.class_distribution}")

            # Use simplified feature set
            X = df[self.features]  # ['returns', 'volatility', 'rsi']
            y = df['target']

            # Check if we have both classes
            if len(y.unique()) < 2:
                logger.error("Target variable has only one class")
                return 0.0

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Apply oversampling
            if use_oversampling and len(y.unique()) == 2 and RandomOverSampler is not None:
                try:
                    ros = RandomOverSampler(random_state=42)
                    X_res, y_res = ros.fit_resample(X_scaled, y)
                    logger.info(f"Applied oversampling: {len(X_scaled)} -> {len(X_res)} samples")
                except Exception as e:
                    logger.warning(f"Oversampling failed, using original data: {e}")
                    X_res, y_res = X_scaled, y
            else:
                if RandomOverSampler is None:
                    logger.warning("RandomOverSampler not available, using original data")
                X_res, y_res = X_scaled, y

            # Train model
            logger.info(f"Training {self.model_type} model...")
            self.model.fit(X_res, y_res)
            self.is_trained = True

            # Calculate training accuracy on original data
            self.training_accuracy = self.model.score(X_scaled, y)

            # Split for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, shuffle=False, random_state=42
            )

            self.test_accuracy = self.model.score(X_test, y_test)
            self.validation_accuracy = self.test_accuracy
            self.hit_rate = self.test_accuracy

            logger.info(f"=== Training Results ===")
            logger.info(f"📊 Model trained with Accuracy: {self.training_accuracy:.4f}")
            logger.info(f"Test Accuracy: {self.test_accuracy:.4f}")
            logger.info(f"Hit Rate: {self.hit_rate:.4f}")
            logger.info(f"Model Type: {self.model_type}")
            logger.info(f"Features Used: {len(self.features)} - {self.features}")

            return self.training_accuracy

        except Exception as e:
            logger.error(f"Model training error: {e}")
            traceback.print_exc()
            return 0.0

    def predict(self, features: np.ndarray, is_live_trading: bool = False) -> Tuple[float, float]:
        """Make prediction and return probability and confidence"""
        try:
            if not self.is_trained or self.model is None:
                logger.warning("Model not trained or None")
                return 0.5, 0.5

            # Ensure features match training features
            if len(features) != len(self.features):
                logger.error(f"Feature mismatch: expected {len(self.features)}, got {len(features)}")
                return 0.5, 0.5

            # Enhanced logging for live trading
            if is_live_trading:
                logger.info(f"🔍 LIVE PREDICTION:")
                logger.info(f"   Features: {[f'{val:.6f}' for val in features]}")

                # Check for NaN or infinite values
                nan_check = [np.isnan(val) or np.isinf(val) for val in features]
                if any(nan_check):
                    logger.error(f"   ❌ INVALID FEATURES: {nan_check}")
                    return 0.5, 0.5

            X_scaled = self.scaler.transform([features])

            if is_live_trading:
                logger.info(f"   Scaled: {[f'{val:.6f}' for val in X_scaled[0]]}")

            probabilities = self.model.predict_proba(X_scaled)[0]

            if is_live_trading:
                logger.info(f"   Raw probabilities: {[f'{p:.6f}' for p in probabilities]}")

            # Fix: Ensure we get the correct class probabilities
            # probabilities[0] = probability of class 0 (no buy)
            # probabilities[1] = probability of class 1 (buy)
            if len(probabilities) > 1:
                probability = probabilities[1]  # Buy probability (class 1)
                confidence = max(probabilities)  # Highest probability (confidence)
            else:
                probability = 0.5
                confidence = 0.5

            if is_live_trading:
                logger.info(f"   ✅ Buy prob: {probability:.6f}, Confidence: {confidence:.6f}")

            return float(probability), float(confidence)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5, 0.5

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available"""
        try:
            if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
                return {}

            importance_dict = dict(zip(self.features, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.error(f"Feature importance error: {e}")
            return {}

    def save_model(self, filepath: str = None) -> str:
        """Save model and scaler to file"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before saving")

            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"models/trading_model_{self.model_type}_{timestamp}.pkl"

            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'features': self.features,
                'training_accuracy': self.training_accuracy,
                'test_accuracy': self.test_accuracy,
                'validation_accuracy': self.validation_accuracy,
                'hit_rate': self.hit_rate,
                'class_distribution': self.class_distribution,
                'saved_at': datetime.now().isoformat()
            }

            joblib.dump(model_data, filepath)
            logger.info(f"Model saved successfully to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise e

    def load_model(self, filepath: str) -> bool:
        """Load model and scaler from file"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")

            model_data = joblib.load(filepath)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.features = model_data['features']
            self.training_accuracy = model_data.get('training_accuracy', 0.0)
            self.test_accuracy = model_data.get('test_accuracy', 0.0)
            self.validation_accuracy = model_data.get('validation_accuracy', 0.0)
            self.hit_rate = model_data.get('hit_rate', 0.0)
            self.class_distribution = model_data.get('class_distribution', {})
            self.is_trained = True

            logger.info(f"Model loaded successfully from: {filepath}")
            logger.info(f"Model type: {self.model_type}, Accuracy: {self.training_accuracy:.4f}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    @staticmethod
    def list_saved_models(models_dir: str = "models") -> List[Dict]:
        """List all saved models with their metadata"""
        try:
            if not os.path.exists(models_dir):
                return []

            models = []
            for filename in os.listdir(models_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(models_dir, filename)
                    try:
                        model_data = joblib.load(filepath)
                        models.append({
                            'filename': filename,
                            'filepath': filepath,
                            'model_type': model_data.get('model_type', 'Unknown'),
                            'training_accuracy': model_data.get('training_accuracy', 0.0),
                            'saved_at': model_data.get('saved_at', 'Unknown'),
                            'features': model_data.get('features', [])
                        })
                    except Exception as e:
                        logger.error(f"Error reading model file {filename}: {e}")
                        continue

            # Sort by saved_at date (newest first)
            models.sort(key=lambda x: x['saved_at'], reverse=True)
            return models

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy,
            'test_accuracy': self.test_accuracy,
            'validation_accuracy': self.validation_accuracy,
            'hit_rate': self.hit_rate,
            'features': self.features,
            'n_features': len(self.features),
            'class_distribution': self.class_distribution,
            'feature_importance': self.get_feature_importance()
        }

class BacktestEngine:
    """Backtesting engine"""

    def __init__(self):
        self.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': False}
        self.logs = []
        self.running = False

    def run(self, params: Dict, start_date: str, end_date: str):
        """Run backtest"""
        self.running = True
        self.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': True}
        self.logs = []

        try:
            self._log(f"🚀 Starting backtest: {params['SYMBOL']} ({start_date} to {end_date})")

            # Get data using configurable backtest period
            backtest_period = params.get('BACKTEST_PERIOD', '30d')
            data = self._get_backtest_data(params['SYMBOL'], params['INTERVAL'], start_date, end_date, backtest_period)
            if data.empty:
                self._log("❌ No data available")
                return

            self._log(f"✅ Retrieved {len(data)} candles using {backtest_period} period")

            # Limit data size for performance (keep last 5000 candles max)
            if len(data) > 5000:
                data = data.tail(5000)
                self._log(f"📊 Limited to {len(data)} candles for performance")

            # Extract features
            self._log("🔧 Extracting technical features...")
            data = TechnicalAnalysis.extract_features(data, params['FUTURE_WINDOW'])

            if data.empty or len(data) < 200:
                self._log("❌ Insufficient data after feature extraction")
                return

            self._log(f"✅ Features extracted from {len(data)} candles")

            # Train model with enhanced parameters including training period
            model = MLModel(params['MODEL_TYPE'])
            training_period = params.get('TRAINING_PERIOD', '60d')
            accuracy = model.train(
                data, 
                params['FUTURE_WINDOW'], 
                params['TAKE_PROFIT_PCT'],
                params.get('USE_OVERSAMPLING', True),
                params.get('TEST_SIZE', 0.3),
                training_period
            )

            # Store the trained model globally
            global current_trained_model
            current_trained_model = model

            # Log model info and store globally
            model_info = model.get_model_info()
            global latest_model_info
            latest_model_info = {
                'training_accuracy': model_info.get('training_accuracy', 0.0),
                'test_accuracy': model_info.get('test_accuracy', 0.0),
                'validation_accuracy': model_info.get('validation_accuracy', 0.0),
                'hit_rate': model_info.get('hit_rate', 0.0),
                'model_type': model_info.get('model_type', 'Unknown'),
                'features_count': model_info.get('n_features', 0),
                'class_distribution': model_info.get('class_distribution', {})
            }
            self._log(f"📊 Model Info: {model_info['model_type']}, Features: {model_info['n_features']}")

            # Split data for testing (use same split as model training)
            split_idx = int(len(data) * (1 - params.get('TEST_SIZE', 0.3)))
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]

            self._log(f"📚 Training: {len(train_data)} candles, Testing: {len(test_data)} candles")

            if accuracy == 0.0:
                self._log("❌ Model training failed")
                return

            self._log(f"🤖 Model trained with {accuracy:.2%} accuracy")

            # Run backtest
            self._simulate_trading(model, test_data, params)

        except Exception as e:
            self._log(f"❌ Backtest error: {str(e)}")
            logger.error(f"Backtest error: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            self.results['running'] = False

    def _get_backtest_data(self, symbol: str, interval: str, start_date: str, end_date: str, period: str = '30d') -> pd.DataFrame:
        """Get data for backtesting"""
        # Check if dates are actually provided (not empty strings)
        has_dates = start_date and end_date and start_date.strip() and end_date.strip()

        if has_dates:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                days_diff = (end_dt - start_dt).days
                self._log(f"📅 Using specific date range: {start_date} to {end_date} ({days_diff} days)")

                # Try Binance API with specific dates first
                data = DataProvider.get_binance_data(symbol, interval, start_date, end_date)

                if data.empty:
                    # If Binance fails, use Yahoo Finance with calculated period
                    calculated_period = f"{max(days_diff, 7)}d"
                    self._log(f"🔄 Binance unavailable, using Yahoo Finance with {calculated_period}...")
                    yf_symbol = 'BTC-USD' if 'BTC' in symbol else symbol
                    data = DataProvider.get_yfinance_data(yf_symbol, calculated_period, interval)

                    # Filter Yahoo data to exact date range
                    if not data.empty:
                        try:
                            start_dt_pd = pd.to_datetime(start_date)
                            end_dt_pd = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                            data = data[(data.index >= start_dt_pd) & (data.index < end_dt_pd)]
                            self._log(f"🔍 Filtered to exact date range: {len(data)} candles")
                        except Exception as e:
                            self._log(f"⚠️ Date filtering failed: {e}")

                return data

            except Exception as e:
                self._log(f"📅 Date parsing error: {e}, falling back to period: {period}")

        # No dates provided or date parsing failed - use period dropdown
        self._log(f"📅 Using backtest period: {period} (no specific dates)")

        # Try Binance/Yahoo Finance with period
        data = DataProvider.get_backtest_data_with_period(symbol, period, interval)

        return data

    def _simulate_trading(self, model: MLModel, data: pd.DataFrame, params: Dict):
        """Simulate trading on historical data"""
        try:
            capital = params['START_CAPITAL']
            position = 0
            btc_amount = 0
            entry_price = 0
            trades = []
            equity_curve = []
            total_fees = 0

            self._log(f"💰 Starting simulation with ${capital:,.2f}")

            # Process data in smaller chunks to avoid hanging
            chunk_size = 100
            total_processed = 0

            for chunk_start in range(0, len(data), chunk_size):
                if not self.running:
                    break

                chunk_end = min(chunk_start + chunk_size, len(data))
                chunk_data = data.iloc[chunk_start:chunk_end]

                for i, (timestamp, candle) in enumerate(chunk_data.iterrows()):
                    if not self.running:
                        break

                    try:
                        price = float(candle['Close'])
                        equity = capital if position == 0 else btc_amount * price
                        equity_curve.append({'timestamp': str(timestamp), 'equity': equity})

                        # Get features for this candle
                        try:
                            available_features = [f for f in model.features if f in candle.index]
                            if len(available_features) != len(model.features):
                                logger.warning(f"Missing features: {set(model.features) - set(available_features)}")
                                continue

                            features = candle[model.features].values
                            prob, confidence = model.predict(features)
                            
                            # Log backtester predictions every 100 candles for comparison
                            if i % 100 == 0:
                                logger.info(f"🔍 BACKTESTER PREDICTION #{i}: Prob={prob:.6f}, Conf={confidence:.6f}, Features={[f'{f:.6f}' for f in features]}")
                        except Exception as e:
                            logger.error(f"Feature extraction error for prediction: {e}")
                            continue

                        # Buy signal
                        if (position == 0 and 
                            prob > params['BUY_PROB_THRESHOLD'] and 
                            confidence > params['CONFIDENCE_THRESHOLD']):

                            fee = capital * params['TAKER_FEE_RATE']
                            btc_amount = (capital - fee) / price
                            entry_price = price
                            highest_price_since_entry = price
                            capital = fee
                            total_fees += fee
                            position = 1

                        # Sell signal
                        elif position == 1:
                            # Update highest price since entry
                            if price > highest_price_since_entry:
                                highest_price_since_entry = price

                            stop_loss = entry_price * (1 - params['STOP_LOSS_PCT'])
                            take_profit = entry_price * (1 + params['TAKE_PROFIT_PCT'])
                            trailing_stop = highest_price_since_entry * (1 - params.get('TRAIL_STOP_PCT', 0.02))

                            # Count how long we've been in this position
                            candles_in_position = getattr(self, '_candles_in_position', 0) + 1
                            self._candles_in_position = candles_in_position

                            # Track consecutive low probability signals
                            if not hasattr(self, '_low_prob_count'):
                                self._low_prob_count = 0

                            if prob < 0.3:
                                self._low_prob_count += 1
                            else:
                                self._low_prob_count = 0

                            exit_reason = None
                            if price <= stop_loss:
                                exit_reason = 'Stop Loss'
                            elif price >= take_profit:
                                exit_reason = 'Take Profit'
                            elif price <= trailing_stop and price >= entry_price * 1.001:
                                exit_reason = 'Trailing Stop'
                            elif (self._low_prob_count >= 3 and
                                  candles_in_position > 6 and
                                  prob < 0.25):
                                exit_reason = 'Signal Exit'

                            if exit_reason:
                                exit_fee = btc_amount * price * params['TAKER_FEE_RATE']
                                capital = btc_amount * price - exit_fee

                                # Calculate PnL including both entry and exit fees
                                entry_fee = capital * params['TAKER_FEE_RATE']
                                total_trade_fees = entry_fee + exit_fee
                                total_fees += total_trade_fees

                                # Calculate both raw P&L and P&L including fees
                                raw_pnl = (price - entry_price) * btc_amount
                                pnl = raw_pnl - total_trade_fees

                                trades.append({
                                    'timestamp': str(timestamp),
                                    'entry_price': entry_price,
                                    'exit_price': price,
                                    'exit_reason': exit_reason,
                                    'pnl': pnl,
                                    'raw_pnl': raw_pnl,
                                    'probability': prob,
                                    'confidence': confidence,
                                    'fees': total_trade_fees,
                                    'highest_price': highest_price_since_entry,
                                    'hold_duration': candles_in_position
                                })

                                # Log trade with both P&L values
                                trade_type = "WIN" if pnl > 0 else "LOSS"
                                self._log(f"🔄 {trade_type}: ${entry_price:.2f} → ${price:.2f} | P&L: ${raw_pnl:.2f} | P&L including fees: ${pnl:.2f} | Fees: ${total_trade_fees:.2f}")

                                position = 0
                                # Reset position tracking
                                self._candles_in_position = 0
                                self._low_prob_count = 0

                        total_processed += 1

                    except Exception as e:
                        logger.error(f"Error processing candle {i}: {e}")
                        continue

                # Log progress every chunk
                progress = (chunk_end / len(data)) * 100
                self._log(f"📈 Processing... {progress:.1f}% complete ({len(trades)} trades so far)")

            # Calculate final metrics
            if trades:
                self._calculate_metrics(trades, params['START_CAPITAL'], total_fees)
            else:
                self._log("⚠️ No trades executed during backtest")
                self.results['metrics'] = {
                    'total_trades': 0,
                    'total_pnl': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'return_pct': 0,
                    'total_fees': 0
                }

            self.results['trades'] = trades
            self.results['equity_curve'] = equity_curve

            self._log(f"✅ Backtest completed - {len(trades)} trades executed")

        except Exception as e:
            self._log(f"❌ Simulation error: {str(e)}")
            logger.error(f"Simulation error: {e}")
            traceback.print_exc()

    def _calculate_metrics(self, trades: List[Dict], start_capital: float, total_fees: float):
        """Calculate backtest performance metrics"""
        try:
            if not trades:
                self.results['metrics'] = {}
                return

            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]

            winner_amount = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
            loser_amount = sum([t['pnl'] for t in losing_trades]) if losing_trades else 0
            total_pnl_with_fees = winner_amount + loser_amount

            # Calculate win/loss ratio
            win_loss_ratio = len(winning_trades) / len(losing_trades) if losing_trades else len(winning_trades)

            metrics = {
                'total_trades': len(trades),
                'winner_count': len(winning_trades),
                'winner_amount': winner_amount,
                'loser_count': len(losing_trades),
                'loser_amount': abs(loser_amount),
                'win_loss_ratio': win_loss_ratio,
                'total_pnl_with_fees': total_pnl_with_fees,
                'total_fees': total_fees
            }

            self.results['metrics'] = metrics

            # Log P&L including fees as header
            self._log(f"💰 === P&L INCLUDING FEES ===")
            self._log(f"💰 Total P&L (with fees): ${total_pnl_with_fees:.2f}")
            self._log(f"💰 Total Fees Paid: ${total_fees:.2f}")
            self._log(f"📊 Final Results: {len(trades)} trades, {len(winning_trades)} winners (${winner_amount:.2f}), {len(losing_trades)} losers (${abs(loser_amount):.2f})")
            self._log(f"📊 Win/Loss Ratio: {win_loss_ratio:.2f}")

        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            self.results['metrics'] = {}

    def _log(self, message: str):
        """Add log message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"{timestamp} | {message}"
        self.logs.append(log_entry)
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]
        logger.info(message)

    def stop(self):
        """Stop backtest"""
        self.running = False
        self._log("🛑 Backtest stopped by user")

class LiveTrader:
    """Live trader - identical copy of BacktestEngine for live trading"""

    def __init__(self):
        self.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': False}
        self.logs = []
        self.running = False
        # Live trading specific properties
        self.model = None
        self.params = {}
        self.current_data = None

    def start(self, params, model):
        """Start live trading using backtester logic"""
        self.running = True
        self.model = model
        self.params = params
        self.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': True}
        self.logs = []
        
        self._log(f"🚀 Live trading started with ${params.get('START_CAPITAL', 10000):.2f}")
        self._log(f"📊 Model: {model.model_type}, Accuracy: {model.training_accuracy:.2%}")
        threading.Thread(target=self._live_trading_loop, daemon=True).start()

    def stop(self):
        """Stop live trading"""
        self.running = False
        self._log("🛑 Live trading stopped")

    def _live_trading_loop(self):
        """Live trading loop using exact backtester logic"""
        try:
            while self.running:
                # Get live data
                data = DataProvider.get_live_data(self.params['SYMBOL'], self.params['INTERVAL'])
                
                if data.empty:
                    self._log("⚠️ No market data, retrying in 30 seconds...")
                    time.sleep(30)
                    continue

                # Extract features exactly like backtester
                data_with_features = TechnicalAnalysis.extract_features(data, self.params['FUTURE_WINDOW'], is_live_trading=True)
                
                if data_with_features.empty:
                    self._log("⚠️ Feature extraction failed, retrying...")
                    time.sleep(30)
                    continue

                # Get latest market data for analysis
                latest_candle = data_with_features.iloc[-1]
                current_price = float(latest_candle['Close'])
                
                # Get prediction
                features = latest_candle[self.model.features].values
                prob, confidence = self.model.predict(features, is_live_trading=False)
                
                # Log market analysis every iteration (every 30 seconds)
                self._log(f"📊 MARKET ANALYSIS:")
                self._log(f"   💰 Current Price: ${current_price:.2f}")
                self._log(f"   🎯 Buy Probability: {prob:.6f} (Threshold: {self.params['BUY_PROB_THRESHOLD']:.2f})")
                self._log(f"   🔒 Confidence: {confidence:.6f} (Threshold: {self.params['CONFIDENCE_THRESHOLD']:.2f})")
                self._log(f"   📈 RSI: {latest_candle['rsi']:.2f}")
                self._log(f"   📊 Volatility: {latest_candle['volatility']:.6f}")
                self._log(f"   💹 Returns: {latest_candle['returns']:.6f}")
                
                # Check if we should buy
                if (prob > self.params['BUY_PROB_THRESHOLD'] and 
                    confidence > self.params['CONFIDENCE_THRESHOLD']):
                    self._log(f"🚀 BUY SIGNAL TRIGGERED!")
                else:
                    self._log(f"⏳ Waiting for buy signal...")
                
                # Run simulation on live data using exact backtester logic
                self._simulate_trading(self.model, data_with_features, self.params, live_mode=True)
                
                # Wait 30 seconds before next iteration
                time.sleep(30)

        except Exception as e:
            logger.error(f"Live trading loop error: {e}")
            self.running = False

    def run(self, params: Dict, start_date: str, end_date: str):
        """Run backtest - identical to BacktestEngine"""
        self.running = True
        self.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': True}
        self.logs = []

        try:
            self._log(f"🚀 Starting backtest: {params['SYMBOL']} ({start_date} to {end_date})")

            # Get data using configurable backtest period
            backtest_period = params.get('BACKTEST_PERIOD', '30d')
            data = self._get_backtest_data(params['SYMBOL'], params['INTERVAL'], start_date, end_date, backtest_period)
            if data.empty:
                self._log("❌ No data available")
                return

            self._log(f"✅ Retrieved {len(data)} candles using {backtest_period} period")

            # Limit data size for performance (keep last 5000 candles max)
            if len(data) > 5000:
                data = data.tail(5000)
                self._log(f"📊 Limited to {len(data)} candles for performance")

            # Extract features
            self._log("🔧 Extracting technical features...")
            data = TechnicalAnalysis.extract_features(data, params['FUTURE_WINDOW'])

            if data.empty or len(data) < 200:
                self._log("❌ Insufficient data after feature extraction")
                return

            self._log(f"✅ Features extracted from {len(data)} candles")

            # Train model with enhanced parameters including training period
            model = MLModel(params['MODEL_TYPE'])
            training_period = params.get('TRAINING_PERIOD', '60d')
            accuracy = model.train(
                data, 
                params['FUTURE_WINDOW'], 
                params['TAKE_PROFIT_PCT'],
                params.get('USE_OVERSAMPLING', True),
                params.get('TEST_SIZE', 0.3),
                training_period
            )

            # Store the trained model globally
            global current_trained_model
            current_trained_model = model

            # Log model info and store globally
            model_info = model.get_model_info()
            global latest_model_info
            latest_model_info = {
                'training_accuracy': model_info.get('training_accuracy', 0.0),
                'test_accuracy': model_info.get('test_accuracy', 0.0),
                'validation_accuracy': model_info.get('validation_accuracy', 0.0),
                'hit_rate': model_info.get('hit_rate', 0.0),
                'model_type': model_info.get('model_type', 'Unknown'),
                'features_count': model_info.get('n_features', 0),
                'class_distribution': model_info.get('class_distribution', {})
            }
            self._log(f"📊 Model Info: {model_info['model_type']}, Features: {model_info['n_features']}")

            # Split data for testing (use same split as model training)
            split_idx = int(len(data) * (1 - params.get('TEST_SIZE', 0.3)))
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]

            self._log(f"📚 Training: {len(train_data)} candles, Testing: {len(test_data)} candles")

            if accuracy == 0.0:
                self._log("❌ Model training failed")
                return

            self._log(f"🤖 Model trained with {accuracy:.2%} accuracy")

            # Run backtest
            self._simulate_trading(model, test_data, params)

        except Exception as e:
            self._log(f"❌ Backtest error: {str(e)}")
            logger.error(f"Backtest error: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            self.results['running'] = False

    def _get_backtest_data(self, symbol: str, interval: str, start_date: str, end_date: str, period: str = '30d') -> pd.DataFrame:
        """Get data for backtesting - identical to BacktestEngine"""
        # Check if dates are actually provided (not empty strings)
        has_dates = start_date and end_date and start_date.strip() and end_date.strip()

        if has_dates:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                days_diff = (end_dt - start_dt).days
                self._log(f"📅 Using specific date range: {start_date} to {end_date} ({days_diff} days)")

                # Try Binance API with specific dates first
                data = DataProvider.get_binance_data(symbol, interval, start_date, end_date)

                if data.empty:
                    # If Binance fails, use Yahoo Finance with calculated period
                    calculated_period = f"{max(days_diff, 7)}d"
                    self._log(f"🔄 Binance unavailable, using Yahoo Finance with {calculated_period}...")
                    yf_symbol = 'BTC-USD' if 'BTC' in symbol else symbol
                    data = DataProvider.get_yfinance_data(yf_symbol, calculated_period, interval)

                    # Filter Yahoo data to exact date range
                    if not data.empty:
                        try:
                            start_dt_pd = pd.to_datetime(start_date)
                            end_dt_pd = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                            data = data[(data.index >= start_dt_pd) & (data.index < end_dt_pd)]
                            self._log(f"🔍 Filtered to exact date range: {len(data)} candles")
                        except Exception as e:
                            self._log(f"⚠️ Date filtering failed: {e}")

                return data

            except Exception as e:
                self._log(f"📅 Date parsing error: {e}, falling back to period: {period}")

        # No dates provided or date parsing failed - use period dropdown
        self._log(f"📅 Using backtest period: {period} (no specific dates)")

        # Try Binance/Yahoo Finance with period
        data = DataProvider.get_backtest_data_with_period(symbol, period, interval)

        return data

    def _simulate_trading(self, model: MLModel, data: pd.DataFrame, params: Dict, live_mode: bool = False):
        """Simulate trading on historical data - identical to BacktestEngine"""
        try:
            capital = params['START_CAPITAL']
            position = 0
            btc_amount = 0
            entry_price = 0
            trades = []
            equity_curve = []
            total_fees = 0

            self._log(f"💰 Starting simulation with ${capital:,.2f}")

            # For live mode, only process the latest candle
            if live_mode:
                data = data.tail(1)

            # Process data in smaller chunks to avoid hanging
            chunk_size = 100 if not live_mode else 1
            total_processed = 0

            for chunk_start in range(0, len(data), chunk_size):
                if not self.running:
                    break

                chunk_end = min(chunk_start + chunk_size, len(data))
                chunk_data = data.iloc[chunk_start:chunk_end]

                for i, (timestamp, candle) in enumerate(chunk_data.iterrows()):
                    if not self.running:
                        break

                    try:
                        price = float(candle['Close'])
                        equity = capital if position == 0 else btc_amount * price
                        equity_curve.append({'timestamp': str(timestamp), 'equity': equity})

                        # Get features for this candle
                        try:
                            available_features = [f for f in model.features if f in candle.index]
                            if len(available_features) != len(model.features):
                                logger.warning(f"Missing features: {set(model.features) - set(available_features)}")
                                continue

                            features = candle[model.features].values
                            prob, confidence = model.predict(features, is_live_trading=live_mode)
                            
                            # Log backtester predictions every 100 candles for comparison
                            if i % 100 == 0 and not live_mode:
                                logger.info(f"🔍 BACKTESTER PREDICTION #{i}: Prob={prob:.6f}, Conf={confidence:.6f}, Features={[f'{f:.6f}' for f in features]}")
                        except Exception as e:
                            logger.error(f"Feature extraction error for prediction: {e}")
                            continue

                        # Buy signal
                        if (position == 0 and 
                            prob > params['BUY_PROB_THRESHOLD'] and 
                            confidence > params['CONFIDENCE_THRESHOLD']):

                            fee = capital * params['TAKER_FEE_RATE']
                            btc_amount = (capital - fee) / price
                            entry_price = price
                            highest_price_since_entry = price
                            capital = fee
                            total_fees += fee
                            position = 1

                            if live_mode:
                                self._log(f"🚀 BUY: ${price:.2f} | Prob: {prob:.6f} | Conf: {confidence:.6f}")

                        # Sell signal
                        elif position == 1:
                            # Update highest price since entry
                            if price > highest_price_since_entry:
                                highest_price_since_entry = price

                            stop_loss = entry_price * (1 - params['STOP_LOSS_PCT'])
                            take_profit = entry_price * (1 + params['TAKE_PROFIT_PCT'])
                            trailing_stop = highest_price_since_entry * (1 - params.get('TRAIL_STOP_PCT', 0.02))

                            # Count how long we've been in this position
                            candles_in_position = getattr(self, '_candles_in_position', 0) + 1
                            self._candles_in_position = candles_in_position

                            # Track consecutive low probability signals
                            if not hasattr(self, '_low_prob_count'):
                                self._low_prob_count = 0

                            if prob < 0.3:
                                self._low_prob_count += 1
                            else:
                                self._low_prob_count = 0

                            exit_reason = None
                            if price <= stop_loss:
                                exit_reason = 'Stop Loss'
                            elif price >= take_profit:
                                exit_reason = 'Take Profit'
                            elif price <= trailing_stop and price >= entry_price * 1.001:
                                exit_reason = 'Trailing Stop'
                            elif (self._low_prob_count >= 3 and
                                  candles_in_position > 6 and
                                  prob < 0.25):
                                exit_reason = 'Signal Exit'

                            if exit_reason:
                                exit_fee = btc_amount * price * params['TAKER_FEE_RATE']
                                capital = btc_amount * price - exit_fee

                                # Calculate PnL including both entry and exit fees
                                entry_fee = capital * params['TAKER_FEE_RATE']
                                total_trade_fees = entry_fee + exit_fee
                                total_fees += total_trade_fees

                                # Calculate both raw P&L and P&L including fees
                                raw_pnl = (price - entry_price) * btc_amount
                                pnl = raw_pnl - total_trade_fees

                                trades.append({
                                    'timestamp': str(timestamp),
                                    'entry_price': entry_price,
                                    'exit_price': price,
                                    'exit_reason': exit_reason,
                                    'pnl': pnl,
                                    'raw_pnl': raw_pnl,
                                    'probability': prob,
                                    'confidence': confidence,
                                    'fees': total_trade_fees,
                                    'highest_price': highest_price_since_entry,
                                    'hold_duration': candles_in_position
                                })

                                # Log trade with both P&L values
                                trade_type = "WIN" if pnl > 0 else "LOSS"
                                log_msg = f"🔄 {trade_type}: ${entry_price:.2f} → ${price:.2f} | P&L: ${raw_pnl:.2f} | P&L including fees: ${pnl:.2f} | Fees: ${total_trade_fees:.2f}"
                                if live_mode:
                                    log_msg = f"🎯 SELL ({exit_reason}): ${price:.2f} | {trade_type}: ${pnl:.2f}"
                                self._log(log_msg)

                                position = 0
                                # Reset position tracking
                                self._candles_in_position = 0
                                self._low_prob_count = 0

                        total_processed += 1

                    except Exception as e:
                        logger.error(f"Error processing candle {i}: {e}")
                        continue

                # Log progress every chunk (only for backtesting)
                if not live_mode:
                    progress = (chunk_end / len(data)) * 100
                    self._log(f"📈 Processing... {progress:.1f}% complete ({len(trades)} trades so far)")

            # Calculate final metrics
            if trades:
                self._calculate_metrics(trades, params['START_CAPITAL'], total_fees)
            else:
                if not live_mode:
                    self._log("⚠️ No trades executed during backtest")
                self.results['metrics'] = {
                    'total_trades': 0,
                    'total_pnl': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'return_pct': 0,
                    'total_fees': 0
                }

            self.results['trades'] = trades
            self.results['equity_curve'] = equity_curve

            if not live_mode:
                self._log(f"✅ Backtest completed - {len(trades)} trades executed")

        except Exception as e:
            self._log(f"❌ Simulation error: {str(e)}")
            logger.error(f"Simulation error: {e}")
            traceback.print_exc()

    def _calculate_metrics(self, trades: List[Dict], start_capital: float, total_fees: float):
        """Calculate backtest performance metrics - identical to BacktestEngine"""
        try:
            if not trades:
                self.results['metrics'] = {}
                return

            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]

            winner_amount = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
            loser_amount = sum([t['pnl'] for t in losing_trades]) if losing_trades else 0
            total_pnl_with_fees = winner_amount + loser_amount

            # Calculate win/loss ratio
            win_loss_ratio = len(winning_trades) / len(losing_trades) if losing_trades else len(winning_trades)

            metrics = {
                'total_trades': len(trades),
                'winner_count': len(winning_trades),
                'winner_amount': winner_amount,
                'loser_count': len(losing_trades),
                'loser_amount': abs(loser_amount),
                'win_loss_ratio': win_loss_ratio,
                'total_pnl_with_fees': total_pnl_with_fees,
                'total_fees': total_fees
            }

            self.results['metrics'] = metrics

            # Log P&L including fees as header
            self._log(f"💰 === P&L INCLUDING FEES ===")
            self._log(f"💰 Total P&L (with fees): ${total_pnl_with_fees:.2f}")
            self._log(f"💰 Total Fees Paid: ${total_fees:.2f}")
            self._log(f"📊 Final Results: {len(trades)} trades, {len(winning_trades)} winners (${winner_amount:.2f}), {len(losing_trades)} losers (${abs(loser_amount):.2f})")
            self._log(f"📊 Win/Loss Ratio: {win_loss_ratio:.2f}")

        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            self.results['metrics'] = {}

    def _log(self, message: str):
        """Add log message - identical to BacktestEngine"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"{timestamp} | {message}"
        self.logs.append(log_entry)
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]
        logger.info(message)

    def get_status(self):
        """Get current trading status"""
        if hasattr(self, 'results') and self.results.get('trades'):
            latest_trade = self.results['trades'][-1] if self.results['trades'] else None
            total_pnl = sum([t['pnl'] for t in self.results['trades']])
            total_trades = len(self.results['trades'])
            
            return {
                'running': self.running,
                'status': 'Running - Live Trading' if self.running else 'Stopped',
                'current_price': latest_trade['exit_price'] if latest_trade else 0,
                'position': 'Cash',
                'pnl': total_pnl,
                'total_trades': total_trades,
                'capital': self.params.get('START_CAPITAL', 10000) + total_pnl
            }
        else:
            return {
                'running': self.running,
                'status': 'Running - Live Trading' if self.running else 'Stopped',
                'current_price': 0,
                'position': 'Cash',
                'pnl': 0,
                'total_trades': 0,
                'capital': self.params.get('START_CAPITAL', 10000) if self.params else 10000
            }

# Flask Application
app = Flask(__name__)
backtest_engine = BacktestEngine()
config = TradingConfig.DEFAULT_PARAMS.copy()
settings_manager = SettingsManager()

# Live trading variables
live_trader = None
live_trading_running = False
latest_backtest_settings = None

current_trained_model = None
latest_model_info = {
    'training_accuracy': 0.0,
    'test_accuracy': 0.0,
    'validation_accuracy': 0.0,
    'hit_rate': 0.0,
    'model_type': 'Not Trained',
    'features_count': 0,
    'class_distribution': {}
}

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/backtest')
def backtest_page():
    return render_template('backtest.html')

@app.route('/status')
def get_status():
    global latest_model_info

    return jsonify({
        'running': False,
        'capital': config.get('START_CAPITAL', 10000),
        'current_price': 0,
        'position': 0,
        'pnl': 0,
        'total_trades': 0,
        'model_confidence': 0,
        'model_type': latest_model_info.get('model_type', 'Not Trained'),
        'last_trade': 'None',
        'training_accuracy': latest_model_info.get('training_accuracy', 0.0),
        'test_accuracy': latest_model_info.get('test_accuracy', 0.0),
        'validation_accuracy': latest_model_info.get('validation_accuracy', 0.0),
        'hit_rate': latest_model_info.get('hit_rate', 0.0),
        'features_count': latest_model_info.get('features_count', 0),
        'signal_exit_threshold': config.get('SIGNAL_EXIT_THRESHOLD', 0.25),
        'signal_exit_consecutive': config.get('SIGNAL_EXIT_CONSECUTIVE', 3),
        'min_hold_candles': config.get('MIN_HOLD_CANDLES', 6),
        'total_return_pct': 0,
        'win_rate': 0,
        'avg_win': 0,
        'avg_loss': 0
    })

@app.route('/params')
def get_params():
    return jsonify(config)

@app.route('/update_params', methods=['POST'])
def update_params():
    global config
    try:
        data = request.json
        for key, value in data.items():
            if key in config:
                config[key] = value
        return jsonify({'status': 'Parameters updated'})
    except Exception as e:
        return jsonify({'status': f'Error: {str(e)}'})

@app.route('/trades')
def get_trades():
    """Get recent trades"""
    return jsonify([])

@app.route('/start_backtest', methods=['POST'])
def start_backtest():
    global latest_backtest_settings
    try:
        data = request.json
        params = data.get('params', config)
        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')

        # Store the latest backtest settings for live trading
        latest_backtest_settings = params.copy()

        if backtest_engine.running:
            return jsonify({'status': 'Backtest already running'})

        def run_backtest():
            backtest_engine.run(params, start_date, end_date)

        threading.Thread(target=run_backtest, daemon=True).start()
        return jsonify({'status': 'Backtest started'})

    except Exception as e:
        logger.error(f"Start backtest error: {e}")
        return jsonify({'status': f'Error: {str(e)}'})

@app.route('/stop_backtest', methods=['POST'])
def stop_backtest():
    try:
        backtest_engine.stop()
        return jsonify({'status': 'Backtest stopped'})
    except Exception as e:
        return jsonify({'status': f'Error: {str(e)}'})

@app.route('/backtest_results')
def get_backtest_results():
    try:
        return jsonify(backtest_engine.results)
    except Exception as e:
        logger.error(f"Get results error: {e}")
        return jsonify({'error': str(e)})

@app.route('/backtest_logs')
def get_backtest_logs():
    try:
        return jsonify({'logs': backtest_engine.logs})
    except Exception as e:
        logger.error(f"Get logs error: {e}")
        return jsonify({'logs': [f"Error getting logs: {str(e)}"]})

@app.route('/reset_backtest', methods=['POST'])
def reset_backtest():
    try:
        backtest_engine.stop()
        backtest_engine.results = {'trades': [], 'metrics': {}, 'equity_curve': [], 'running': False}
        backtest_engine.logs = ['Ready to start backtest...']
        backtest_engine.running = False

        logger.info("Backtest reset successfully")
        return jsonify({'status': 'Backtest reset successfully'})
    except Exception as e:
        logger.error(f"Reset backtest error: {e}")
        return jsonify({'status': f'Error: {str(e)}'})

@app.route('/save_model', methods=['POST'])
def save_current_model():
    """Save the currently trained model"""
    global current_trained_model
    try:
        if current_trained_model is None or not current_trained_model.is_trained:
            return jsonify({'status': 'error', 'message': 'No trained model available to save'})

        data = request.json or {}
        custom_name = data.get('filename', None)

        if custom_name:
            if not custom_name.endswith('.pkl'):
                custom_name += '.pkl'
            filepath = f"models/{custom_name}"
        else:
            filepath = None

        saved_path = current_trained_model.save_model(filepath)
        return jsonify({
            'status': 'success', 
            'message': f'Model saved successfully to {saved_path}',
            'filepath': saved_path
        })

    except Exception as e:
        logger.error(f"Save model error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/load_model', methods=['POST'])
def load_saved_model():
    """Load a saved model"""
    global current_trained_model, latest_model_info
    try:
        data = request.json
        filepath = data.get('filepath')

        if not filepath:
            return jsonify({'status': 'error', 'message': 'No filepath provided'})

        # Create new model instance and load
        model = MLModel()
        success = model.load_model(filepath)

        if success:
            current_trained_model = model
            # Update latest_model_info
            model_info = model.get_model_info()
            latest_model_info.update({
                'training_accuracy': model_info.get('training_accuracy', 0.0),
                'test_accuracy': model_info.get('test_accuracy', 0.0),
                'validation_accuracy': model_info.get('validation_accuracy', 0.0),
                'hit_rate': model_info.get('hit_rate', 0.0),
                'model_type': model_info.get('model_type', 'Unknown'),
                'features_count': model_info.get('n_features', 0),
                'class_distribution': model_info.get('class_distribution', {})
            })

            return jsonify({
                'status': 'success', 
                'message': f'Model loaded successfully from {filepath}',
                'model_info': model_info
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to load model'})

    except Exception as e:
        logger.error(f"Load model error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/list_models')
def list_saved_models():
    """List all saved models"""
    try:
        models = MLModel.list_saved_models()
        return jsonify({'models': models})
    except Exception as e:
        logger.error(f"List models error: {e}")
        return jsonify({'error': str(e)})

@app.route('/delete_model', methods=['POST'])
def delete_saved_model():
    """Delete a saved model"""
    try:
        data = request.json
        filepath = data.get('filepath')

        if not filepath or not os.path.exists(filepath):
            return jsonify({'status': 'error', 'message': 'Model file not found'})

        os.remove(filepath)
        return jsonify({'status': 'success', 'message': f'Model deleted: {filepath}'})

    except Exception as e:
        logger.error(f"Delete model error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/save_settings', methods=['POST'])
def save_current_settings():
    """Save current parameters to a settings file"""
    global config, settings_manager
    try:
        data = request.json or {}
        custom_name = data.get('filename', None)

        saved_path = settings_manager.save_settings(config, custom_name)
        return jsonify({
            'status': 'success', 
            'message': f'Settings saved successfully to {saved_path}',
            'filepath': saved_path
        })

    except Exception as e:
        logger.error(f"Save settings error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/load_settings', methods=['POST'])
def load_saved_settings():
    """Load settings from a file"""
    global config, settings_manager
    try:
        data = request.json
        filepath = data.get('filepath')

        if not filepath:
            return jsonify({'status': 'error', 'message': 'No filepath provided'})

        loaded_settings = settings_manager.load_settings(filepath)

        # Update global config
        config.update(loaded_settings)

        return jsonify({
            'status': 'success', 
            'message': f'Settings loaded successfully from {filepath}',
            'settings': loaded_settings
        })

    except Exception as e:
        logger.error(f"Load settings error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/list_settings')
def list_saved_settings():
    """List all saved settings"""
    try:
        settings = settings_manager.list_saved_settings()
        return jsonify({'settings': settings})
    except Exception as e:
        logger.error(f"List settings error: {e}")
        return jsonify({'error': str(e)})

@app.route('/delete_settings', methods=['POST'])
def delete_saved_settings():
    """Delete a saved settings file"""
    try:
        data = request.json
        filepath = data.get('filepath')

        if not filepath:
            return jsonify({'status': 'error', 'message': 'No filepath provided'})

        success = settings_manager.delete_settings(filepath)

        if success:
            return jsonify({'status': 'success', 'message': f'Settings deleted: {filepath}'})
        else:
            return jsonify({'status': 'error', 'message': 'Settings file not found'})

    except Exception as e:
        logger.error(f"Delete settings error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/save_backtest_settings', methods=['POST'])
def save_backtest_settings():
    """Save backtest parameters to a settings file"""
    global settings_manager
    try:
        data = request.json or {}
        custom_name = data.get('filename', None)
        params = data.get('params', {})

        if not params:
            return jsonify({'status': 'error', 'message': 'No parameters provided'})

        saved_path = settings_manager.save_settings(params, custom_name)
        return jsonify({
            'status': 'success', 
            'message': f'Backtest settings saved successfully to {saved_path}',
            'filepath': saved_path
        })

    except Exception as e:
        logger.error(f"Save backtest settings error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/create_project_zip', methods=['POST'])
def create_project_zip():
    """Create a ZIP file containing all project files"""
    try:
        import zipfile
        from io import BytesIO
        from flask import send_file

        data = request.json or {}
        custom_filename = data.get('filename', None)

        if custom_filename:
            if not custom_filename.endswith('.zip'):
                custom_filename += '.zip'
            zip_filename = custom_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"bitcoin_trading_bot_project_{timestamp}.zip"

        # Create ZIP file in memory
        zip_buffer = BytesIO()
        files_added = 0

        # Define files and directories to exclude
        exclude_dirs = {
            '__pycache__', '.git', '.replit', 'node_modules', '.env', 
            'venv', '.venv', '.DS_Store', 'Thumbs.db', '.pytest_cache',
            'attached_assets', 'bitcoin_bot_extracted'
        }

        exclude_files = {
            'uv.lock', 'generated-icon.png', 'bitcoin_trading_bot_duplicate.zip', '.replit'
        }

        logger.info(f"🔧 Creating ZIP file: {zip_filename}")

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
            # Add main project files first
            main_files = ['main.py', 'pyproject.toml']
            for main_file in main_files:
                if os.path.exists(main_file):
                    zip_file.write(main_file, main_file)
                    files_added += 1
                    logger.info(f"✅ Added: {main_file}")

            # Add templates directory
            if os.path.exists('templates'):
                for root, dirs, files in os.walk('templates'):
                    for file in files:
                        if file.endswith(('.html', '.css', '.js')):
                            file_path = os.path.join(root, file)
                            archive_path = file_path.replace('\\', '/')
                            zip_file.write(file_path, archive_path)
                            files_added += 1
                            logger.info(f"✅ Added: {archive_path}")

            # Add models directory if it exists
            if os.path.exists('models'):
                for root, dirs, files in os.walk('models'):
                    for file in files:
                        if file.endswith('.pkl'):
                            file_path = os.path.join(root, file)
                            archive_path = file_path.replace('\\', '/')
                            zip_file.write(file_path, archive_path)
                            files_added += 1
                            logger.info(f"✅ Added: {archive_path}")

            # Add settings directory if it exists
            if os.path.exists('settings'):
                for root, dirs, files in os.walk('settings'):
                    for file in files:
                        if file.endswith('.json'):
                            file_path = os.path.join(root, file)
                            archive_path = file_path.replace('\\', '/')
                            zip_file.write(file_path, archive_path)
                            files_added += 1
                            logger.info(f"✅ Added: {archive_path}")

            # Add any Python utility files
            for file in os.listdir('.'):
                if (file.endswith('.py') and 
                    file not in exclude_files and 
                    file != 'main.py' and
                    os.path.isfile(file)):
                    zip_file.write(file, file)
                    files_added += 1
                    logger.info(f"✅ Added: {file}")

        zip_buffer.seek(0)

        if files_added == 0:
            logger.error("❌ No files were added to ZIP")
            return jsonify({'status': 'error', 'message': 'No files found to zip'}), 400

        logger.info(f"✅ ZIP created successfully: {zip_filename} ({files_added} files)")

        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )

    except Exception as e:
        logger.error(f"❌ ZIP creation error: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_latest_settings')
def get_latest_settings():
    """Get the latest backtest settings for live trading"""
    global latest_backtest_settings
    try:
        if latest_backtest_settings:
            return jsonify({
                'status': 'success',
                'settings': latest_backtest_settings
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'No backtest settings found. Please run a backtest first.'
            })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/start_live_trading', methods=['POST'])
def start_live_trading():
    """Start live trading - simplified version using LiveTrader"""
    global live_trader, live_trading_running, current_trained_model
    try:
        data = request.json
        params = data.get('params', {})

        if live_trading_running:
            return jsonify({'status': 'error', 'message': 'Live trading already running'})

        if not current_trained_model or not current_trained_model.is_trained:
            return jsonify({'status': 'error', 'message': 'No trained model available. Please run a backtest first.'})

        # Create and start simple live trader
        live_trader = LiveTrader()
        live_trader.start(params, current_trained_model)
        live_trading_running = True

        logger.info("Live trading started")
        return jsonify({
            'status': 'success',
            'message': 'Live trading started successfully'
        })

    except Exception as e:
        logger.error(f"Start live trading error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_live_trading', methods=['POST'])
def stop_live_trading():
    """Stop live trading"""
    global live_trader, live_trading_running
    try:
        if live_trader:
            live_trader.stop()
        live_trading_running = False

        logger.info("Live trading stopped")
        return jsonify({
            'status': 'success',
            'message': 'Live trading stopped'
        })

    except Exception as e:
        logger.error(f"Stop live trading error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/live_trading_status')
def get_live_trading_status():
    """Get live trading status"""
    global live_trader, live_trading_running
    try:
        if live_trader and live_trading_running:
            status = live_trader.get_status()
            return jsonify(status)
        else:
            return jsonify({
                'running': False,
                'status': 'Stopped',
                'current_price': 0,
                'position': 'Cash',
                'pnl': 0,
                'total_trades': 0,
                'capital': 0
            })

    except Exception as e:
        logger.error(f"Get live trading status error: {e}")
        return jsonify({'running': False, 'error': str(e)})

@app.route('/live_trading_logs')
def get_live_trading_logs():
    """Get live trading logs"""
    global live_trader
    try:
        if live_trader and hasattr(live_trader, 'logs'):
            return jsonify({'logs': live_trader.logs})
        else:
            return jsonify({'logs': ['No live trading session active']})
    except Exception as e:
        logger.error(f"Get live trading logs error: {e}")
        return jsonify({'logs': [f"Error getting logs: {str(e)}"]})

@app.route('/ping')
def ping():
    """Simple ping endpoint"""
    try:
        return jsonify({
            'status': 'alive',
            'timestamp': datetime.now().isoformat(),
            'server': 'Bitcoin Trading Bot',
            'uptime': 'running'
        })
    except Exception as e:
        return jsonify({'status': 'alive', 'error': str(e)})

if __name__ == '__main__':
    logger.info("🚀 Starting Bitcoin Trading Bot Server")
    app.run(host='0.0.0.0', port=5000, debug=True)