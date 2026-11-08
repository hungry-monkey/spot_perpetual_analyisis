import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats
import logging
import os
import re
from typing import Dict, Union, Any, Optional, Tuple, List, Set
from numpy.typing import DTypeLike

class HFTLeadLagAnalyzer:
    def __init__(self, threshold_bps: float = 7, impact_window_ms: int = 5):
        """
        Initialize analyzer with configurable thresholds.
        
        Args:
            threshold_bps: Price movement threshold in basis points (0.07% default)
            impact_window_ms: Time window to measure impact effects (5ms default)
        """
        self.THRESHOLD = threshold_bps / 10000  # Convert bps to decimal
        self.IMPACT_WINDOW = pd.Timedelta(impact_window_ms, 'ms')
        self.SUSTAIN_WINDOWS = [10, 20, 50]  # Additional analysis windows in ms
        self.logger = self._setup_logging()
        self.required_columns = {
            'time', 
            'best_bid_spot', 'best_ask_spot',
            'best_bid_perp', 'best_ask_perp'
        }

    def _setup_logging(self):
        """Configure application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hft_analysis.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flexible timestamp parser that handles various formats"""
        if 'time' not in df.columns:
            # Try to find a time-like column
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
            if time_cols:
                self.logger.warning(f"Using '{time_cols[0]}' as time column")
                df = df.rename(columns={time_cols[0]: 'time'})
            else:
                raise ValueError("No time column found in dataset")
        
        df['time'] = pd.to_datetime(
            df['time'],
            format='ISO8601',
            errors='coerce',
            utc=True
        )
        
        # Verify we didn't get any NaT values
        if df['time'].isna().any():
            bad_count = df['time'].isna().sum()
            self.logger.warning(f"Found {bad_count} problematic timestamps")
            df = df.dropna(subset=['time'])
        
        return df

    def _standardize_column_names(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Standardize column names to our expected format"""
        # Create a mapping of possible column names to standardized names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            
            # Handle time column
            if 'time' in col_lower or 'timestamp' in col_lower:
                column_mapping[col] = 'time'
                continue
                
            # Map bid/ask columns
            if 'bid' in col_lower and 'ask' not in col_lower:
                if 'qty' in col_lower or 'size' in col_lower or 'vol' in col_lower:
                    column_mapping[col] = f'best_bid_qty_{market}'
                elif 'price' in col_lower or 'px' in col_lower:
                    column_mapping[col] = f'best_bid_{market}'
                else:
                    column_mapping[col] = f'best_bid_{market}'
                    
            elif 'ask' in col_lower and 'bid' not in col_lower:
                if 'qty' in col_lower or 'size' in col_lower or 'vol' in col_lower:
                    column_mapping[col] = f'best_ask_qty_{market}'
                elif 'price' in col_lower or 'px' in col_lower:
                    column_mapping[col] = f'best_ask_{market}'
                else:
                    column_mapping[col] = f'best_ask_{market}'
        
        # Apply the mapping
        df = df.rename(columns=column_mapping)
        
        # Log any columns we couldn't map
        unmapped = [c for c in df.columns if c not in column_mapping.values() and c != 'time']
        if unmapped:
            self.logger.warning(f"Unmapped columns for {market}: {unmapped}")
            
        return df

    def load_and_preprocess(self, spot_path: str, perp_path: str, trade_path: Optional[str] = None) -> pd.DataFrame:
        """Load and synchronize datasets with nanosecond precision"""
        try:
            # Verify files exist
            if not os.path.exists(spot_path):
                raise FileNotFoundError(f"Spot file not found: {spot_path}")
            if not os.path.exists(perp_path):
                raise FileNotFoundError(f"Perp file not found: {perp_path}")
            if trade_path and not os.path.exists(trade_path):
                raise FileNotFoundError(f"Trade file not found: {trade_path}")

            # Load spot data
            spot = pd.read_csv(spot_path)
            self.logger.info(f"Original spot columns: {spot.columns.tolist()}")
            
            # Standardize column names
            spot = self._standardize_column_names(spot, 'spot')
            self.logger.info(f"Standardized spot columns: {spot.columns.tolist()}")
            
            # Load perp data
            perp = pd.read_csv(perp_path)
            self.logger.info(f"Original perp columns: {perp.columns.tolist()}")
            
            # Standardize column names
            perp = self._standardize_column_names(perp, 'perp')
            self.logger.info(f"Standardized perp columns: {perp.columns.tolist()}")

            # Convert timestamps with flexible parser
            spot = self._parse_timestamps(spot)
            perp = self._parse_timestamps(perp)

            # Merge with sub-millisecond precision
            merged = pd.merge_asof(
                spot.sort_values('time'),
                perp.sort_values('time'),
                on='time',
                direction='nearest',
                tolerance=pd.Timedelta('500us'),
                suffixes=('', '_y')
            )

            # Drop any duplicate columns from the merge
            merged = merged.loc[:, ~merged.columns.duplicated()]

            # Handle trade data if provided
            if trade_path:
                trades = pd.read_csv(trade_path)
                trades = self._parse_timestamps(trades)
                trades = trades.set_index('time').resample('1ms').agg({
                    'price': ['mean', 'std'],
                    'quantity': 'sum'
                })
                trades.columns = ['trade_price', 'trade_std', 'trade_volume']
                merged = merged.merge(
                    trades,
                    left_on='time',
                    right_index=True,
                    how='left'
                )

            # Validate data quality
            self._validate_data(merged)
            
            return merged.dropna().reset_index(drop=True)
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise

    def _validate_data(self, df: pd.DataFrame):
        """Perform critical data validation checks with flexible column matching"""
        # Find best matching columns for our requirements
        actual_cols = set(df.columns)
        found_cols = set()
        
        # Check for each required column pattern
        for pattern in self.required_columns:
            # Look for exact match first
            if pattern in actual_cols:
                found_cols.add(pattern)
                continue
                
            # Look for case-insensitive match
            pattern_lower = pattern.lower()
            matches = [col for col in actual_cols if pattern_lower in col.lower()]
            
            if matches:
                self.logger.info(f"Using '{matches[0]}' as '{pattern}'")
                # Rename to standard name
                df.rename(columns={matches[0]: pattern}, inplace=True)
                found_cols.add(pattern)
        
        # Check if we found all required columns
        missing = self.required_columns - found_cols
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            self.logger.error(f"Actual columns: {actual_cols}")
            raise ValueError(f"Missing required columns: {missing}")
            
        if not df['time'].is_monotonic_increasing:
            self.logger.warning("Timestamps not strictly increasing - sorting")
            df.sort_values('time', inplace=True)
            
        if len(df) < 1000:
            raise ValueError(f"Insufficient data points: {len(df)}")
            
        # Ensure numeric columns are actually numeric
        for col in self.required_columns:
            if col != 'time':
                df[col] = pd.to_numeric(df[col], errors='coerce')

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate microstructure features with lag/lead indicators"""
        try:
            # Ensure chronological order
            df = df.sort_values('time').reset_index(drop=True)
            
            # Core price features
            for market in ['spot', 'perp']:
                # Mid price and weighted mid
                df[f'mid_{market}'] = (df[f'best_bid_{market}'] + df[f'best_ask_{market}']) / 2
                df[f'wmid_{market}'] = (
                    df[f'best_bid_{market}'] * df[f'best_ask_qty_{market}'] + 
                    df[f'best_ask_{market}'] * df[f'best_bid_qty_{market}']
                ) / (df[f'best_bid_qty_{market}'] + df[f'best_ask_qty_{market}'])
                
                # Spread and depth imbalance
                df[f'spread_{market}'] = df[f'best_ask_{market}'] - df[f'best_bid_{market}']
                df[f'depth_imbalance_{market}'] = (
                    df[f'best_bid_qty_{market}'] - df[f'best_ask_qty_{market}']
                ) / (df[f'best_bid_qty_{market}'] + df[f'best_ask_qty_{market}'])
                
                # Microstructure momentum (bid vs ask)
                for lookback in [1, 3, 5]:  # Milliseconds
                    df[f'bid_mom_{market}_{lookback}ms'] = df[f'best_bid_{market}'].pct_change(lookback)
                    df[f'ask_mom_{market}_{lookback}ms'] = df[f'best_ask_{market}'].pct_change(lookback)
                    df[f'mid_mom_{market}_{lookback}ms'] = df[f'mid_{market}'].pct_change(lookback)
            
            # Cross-market features
            df['mid_spread'] = df['mid_spot'] - df['mid_perp']
            df['wmid_ratio'] = df['wmid_spot'] / df['wmid_perp']
            
            # Trade-induced features (if available)
            if 'trade_price' in df.columns:
                df['trade_vwap'] = df['trade_price'].rolling(10, min_periods=1).mean()
                df['trade_impact'] = (df['trade_price'] - df['mid_spot']) / df['mid_spot']
                
            # Lagged features for prediction
            for lag in [1, 2, 3]:  # 1ms, 2ms, 3ms lags
                df[f'mid_spot_lag{lag}'] = df['mid_spot'].shift(lag)
                df[f'mid_perp_lag{lag}'] = df['mid_perp'].shift(lag)
            
            return df.dropna()
            
        except Exception as e:
            self.logger.error(f"Feature creation failed: {str(e)}")
            raise

    def detect_jumps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify significant price movements with directional labeling"""
        try:
            jumps = []
            for market in ['spot', 'perp']:
                for side in ['bid', 'ask']:
                    price_col = f'best_{side}_{market}'
                    
                    # Skip if column doesn't exist
                    if price_col not in df.columns:
                        continue
                        
                    # Calculate rolling changes with 3ms lookback
                    changes = df[price_col].pct_change(3).rename('price_change')
                    
                    # Detect threshold crossings
                    mask = (abs(changes) > self.THRESHOLD)
                    if not mask.any():
                        continue
                        
                    # Label jump direction
                    jump_df = df.loc[mask].copy()
                    jump_df['jump_direction'] = np.where(
                        changes[mask] > 0, 
                        f'{market}_{side}_up', 
                        f'{market}_{side}_down'
                    )
                    jump_df['jump_magnitude'] = changes[mask]
                    jump_df['jump_type'] = market
                    
                    # Microstructure context
                    jump_df['init_spread'] = jump_df[f'spread_{market}']
                    jump_df['init_imbalance'] = jump_df[f'depth_imbalance_{market}']
                    
                    jumps.append(jump_df)
            
            if not jumps:
                self.logger.warning("No significant jumps detected - adjust threshold")
                return pd.DataFrame()
                
            return pd.concat(jumps).sort_values('time')
            
        except Exception as e:
            self.logger.error(f"Jump detection failed: {str(e)}")
            raise

    def analyze_impact(self, jumps: pd.DataFrame, full_data: pd.DataFrame) -> pd.DataFrame:
        """Measure cross-market impact of jumps with microsecond precision"""
        results = []
        for _, jump in jumps.iterrows():
            try:
                # Get precise post-jump window
                window_end = jump['time'] + self.IMPACT_WINDOW
                post_data = full_data[
                    (full_data['time'] > jump['time']) & 
                    (full_data['time'] <= window_end)
                ]
                
                if len(post_data) == 0:
                    continue
                    
                # Determine target market (opposite of jump origin)
                target_market = 'perp' if 'spot' in jump['jump_type'] else 'spot'
                price_col = f'mid_{target_market}'
                
                # Calculate impact metrics
                init_price = jump[f'mid_{target_market}']
                final_price = post_data[price_col].iloc[-1]
                max_price = post_data[price_col].max() if 'up' in jump['jump_direction'] else post_data[price_col].min()
                
                impact = {
                    'jump_time': jump['time'],
                    'jump_type': jump['jump_type'],
                    'jump_direction': jump['jump_direction'],
                    'target_market': target_market,
                    'init_price': init_price,
                    'final_price': final_price,
                    'max_price': max_price,
                    'impact_delta': final_price - init_price,
                    'max_delta': max_price - init_price,
                    'duration_ns': (post_data['time'].iloc[-1] - jump['time']).total_seconds() * 1e9,
                    'init_spread': jump['init_spread'],
                    'init_imbalance': jump['init_imbalance']
                }
                
                # Add trade metrics if available
                if 'trade_volume' in jump:
                    impact.update({
                        'trade_volume': jump['trade_volume'],
                        'trade_vwap': jump.get('trade_vwap', np.nan)
                    })
                
                results.append(impact)
                
            except Exception as e:
                self.logger.warning(f"Skipping jump at {jump['time']}: {str(e)}")
                continue
                
        return pd.DataFrame(results)

    def analyze_price_discovery(self, df: pd.DataFrame) -> Dict[str, Union[Dict[str, Any], str]]:
        """Quantify which market leads price discovery with proper Granger causality typing"""
        try:
            # Sample data
            sampled = df.set_index('time').resample('1ms').last().ffill()
            
            def calculate_lead_lag(main_col: str, ref_col: str) -> Dict[str, Any]:
                # Granger causality returns complex nested structure
                gc_results = grangercausalitytests(
                    sampled[[main_col, ref_col]].dropna(),
                    maxlag=5,
                    verbose=False
                )
                
                # Extract p-values from the nested results
                p_values = [gc_results[i+1][0]['ssr_ftest'][1] for i in range(5)]
                
                return {
                    'best_lag_ms': float(np.argmax(p_values) + 1),  # Convert to float
                    'granger_pvalues': [float(p) for p in p_values],  # List of floats
                    'mean_pvalue': float(np.mean(p_values))
                }
            
            spot_lead = calculate_lead_lag('mid_spot', 'mid_perp')
            perp_lead = calculate_lead_lag('mid_perp', 'mid_spot')
            
            return {
                'spot_leading_metrics': spot_lead,
                'perp_leading_metrics': perp_lead,
                'primary_leader': 'spot' if spot_lead['mean_pvalue'] < perp_lead['mean_pvalue'] else 'perp'
            }
            
        except Exception as e:
            return {'error': str(e)}

    def analyze_momentum(self, impacts: pd.DataFrame) -> Dict[str, Union[Dict[str, Union[int, float, Dict[str, float]]], str]]:
        """Analyze momentum with proper return type hints."""
        results: Dict[str, Union[Dict[str, Union[int, float, Dict[str, float]]], str]] = {}
        
        try:
            for direction in ['up', 'down']:
                dir_impacts = impacts[impacts['jump_direction'].str.contains(direction)]
                
                if len(dir_impacts) == 0:
                    continue
                    
                # Momentum decay analysis
                decay_metrics = []
                for _, row in dir_impacts.iterrows():
                    max_delta = float(row['max_delta'])
                    impact_delta = float(row['impact_delta'])
                    
                    decay_metrics.append({
                        'init_magnitude': abs(max_delta),
                        'sustained_5ms': abs(impact_delta) / abs(max_delta) if max_delta != 0 else 0.0,
                        'decay_ratio': 1 - (abs(impact_delta) / abs(max_delta)) if max_delta != 0 else 0.0,
                        'duration_ns': float(row['duration_ns'])
                    })
                
                decay_df = pd.DataFrame(decay_metrics)
                results[f'{direction}_moves'] = {
                    'count': int(len(dir_impacts)),
                    'mean_magnitude': float(decay_df['init_magnitude'].mean()),
                    'mean_sustained': float(decay_df['sustained_5ms'].mean()),
                    'decay_stats': {
                        'mean': float(decay_df['decay_ratio'].mean()),
                        'std': float(decay_df['decay_ratio'].std()),
                        'min': float(decay_df['decay_ratio'].min()),
                        'max': float(decay_df['decay_ratio'].max())
                    }
                }
            
            return results
                
        except Exception as e:
            self.logger.error(f"Momentum analysis failed: {str(e)}")
            return {'error': str(e)}

    def detect_noise_events(self, jumps: pd.DataFrame, impacts: pd.DataFrame) -> pd.DataFrame:
        """Identify and characterize noise events (failed propagations)"""
        try:
            merged = pd.merge(
                jumps[['time', 'jump_type', 'jump_direction', 'jump_magnitude']],
                impacts[['jump_time', 'impact_delta', 'max_delta']],
                left_on='time',
                right_on='jump_time',
                how='left'
            )
            
            # Noise criteria:
            # 1. Small final impact relative to initial jump
            # 2. Reversion pattern (opposite sign)
            merged['is_noise'] = (
                (abs(merged['impact_delta']) < 0.5 * abs(merged['jump_magnitude'])) |
                (np.sign(merged['impact_delta']) != np.sign(merged['jump_magnitude'])))
            
            return merged[merged['is_noise']].drop(columns='is_noise')
            
        except Exception as e:
            self.logger.error(f"Noise detection failed: {str(e)}")
            return pd.DataFrame()

    def build_models(self, features: pd.DataFrame) -> Dict:
        """Build and validate predictive models for both directions"""
        try:
            # Feature selection
            spot_features = [f for f in features.columns if ('perp_' in f and ('mom' in f or 'imbalance' in f)) or f.startswith('trade_')]
            perp_features = [f for f in features.columns if ('spot_' in f and ('mom' in f or 'imbalance' in f)) or f.startswith('trade_')]
            
            # Temporal split (last 20% for testing)
            cutoff = features['time'].quantile(0.8)
            train = features[features['time'] < cutoff]
            test = features[features['time'] >= cutoff]
            
            if len(train) < 1000 or len(test) < 100:
                raise ValueError(f"Insufficient data for modeling: {len(train)} train, {len(test)} test")
            
            # Model pipeline with embedded feature selection
            def make_pipeline():
                return Pipeline([
                    ('scaler', RobustScaler()),
                    ('selector', SelectKBest(f_regression, k=10)),
                    ('model', HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.05,
                        max_depth=5,
                        early_stopping=True,
                        validation_fraction=0.2,
                        random_state=42
                    ))
                ])
            
            # Spot prediction model (perp -> spot)
            spot_pipe = make_pipeline()
            spot_pipe.fit(train[spot_features], train['mid_spot'])
            spot_pred = spot_pipe.predict(test[spot_features])
            
            # Perp prediction model (spot -> perp)
            perp_pipe = make_pipeline()
            perp_pipe.fit(train[perp_features], train['mid_perp'])
            perp_pred = perp_pipe.predict(test[perp_features])
            
            # Calculate metrics
            def calc_metrics(y_true, y_pred):
                errors = y_pred - y_true
                return {
                    'mae': np.mean(np.abs(errors)),
                    'rmse': np.sqrt(np.mean(errors**2)),
                    'r2': 1 - (np.sum(errors**2) / np.sum((y_true - np.mean(y_true))**2)),
                    'precision_5bps': np.mean(np.sign(y_pred - y_true.shift(1)) == np.sign(y_true - y_true.shift(1)))
                }
            
            return {
                'spot_model': {
                    'pipeline': spot_pipe,
                    'features': spot_features,
                    'metrics': calc_metrics(test['mid_spot'], spot_pred),
                    'feature_importance': dict(zip(
                        spot_features,
                        spot_pipe.named_steps['selector'].scores_
                    ))
                },
                'perp_model': {
                    'pipeline': perp_pipe,
                    'features': perp_features,
                    'metrics': calc_metrics(test['mid_perp'], perp_pred),
                    'feature_importance': dict(zip(
                        perp_features,
                        perp_pipe.named_steps['selector'].scores_
                    ))
                },
                'train_period': (train['time'].min(), train['time'].max()),
                'test_period': (test['time'].min(), test['time'].max())
            }
            
        except Exception as e:
            self.logger.error(f"Model building failed: {str(e)}")
            return {'error': str(e)}

    def full_analysis(self, spot_path: str, perp_path: str, trade_path: Optional[str] = None) -> Dict:
        """Complete analysis pipeline with error handling"""
        analysis = {}
        
        try:
            # 1. Data loading
            self.logger.info("Loading and preprocessing data...")
            data = self.load_and_preprocess(spot_path, perp_path, trade_path)
            analysis['data_stats'] = {
                'start_time': str(data['time'].min()),
                'end_time': str(data['time'].max()),
                'num_observations': len(data),
                'columns': list(data.columns)
            }
            
            # 2. Feature engineering
            self.logger.info("Creating features...")
            features = self.create_features(data)
            analysis['feature_summary'] = {
                'num_features': len(features.columns),
                'feature_types': str(features.dtypes.value_counts())
            }
            
            # 3. Jump detection
            self.logger.info("Detecting price jumps...")
            jumps = self.detect_jumps(features)
            analysis['jump_stats'] = {
                'total_jumps': len(jumps),
                'by_type': jumps['jump_type'].value_counts().to_dict() if not jumps.empty else {},
                'by_direction': jumps['jump_direction'].value_counts().to_dict() if not jumps.empty else {}
            }
            
            # 4. Impact analysis
            self.logger.info("Analyzing market impact...")
            impacts = self.analyze_impact(jumps, features)
            if not impacts.empty:
                analysis['impact_stats'] = {
                    'mean_impact': impacts['impact_delta'].mean(),
                    'max_impact': impacts['impact_delta'].max(),
                    'success_rate': len(impacts[impacts['impact_delta'].abs() > self.THRESHOLD/2]) / len(impacts)
                }
            else:
                analysis['impact_stats'] = "No significant impacts detected"
            
            # 5. Secondary analyses
            self.logger.info("Running secondary analyses...")
            analysis['price_discovery'] = self.analyze_price_discovery(features)
            analysis['momentum_analysis'] = self.analyze_momentum(impacts) if not impacts.empty else "No momentum data"
            noise_events = self.detect_noise_events(jumps, impacts)
            analysis['noise_events'] = {
                'count': len(noise_events),
                'examples': noise_events.head(3).to_dict('records') if not noise_events.empty else []
            }
            
            # 6. Model building
            self.logger.info("Building predictive models...")
            analysis['models'] = self.build_models(features)
            
            self.logger.info("Analysis completed successfully")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Analysis pipeline failed: {str(e)}")
            analysis['error'] = str(e)
            return analysis