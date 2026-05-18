import logging
import warnings
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

# Statistical and Machine Learning Forecasting Engines
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Silence mathematical optimization and convergence warnings for clean logs
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WorkforceTimelineReconstructor:
    """Reconstructs historical, time-indexed events from tenure attributes."""
    def __init__(self, data_path: str):
        self.data_path = data_path

    def reconstruct_quarterly_series(self) -> pd.Series:
        logger.info("Ingesting dataset and mapping synthetic timeline indexes...")
        df = pd.read_csv(self.data_path)
        
        df['Is_Attrited'] = df['Attrition'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
        anchor_date = pd.Timestamp('2025-12-31')
        
        # Calculate approximate hire date using fractional years translated into quarters
        df['Hire_Quarter'] = df['YearsAtCompany'].apply(
            lambda x: anchor_date - pd.DateOffset(months=int(x * 12))
        )
        df['Hire_Quarter'] = df['Hire_Quarter'].dt.to_period('Q')

        hires_series = df.groupby('Hire_Quarter').size()

        # Isolate exited cohorts and project their exit timeline
        exited_df = df[df['Is_Attrited'] == 1].copy()
        exited_df['Exit_Quarter'] = exited_df['Hire_Quarter'] + exited_df['YearsAtCompany'].apply(lambda x: int(x * 4))
        exited_series = exited_df.groupby('Exit_Quarter').size()

        all_quarters = pd.period_range(
            start=df['Hire_Quarter'].min(), 
            end=anchor_date.to_period('Q'), 
            freq='Q'
        )
        
        timeline_df = pd.DataFrame(index=all_quarters)
        timeline_df['Hires'] = hires_series
        timeline_df['Exits'] = exited_series
        timeline_df.fillna(0, inplace=True)

        timeline_df['Cumulative_Hires'] = timeline_df['Hires'].cumsum()
        timeline_df['Cumulative_Exits'] = timeline_df['Exits'].cumsum()
        timeline_df['Active_Headcount'] = timeline_df['Cumulative_Hires'] - timeline_df['Cumulative_Exits']
        timeline_df['Active_Headcount'] = timeline_df['Active_Headcount'].replace(0, 1)

        # Operational Metric: Quarterly Attrition Rate (%)
        timeline_df['Quarterly_Attrition_Rate'] = (timeline_df['Exits'] / timeline_df['Active_Headcount']) * 100
        
        # Isolate the last 40 quarters of stable tracking data
        target_series = timeline_df['Quarterly_Attrition_Rate'].tail(40)
        logger.info(f"Time series generation finalized. Effective quarters tracked: {len(target_series)}")
        return target_series


# ==========================================
# FORECAST STRATEGY PATTERN
# ==========================================
class BaseForecastWrapper(ABC):
    """Abstract Strategy interface for standardizing time series models."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit_and_backtest(self, train_series: pd.Series, test_series: pd.Series) -> pd.Series:
        """Fits model on training data and returns predictions matching the test horizon index."""
        pass

    @abstractmethod
    def generate_future_forecast(self, full_series: pd.Series, steps_ahead: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Fits model on the entire historical sequence and projects forward."""
        pass


class SarimaxStrategy(BaseForecastWrapper):
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 4)):
        super().__init__("SARIMAX")
        self.order = order
        self.seasonal_order = seasonal_order

    def fit_and_backtest(self, train_series: pd.Series, test_series: pd.Series) -> pd.Series:
        model = SARIMAX(
            train_series, order=self.order, seasonal_order=self.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
        forecast = model.get_forecast(steps=len(test_series)).summary_frame()
        return pd.Series(forecast['mean'].values, index=test_series.index)

    def generate_future_forecast(self, full_series: pd.Series, steps_ahead: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        model = SARIMAX(
            full_series, order=self.order, seasonal_order=self.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
        forecast = model.get_forecast(steps=steps_ahead).summary_frame()
        return forecast['mean'], forecast['mean_ci_lower'], forecast['mean_ci_upper']


class ProphetStrategy(BaseForecastWrapper):
    def __init__(self):
        super().__init__("Meta Prophet")

    def _convert_to_prophet_df(self, series: pd.Series) -> pd.DataFrame:
        """Translates pandas PeriodIndex ('Q') into strict timestamp dataframes for Meta Prophet."""
        # Isolate the index explicitly before reset_index eliminates the PeriodIndex type
        period_index = series.index
        
        df = series.reset_index(drop=True)
        df = pd.DataFrame({
            'ds': period_index.to_timestamp(),  # Safely convert the PeriodIndex arrays directly
            'y': df.values
        })
        return df

    def fit_and_backtest(self, train_series: pd.Series, test_series: pd.Series) -> pd.Series:
        train_df = self._convert_to_prophet_df(train_series)
        
        # Initialize Prophet configured for quarterly trends
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(train_df)
        
        # Safely extract timestamps from the test series PeriodIndex container
        future = pd.DataFrame({'ds': test_series.index.to_timestamp()})
        forecast = model.predict(future)
        return pd.Series(forecast['yhat'].values, index=test_series.index)

    def generate_future_forecast(self, full_series: pd.Series, steps_ahead: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        full_df = self._convert_to_prophet_df(full_series)
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(full_df)
        
        # Generate target timeline extension matching quarterly frequency intervals
        future = model.make_future_dataframe(periods=steps_ahead, freq='Q')
        forecast = model.predict(future)
        
        # Extract exclusively the newly generated out-of-sample future horizon
        future_slice = forecast.tail(steps_ahead)
        
        # Reconstruct the standard business PeriodIndex for deployment dashboard consistency
        period_index = pd.PeriodIndex(future_slice['ds'], freq='Q')
        
        mean_pred = pd.Series(future_slice['yhat'].values, index=period_index)
        ci_lower = pd.Series(future_slice['yhat_lower'].values, index=period_index)
        ci_upper = pd.Series(future_slice['yhat_upper'].values, index=period_index)
        
        return mean_pred, ci_lower, ci_upper


# ==========================================
# ORCHESTRATION & EVALUATION BENCHMARK SUITE
# ==========================================
class ForecastingBenchmarkSuite:
    """Orchestrates model execution, validation logging, and automatic champion selection."""
    def __init__(self, strategies: List[BaseForecastWrapper]):
        self.strategies = strategies
        self.registry: Dict[str, Dict[str, Any]] = {}

    def execute_benchmarks(self, series: pd.Series):
        # Isolate the final 4 quarters (1 Year) as our validation holdout matrix
        train_data = series.iloc[:-4]
        test_data = series.iloc[-4:]

        logger.info("Initializing Workforce Forecasting Competitive Run...")
        for strategy in self.strategies:
            # Backtest model accuracy
            predictions = strategy.fit_and_backtest(train_data, test_data)
            
            mae = mean_absolute_error(test_data, predictions)
            mse = mean_squared_error(test_data, predictions)
            
            self.registry[strategy.name] = {
                "strategy_instance": strategy,
                "MAE": mae,
                "MSE": mse
            }
            logger.info(f"{strategy.name} Evaluated -> Validation MAE: {mae:.4f}% Attrition")

    def select_champion_strategy(self) -> Tuple[str, BaseForecastWrapper, float]:
        """Identifies the optimal strategy based on minimum Mean Absolute Error (MAE)."""
        best_mae = float('inf')
        champion_name = None
        champion_instance = None

        for name, metrics in self.registry.items():
            if metrics["MAE"] < best_mae:
                best_mae = metrics["MAE"]
                champion_name = name
                champion_instance = metrics["strategy_instance"]

        return champion_name, champion_instance, best_mae

    def print_leaderboard(self):
        print("\n" + "="*20 + " FORECASTING LEADERBOARD COMPARISON " + "="*20)
        leaderboard = []
        for name, data in self.registry.items():
            leaderboard.append({
                "Forecasting Engine": name,
                "Validation MAE": f"{data['MAE']:.4f}%",
                "Validation MSE": f"{data['MSE']:.4f}"
            })
        print(pd.DataFrame(leaderboard).to_string(index=False))
        print("="*76)


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

    try:
        # 1. Timeline Data Reconstruction
        reconstructor = WorkforceTimelineReconstructor(data_path=DATA_PATH)
        attrition_ts = reconstructor.reconstruct_quarterly_series()

        # 2. Register Competitor Forecasting Strategies
        forecasting_models = [
            SarimaxStrategy(),
            ProphetStrategy()
        ]

        # 3. Execute Comparative Testing
        suite = ForecastingBenchmarkSuite(strategies=forecasting_models)
        suite.execute_benchmarks(attrition_ts)
        suite.print_leaderboard()

        # 4. Extract Champion Engine
        champion_name, champion_model, best_mae_score = suite.select_champion_strategy()
        print(f"\n🏆 CHAMPION FORECASTER ISOLATED: {champion_name}")
        print(f"-> Minimum Backtest Error Score: {best_mae_score:.4f}% Attrition Deviation")

        # 5. Project True Out-of-Sample Corporate Trend Lines Using the Champion
        forecast_horizon_quarters = 4
        means, lower_bounds, upper_bounds = champion_model.generate_future_forecast(
            attrition_ts, steps_ahead=forecast_horizon_quarters
        )

        print("\n" + "="*19 + f" {champion_name.upper()} PRODUCTION DEMAND FORECAST " + "="*19)
        print("Metric Target: Systemic Attrition Risk Velocity Projections per Horizon Quarter")
        
        results_df = pd.DataFrame({
            'Target Quarter': means.index.astype(str),
            'Point Forecast (%)': means.values,
            'Confidence Lower Bound': lower_bounds.values,
            'Confidence Upper Bound': upper_bounds.values
        })
        print(results_df.to_string(index=False, formatters={
            'Point Forecast (%)': '{:,.2f}%'.format,
            'Confidence Lower Bound': '{:,.2f}%'.format,
            'Confidence Upper Bound': '{:,.2f}%'.format
        }))
        print("="*80)

    except Exception as e:
        logger.error(f"Time Series Competitive Suite execution failed: {str(e)}", exc_info=True)