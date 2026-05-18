import logging
import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd

# Core Natural Language Processing Packages
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Force verification downloading of foundational NLTK artifacts cleanly
for resource in ['stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon']:
    try:
        nltk.data.find(f'corpora/{resource}' if 'lexicon' not in resource else f'sentiment/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Time Series Forecasting Core Frameworks
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Modern Google GenAI Orchestration SDK
from google import genai
from google.genai import types

# Suppress underlying optimization and mathematical convergence warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: QUALITATIVE EMPLOYEE FEEDBACK WORKFLOWS (NLP)
# ==============================================================================

class ProductionTextPreprocessor:
    """Advanced cleaning pipeline utilizing regulatory stopword filtering and lemmatization."""
    def __init__(self):
        self.tokenizer = WordPunctTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        base_stopwords = set(stopwords.words('english'))
        corporate_extensions = {'amazon', 'company', 'work', 'job', 'get', 'would', 'could', 'also', 'lot', 'even'}
        self.stop_words = base_stopwords.union(corporate_extensions)

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = self.tokenizer.tokenize(text)
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        return " ".join(cleaned_tokens)


class EmployeeFeedbackPipeline:
    """Orchestrates Sentiment Extraction and Latent Dirichlet Allocation (LDA) Topic Models."""
    def __init__(self, n_topics: int = 4):
        self.n_topics = n_topics
        self.vader = SentimentIntensityAnalyzer()
        self.preprocessor = ProductionTextPreprocessor()
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=3, stop_words='english')
        self.lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=15)

    def process_feedback_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        logger.info(f"Ingesting raw unstructured feedback text from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target review file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df['Likes'] = df['Likes'].fillna("").astype(str)
        df['Dislikes'] = df['Dislikes'].fillna("").astype(str)
        df['Combined_Feedback'] = df['Likes'] + " " + df['Dislikes']

        # 1. Execute VADER Sentiment Analytics
        scores = df['Combined_Feedback'].apply(lambda x: self.vader.polarity_scores(x))
        df['Sentiment_Compound'] = [s['compound'] for s in scores]

        # 2. Text Normalization Normalization Loop
        df['Cleaned_Feedback'] = df['Combined_Feedback'].apply(self.preprocessor.clean_text)
        modeling_df = df[df['Cleaned_Feedback'].str.strip().str.len() > 0].reset_index(drop=True)

        # 3. Fit Topic Decompositions
        dtm = self.vectorizer.fit_transform(modeling_df['Cleaned_Feedback'])
        topic_distributions = self.lda.fit_transform(dtm)
        feature_names = self.vectorizer.get_feature_names_out()
        
        modeling_df['Dominant_Topic'] = np.argmax(topic_distributions, axis=1)

        # 4. Map Clusters to Clean Dictionary Structs
        topic_keywords = {}
        for topic_idx, topic in enumerate(self.lda.components_):
            top_word_indices = topic.argsort()[:-6 - 1:-1]
            topic_keywords[topic_idx] = [feature_names[i] for i in top_word_indices]

        # Aggregate metrics across text blocks
        topic_pivot = modeling_df.groupby('Dominant_Topic').agg(
            Volume=('Dominant_Topic', 'count'),
            Avg_Sentiment=('Sentiment_Compound', 'mean'),
            Avg_Rating=('Overall_rating', 'mean')
        ).reset_index()

        nlp_payload = []
        for _, row in topic_pivot.iterrows():
            t_id = int(row['Dominant_Topic'])
            nlp_payload.append({
                "id": t_id,
                "keywords": ", ".join(topic_keywords[t_id]),
                "volume": int(row['Volume']),
                "sentiment_score": float(row['Avg_Sentiment']),
                "rating": float(row['Avg_Rating'])
            })
        return nlp_payload


# ==============================================================================
# SECTION 2: QUANTITATIVE WORKFORCE TIMELINE & TIME-SERIES FORECASTS
# ==============================================================================

class WorkforceTimelineReconstructor:
    """Translates point-in-time HR cross-sectional metrics into continuous timelines."""
    def __init__(self, data_path: str):
        self.data_path = data_path

    def reconstruct_quarterly_series(self) -> pd.Series:
        logger.info(f"Ingesting core workforce snapshot arrays from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        df['Is_Attrited'] = df['Attrition'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
        anchor_date = pd.Timestamp('2025-12-31')
        
        df['Hire_Quarter'] = df['YearsAtCompany'].apply(
            lambda x: anchor_date - pd.DateOffset(months=int(x * 12))
        )
        df['Hire_Quarter'] = df['Hire_Quarter'].dt.to_period('Q')
        hires_series = df.groupby('Hire_Quarter').size()

        exited_df = df[df['Is_Attrited'] == 1].copy()
        exited_df['Exit_Quarter'] = exited_df['Hire_Quarter'] + exited_df['YearsAtCompany'].apply(lambda x: int(x * 4))
        exited_series = exited_df.groupby('Exit_Quarter').size()

        all_quarters = pd.period_range(start=df['Hire_Quarter'].min(), end=anchor_date.to_period('Q'), freq='Q')
        
        timeline_df = pd.DataFrame(index=all_quarters)
        timeline_df['Hires'] = hires_series
        timeline_df['Exits'] = exited_series
        timeline_df.fillna(0, inplace=True)

        timeline_df['Cumulative_Hires'] = timeline_df['Hires'].cumsum()
        timeline_df['Cumulative_Exits'] = timeline_df['Exits'].cumsum()
        timeline_df['Active_Headcount'] = timeline_df['Cumulative_Hires'] - timeline_df['Cumulative_Exits']
        timeline_df['Active_Headcount'] = timeline_df['Active_Headcount'].replace(0, 1)

        timeline_df['Quarterly_Attrition_Rate'] = (timeline_df['Exits'] / timeline_df['Active_Headcount']) * 100
        return timeline_df['Quarterly_Attrition_Rate'].tail(40)


class BaseForecastWrapper(ABC):
    """Abstract Strategy interface standardizing competitive forecaster executions."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit_and_backtest(self, train_series: pd.Series, test_series: pd.Series) -> pd.Series:
        pass

    @abstractmethod
    def generate_future_forecast(self, full_series: pd.Series, steps_ahead: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        pass


class SarimaxStrategy(BaseForecastWrapper):
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 4)):
        super().__init__("SARIMAX")
        self.order = order
        self.seasonal_order = seasonal_order

    def fit_and_backtest(self, train_series: pd.Series, test_series: pd.Series) -> pd.Series:
        model = SARIMAX(train_series, order=self.order, seasonal_order=self.seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        forecast = model.get_forecast(steps=len(test_series)).summary_frame()
        return pd.Series(forecast['mean'].values, index=test_series.index)

    def generate_future_forecast(self, full_series: pd.Series, steps_ahead: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        model = SARIMAX(full_series, order=self.order, seasonal_order=self.seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        forecast = model.get_forecast(steps=steps_ahead).summary_frame()
        return forecast['mean'], forecast['mean_ci_lower'], forecast['mean_ci_upper']


class ProphetStrategy(BaseForecastWrapper):
    def __init__(self):
        super().__init__("Meta Prophet")

    def _convert_to_prophet_df(self, series: pd.Series) -> pd.DataFrame:
        period_index = series.index
        df = series.reset_index(drop=True)
        return pd.DataFrame({'ds': period_index.to_timestamp(), 'y': df.values})

    def fit_and_backtest(self, train_series: pd.Series, test_series: pd.Series) -> pd.Series:
        train_df = self._convert_to_prophet_df(train_series)
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(train_df)
        future = pd.DataFrame({'ds': test_series.index.to_timestamp()})
        forecast = model.predict(future)
        return pd.Series(forecast['yhat'].values, index=test_series.index)

    def generate_future_forecast(self, full_series: pd.Series, steps_ahead: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        full_df = self._convert_to_prophet_df(full_series)
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(full_df)
        future = model.make_future_dataframe(periods=steps_ahead, freq='Q')
        forecast = model.predict(future)
        future_slice = forecast.tail(steps_ahead)
        period_index = pd.PeriodIndex(future_slice['ds'], freq='Q')
        return (pd.Series(future_slice['yhat'].values, index=period_index),
                pd.Series(future_slice['yhat_lower'].values, index=period_index),
                pd.Series(future_slice['yhat_upper'].values, index=period_index))


class ForecastingOrchestrationEngine:
    """Executes comparative testing and isolates the champion forecasting matrix."""
    def __init__(self):
        self.strategies: List[BaseForecastWrapper] = [SarimaxStrategy(), ProphetStrategy()]

    def run_competitive_forecast(self, series: pd.Series, steps_ahead: int = 4) -> pd.DataFrame:
        train_data = series.iloc[:-4]
        test_data = series.iloc[-4:]
        
        best_mae = float('inf')
        champion_strategy = None

        logger.info("Evaluating competitive workforce demand models...")
        for strategy in self.strategies:
            preds = strategy.fit_and_backtest(train_data, test_data)
            mae = mean_absolute_error(test_data, preds)
            logger.info(f" -> Strategy {strategy.name} evaluated validation MAE: {mae:.4f}%")
            if mae < best_mae:
                best_mae = mae
                champion_strategy = strategy

        logger.info(f"🥇 Isolated Forecasting Champion: {champion_strategy.name}")
        means, lower, upper = champion_strategy.generate_future_forecast(series, steps_ahead)
        
        return pd.DataFrame({
            'Target Quarter': means.index.astype(str),
            'Point Forecast (%)': [f"{v:.2f}%" for v in means.values]
        })


# ==============================================================================
# SECTION 3: GENERATIVE AI EXECUTIVE NARRATIVE SYNTHESIS
# ==============================================================================

class ExecutiveNarrativeOrchestrator:
    """Synthesizes downstream analytical context payloads into boardroom-ready reports."""
    def __init__(self):
        self.client = genai.Client()
        self.model_name = "gemini-2.5-flash"

    def generate_report(
        self,
        classification_leaderboard: Dict[str, Any],
        permutation_importance: pd.DataFrame,
        forecast_df: pd.DataFrame,
        nlp_payload: List[Dict[str, Any]]
    ) -> str:
        logger.info("Serializing analytics structures into systemic Gemini tokens...")
        
        # Build contextual data string for the prompt context
        context = "=== COMPONENT A: PREDICTIVE MACHINE LEARNING TELEMETRY ===\n"
        context += f"Champion Classifier: {classification_leaderboard['name']} (PR-AUC: {classification_leaderboard['pr_auc']:.3f})\n"
        context += "Top Attrition Predictors (Permutation Drop Metrics):\n"
        for _, row in permutation_importance.head(3).iterrows():
            context += f" - {row['Feature']}: Drop Mean: {row['PR-AUC Drop Mean']:.4f}\n"
            
        context += "\n=== COMPONENT B: QUARTERLY BASELINE CHURN PROJECTIONS ===\n"
        for _, row in forecast_df.iterrows():
            context += f" - Period: {row['Target Quarter']} | Projected Attrition Rate: {row['Point Forecast (%)']}\n"
            
        context += "\n=== COMPONENT C: EMPLOYEE VOICE NATURAL LANGUAGE THEMES ===\n"
        for theme in nlp_payload:
            context += f"Theme Block #{theme['id']} [{theme['keywords']}] -> Vol: {theme['volume']} records | Sentiment Ind: {theme['sentiment_score']:.3f} | Rating: {theme['rating']:.1f}/5.0\n"

        system_instruction = (
            "You are an elite Chief Human Resources Officer (CHRO) and Principal Corporate People Analytics Consultant. "
            "Your objective is to synthesize raw data science telemetry inputs into an executive summary brief. "
            "Organize your report using the following structure:\n"
            "1. EXECUTIVE SUMMARY & FORWARD RETENTION RISK (Discuss baseline quarterly projections)\n"
            "2. RETENTION FRICTION VECTORS (Directly combine structural quantitative features with feedback themes)\n"
            "3. ACTIONABLE LEADERSHIP INTERVENTIONS (Provide 2-3 specific, measurable operational action items).\n\n"
            "CRITICAL PROHIBITION: Do not use data science terminology like 'PR-AUC', 'VADER', 'Prophet', 'SARIMAX', or 'LDA' in the final narrative. "
            "Translate metrics into corporate terms like 'predictive reliability framework', 'sentiment trends', 'historical baseline tracking', and 'thematic focus groups'. "
            "Maintain an objective, strategic tone."
        )

        user_prompt = f"Analyze the attached model metrics and generate the corporate analytics brief:\n\n{context}"

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                max_output_tokens=1500
            )
        )
        return response.text


# ==============================================================================
# SECTION 4: UNIFIED MASTER SYSTEM RUN ROUTINE
# ==============================================================================

if __name__ == "__main__":
    # Absolute paths pointing directly to your local file allocations
    HR_SNAPSHOT_DATA = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    EMPLOYEE_FEEDBACK_DATA = "Amazon_Reviews.csv"

    if not os.getenv("GEMINI_API_KEY"):
        print("CRITICAL EXCEPTION: Environment token missing. Export your key before running: export GEMINI_API_KEY='...'")
    else:
        try:
            print("⚡ STEP 1: INITIALIZING UNSTRUCTURED NLP EXCHANGES...")
            nlp_pipeline = EmployeeFeedbackPipeline(n_topics=4)
            extracted_nlp_themes = nlp_pipeline.process_feedback_dataset(EMPLOYEE_FEEDBACK_DATA)

            print("\n⚡ STEP 2: RECONSTRUCTING WORKFORCE TIMELINE & TIME-SERIES BENCHMARKS...")
            timeline_reconstructor = WorkforceTimelineReconstructor(data_path=HR_SNAPSHOT_DATA)
            attrition_time_series = timeline_reconstructor.reconstruct_quarterly_series()
            
            forecasting_engine = ForecastingOrchestrationEngine()
            future_trend_df = forecasting_engine.run_competitive_forecast(attrition_time_series, steps_ahead=4)

            print("\n⚡ STEP 3: CONSOLIDATING METRICS TO GENERATE THE STRATEGIC EXECUTIVE BRIEF...")
            
            # Simulated upstream champion configuration parameters matching your baseline assets
            champion_classification_telemetry = {
                "name": "XGBoost Strategy Optimization Model",
                "pr_auc": 0.8314
            }
            champion_permutation_importance_df = pd.DataFrame([
                {"Feature": "OverTime_Yes", "PR-AUC Drop Mean": 0.0891},
                {"Feature": "MonthlyIncome", "PR-AUC Drop Mean": 0.0642},
                {"Feature": "StockOptionLevel", "PR-AUC Drop Mean": 0.0385},
                {"Feature": "WorkLifeBalance", "PR-AUC Drop Mean": 0.0210}
            ])

            narrative_orchestrator = ExecutiveNarrativeOrchestrator()
            boardroom_ready_narrative = narrative_orchestrator.generate_report(
                classification_leaderboard=champion_classification_telemetry,
                permutation_importance=champion_permutation_importance_df,
                forecast_df=future_trend_df,
                nlp_payload=extracted_nlp_themes
            )

            print("\n" + "═"*25 + " GENERATED EXECUTIVE INSIGHT DOCUMENT " + "═"*25)
            print(boardroom_ready_narrative)
            print("═"*88)

        except Exception as err:
            logger.error(f"Unified Execution Suite Terminated Abnormally: {str(err)}", exc_info=True)