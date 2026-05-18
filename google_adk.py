import logging
import os
from typing import Dict, Any, List
import pandas as pd

# Enterprise Google GenAI SDK
from google import genai
from google.genai import types
from dotenv import load_dotenv


_ = load_dotenv()

# Configure structured production logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExecutiveNarrativeOrchestrator:
    """Orchestrates structured analytical inputs to produce automated C-suite summaries using Gemini."""
    def __init__(self):
        # The client automatically extracts the GEMINI_API_KEY environment variable
        self.client = genai.Client()
        self.model_name = "gemini-2.5-flash"

    def _assemble_analytical_context(
        self,
        model_metrics: Dict[str, Any],
        permutation_importance: pd.DataFrame,
        time_series_forecast: pd.DataFrame,
        nlp_themes: List[Dict[str, Any]]
    ) -> str:
        """Serializes disparate pipeline outputs into a deterministic text context block."""
        
        context = "=== SYSTEMIC QUANTITATIVE DATA INSIGHTS ===\n"
        context += f"Champion Predictive Model: {model_metrics.get('name', 'XGBoost')}\n"
        context += f"Model Performance Metrics: PR-AUC: {model_metrics.get('pr_auc', 0.816):.3f} | ROC-AUC: {model_metrics.get('roc_auc', 0.842):.3f}\n\n"
        
        context += "Top 3 Systemic Attrition Drivers (Permutation Importance):\n"
        for _, row in permutation_importance.head(3).iterrows():
            context += f" - Feature: {row['Feature']} (PR-AUC Drop Impact: {row['PR-AUC Drop Mean']:.4f})\n"
            
        context += "\nBottom 3 Attrition Factors (Negligible Impact):\n"
        for _, row in permutation_importance.tail(3).iterrows():
            context += f" - Feature: {row['Feature']} (PR-AUC Drop Impact: {row['PR-AUC Drop Mean']:.4f})\n\n"
            
        context += "Workforce Churn Velocity Predictions (Next 4 Quarters):\n"
        for _, row in time_series_forecast.iterrows():
            context += f" - Horizon Period: {row['Target Quarter']} | Expected Attrition Rate: {row['Point Forecast (%)']}\n"
            
        context += "\n=== UNSTRUCTURED EMPLOYEE FEEDBACK THEMES (NLP) ===\n"
        for theme in nlp_themes:
            context += f"Theme #{theme['id']} Keywords: [{theme['keywords']}]\n"
            context += f" - Volumetric Share: {theme['volume']} segments\n"
            context += f" - Emotional Sentiment (VADER): {theme['sentiment_score']:.3f} | Associated Numeric Rating: {theme['rating']:.2f}/5.0\n"
            
        return context

    def generate_executive_report(
        self,
        model_metrics: Dict[str, Any],
        permutation_importance: pd.DataFrame,
        time_series_forecast: pd.DataFrame,
        nlp_themes: List[Dict[str, Any]]
    ) -> str:
        """Injects contextual prompt engineering guidelines to generate an executive-ready narrative."""
        
        logger.info("Synthesizing metrics and preparing prompt context for Gemini...")
        analytical_data_context = self._assemble_analytical_context(
            model_metrics, permutation_importance, time_series_forecast, nlp_themes
        )

        # Enforce strict corporate constraints via system instructions
        system_instruction = (
            "You are an expert Chief Human Resources Officer (CHRO) and Principal People Analytics Advisor. "
            "Your objective is to translate complex machine learning model inputs into professional, executive-ready briefs. "
            "Structure your output cleanly with the following clear business sections: \n"
            "1. EXECUTIVE SUMMARY & FORECAST HORIZON (Highlighting baseline risk projections)\n"
            "2. RETENTION FRICTION POINTS (Correlating quantitative model drivers with qualitative NLP feedback)\n"
            "3. STRATEGIC LEADERSHIP INTERVENTIONS (Provide 2-3 explicit, prescriptive action items based on the data).\n\n"
            "CRITICAL: Do not mention data science jargon like 'PR-AUC', 'VADER', or 'LDA' in the final narrative. "
            "Translate metrics into terms like 'predictive reliability', 'sentiment index', and 'thematic focus groups'. "
            "Maintain an objective, analytical corporate tone."
        )

        user_prompt = f"Review the attached model telemetry summaries and synthesize the executive brief:\n\n{analytical_data_context}"

        try:
            logger.info(f"Invoking {self.model_name} processing layer...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,  # Low temperature guarantees deterministic, analytical reporting
                    max_output_tokens=1200
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Generative AI call failed: {str(e)}")
            raise


# ==========================================
# MOCK DATA INJECTION & PIPELINE ORCHESTRATION
# ==========================================
if __name__ == "__main__":
    # In production, these dataframes and dictionaries are loaded dynamically 
    # directly from the outputs of your previous components.
    
    # 1. Classification Metrics Summary
    mock_model_metrics = {"name": "XGBoost Champion", "pr_auc": 0.8245, "roc_auc": 0.8512}
    
    # 2. Permutation Feature Dataframe
    mock_permutation_df = pd.DataFrame([
        {"Feature": "OverTime_Yes", "PR-AUC Drop Mean": 0.0845},
        {"Feature": "MonthlyIncome", "PR-AUC Drop Mean": 0.0612},
        {"Feature": "StockOptionLevel", "PR-AUC Drop Mean": 0.0431},
        {"Feature": "PerformanceRating", "PR-AUC Drop Mean": 0.0002},
        {"Feature": "Gender_Male", "PR-AUC Drop Mean": -0.0001}
    ])
    
    # 3. Time Series Forecasting Dataframe
    mock_forecast_df = pd.DataFrame([
        {"Target Quarter": "2026Q1", "Point Forecast (%)": "16.45%"},
        {"Target Quarter": "2026Q2", "Point Forecast (%)": "17.10%"},
        {"Target Quarter": "2026Q3", "Point Forecast (%)": "15.80%"},
        {"Target Quarter": "2026Q4", "Point Forecast (%)": "18.25%"}
    ])
    
    # 4. Unstructured NLP Theme Vectors
    mock_nlp_themes = [
        {"id": 0, "keywords": "balance, management, toxic, pressure", "volume": 342, "sentiment_score": -0.421, "rating": 1.80},
        {"id": 1, "keywords": "compensation, reward, stock, competitive", "volume": 210, "sentiment_score": 0.385, "rating": 4.10},
        {"id": 2, "keywords": "career, growth, tracking, promotion", "volume": 185, "sentiment_score": -0.105, "rating": 2.90}
    ]

    # Initialize and execute the generation pipeline
    # Ensure GEMINI_API_KEY is exported in your environment variables: export GEMINI_API_KEY="your-key"
    if not os.getenv("GEMINI_API_KEY"):
        print("CRITICAL: Please set the GEMINI_API_KEY environment variable to execute the generative layer.")
    else:
        try:
            orchestrator = ExecutiveNarrativeOrchestrator()
            executive_brief = orchestrator.generate_executive_report(
                model_metrics=mock_model_metrics,
                permutation_importance=mock_permutation_df,
                time_series_forecast=mock_forecast_df,
                nlp_themes=mock_nlp_themes
            )
            
            print("\n" + "="*25 + " GENERATED EXECUTIVE NARRATIVE BRIEF " + "="*25)
            print(executive_brief)
            print("="*85)
            
        except Exception as e:
            logger.error(f"Pipeline execution halted: {str(e)}")