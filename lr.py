import logging
import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Core ML Frameworks
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Performance & Explainability Engineering
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

# Configure structured enterprise logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles production data loading and basic structural validation."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading dataset from: {self.file_path}")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Target file path does not exist: {self.file_path}")
        df = pd.read_csv(self.file_path)
        logger.info(f"Dataset successfully loaded. Shape: {df.shape}")
        return df


class DropZeroVarianceFeatures(BaseEstimator, TransformerMixin):
    """Transformer ensuring deterministic removal of zero-variance or constant indicators."""
    def __init__(self, columns_to_drop: List[str] = None):
        self.columns_to_drop = columns_to_drop or ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        existing_cols = [col for col in self.columns_to_drop if col in X_copy.columns]
        if existing_cols:
            X_copy.drop(columns=existing_cols, inplace=True)
        return X_copy


class FeatureEngineeringPipeline:
    """Prepares matrices, enforces strict scaling, and maps categorical elements safely."""
    def __init__(self, target_column: str = 'Attrition'):
        self.target_column = target_column
        self.preprocessor = None

    def prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' missing from DataFrame.")
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
        return X, y

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        dropper = DropZeroVarianceFeatures()
        X_filtered = dropper.transform(X)

        categorical_cols = X_filtered.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X_filtered.select_dtypes(include=['int64', 'float64']).columns.tolist()

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        try:
            encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)

        categorical_transformer = Pipeline(steps=[('onehot', encoder)])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )
        return self.preprocessor


# ==========================================
# STRATEGY PATTERN FOR MULTI-MODEL TESTING
# ==========================================
class BaseModelWrapper(ABC):
    """Abstract Strategy interface for standardizing model workflows across diverse APIs."""
    def __init__(self, name: str):
        self.name = name
        self.model = None

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: pd.Series, scale_pos_weight: float):
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class LogisticRegressionWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("Logistic Regression")

    def fit(self, X_train: np.ndarray, y_train: pd.Series, scale_pos_weight: float):
        self.model = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', class_weight='balanced', random_state=42, max_iter=1000)
        self.model.fit(X_train, y_train)


class RandomForestWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("Random Forest")

    def fit(self, X_train: np.ndarray, y_train: pd.Series, scale_pos_weight: float):
        self.model = RandomForestClassifier(n_estimators=250, max_depth=8, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)


class XGBoostWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("XGBoost")

    def fit(self, X_train: np.ndarray, y_train: pd.Series, scale_pos_weight: float):
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)


class CatBoostWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("CatBoost")

    def fit(self, X_train: np.ndarray, y_train: pd.Series, scale_pos_weight: float):
        self.model = CatBoostClassifier(
            iterations=400,
            depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            verbose=0,
            random_state=42
        )
        self.model.fit(X_train, y_train)


# ==========================================
# ORCHESTRATION ENGINE WITH PERMUTATION
# ==========================================
class ModelBenchmarkingSuite:
    def __init__(self, models: List[BaseModelWrapper]):
        self.models = models
        self.results_registry: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def compute_imbalance_ratio(y: pd.Series) -> float:
        return float(np.sum(y == 0) / np.sum(y == 1))

    def run_benchmark(self, X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray, y_test: pd.Series):
        scale_pos_weight = self.compute_imbalance_ratio(y_train)
        logger.info(f"Calculated Dynamic Imbalance Target Multiplier: {scale_pos_weight:.3f}")

        for wrapper in self.models:
            logger.info(f"Executing Training Strategy for: {wrapper.name}")
            wrapper.fit(X_train, y_train, scale_pos_weight)
            
            y_probs = wrapper.predict_proba(X_test)
            y_preds = wrapper.predict(X_test)
            
            roc_auc = roc_auc_score(y_test, y_probs)
            pr_auc = average_precision_score(y_test, y_probs)
            
            self.results_registry[wrapper.name] = {
                "wrapper_instance": wrapper,
                "ROC-AUC": roc_auc,
                "PR-AUC": pr_auc,
                "Report": classification_report(y_test, y_preds, output_dict=True)
            }

    def select_best_model(self, selection_metric: str = "PR-AUC") -> Tuple[str, BaseModelWrapper, float]:
        best_score = -1.0
        champion_name = None
        champion_instance = None

        for name, metrics in self.results_registry.items():
            score = metrics[selection_metric]
            if score > best_score:
                best_score = score
                champion_name = name
                champion_instance = metrics["wrapper_instance"]

        return champion_name, champion_instance, best_score

    def compute_and_print_permutation_importance(self, champion_wrapper: BaseModelWrapper, X_test: np.ndarray, y_test: pd.Series, feature_names: List[str]):
        """Computes Permutation Feature Importance based on drop in PR-AUC (average_precision)."""
        logger.info(f"Calculating Permutation Feature Importance on Test Set for {champion_wrapper.name}...")
        
        # We pass the underlying scikit-learn compatible estimator or wrapper implementation natively
        # average_precision corresponds precisely to Precision-Recall AUC
        result = permutation_importance(
            champion_wrapper.model, 
            X_test, 
            y_test, 
            scoring='average_precision', 
            n_repeats=10, 
            random_state=42,
            n_jobs=-1
        )
        
        # Package raw outputs into an enterprise dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'PR-AUC Drop Mean': result.importances_mean,
            'PR-AUC Drop Std': result.importances_std
        }).sort_values(by='PR-AUC Drop Mean', ascending=False).reset_index(drop=True)

        print("\n" + "="*20 + f" PERMUTATION IMPORTANCE EXPLAINABILITY REPORT ({champion_wrapper.name}) " + "="*20)
        print("Scoring Metric: PR-AUC (Average Precision) drop when feature vectors are randomized.")
        
        print("\nTOP 5 MOST IMPORTANT FEATURES (Critical Turnover Drivers):")
        print(importance_df.head(5).to_string(index=False, formatters={
            'PR-AUC Drop Mean': '{:,.4f}'.format, 'PR-AUC Drop Std': '{:,.4f}'.format
        }))
        
        print("\nBOTTOM 5 LEAST IMPORTANT FEATURES (Negligible Performance Impact):")
        print(importance_df.tail(5).to_string(index=False, formatters={
            'PR-AUC Drop Mean': '{:,.4f}'.format, 'PR-AUC Drop Std': '{:,.4f}'.format
        }))
        print("="*95)

    def print_leaderboard(self):
        print("\n" + "="*23 + " EXPERIMENTATION LEADERBOARD " + "="*23)
        leaderboard = []
        for name, data in self.results_registry.items():
            leaderboard.append({
                "Model Architecture": name,
                "PR-AUC (Primary)": f"{data['PR-AUC']:.4f}",
                "ROC-AUC": f"{data['ROC-AUC']:.4f}",
                "Attrition Recall": f"{data['Report']['1']['recall']:.4f}"
            })
        print(pd.DataFrame(leaderboard).to_string(index=False))
        print("="*75)


# ==========================================
# MAIN ROUTINE EXECUTION
# ==========================================
if __name__ == "__main__":
    DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

    try:
        # 1. Ingest Data
        ingestor = DataIngestion(file_path=DATA_PATH)
        raw_df = ingestor.load_data()

        # 2. Vector Splits
        fe_pipeline = FeatureEngineeringPipeline(target_column='Attrition')
        X, y = fe_pipeline.prepare_xy(raw_df)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )

        # 3. Transform Arrays
        preprocessor = fe_pipeline.build_preprocessor(X_train_raw)
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        # 4. Extract Dynamic Columns Names Post-Encoding
        try:
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            encoded_cat_cols = cat_encoder.get_feature_names_out().tolist()
        except AttributeError:
            # Fallback alignment for older scikit-learn syntax
            encoded_cat_cols = cat_encoder.get_feature_names().tolist()
        
        dropper = DropZeroVarianceFeatures()
        filtered_X_train = dropper.transform(X_train_raw)
        num_cols = filtered_X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        transformed_feature_names = num_cols + encoded_cat_cols

        # 5. Execute Benchmarking Model Matrix
        model_strategies = [
            LogisticRegressionWrapper(),
            RandomForestWrapper(),
            XGBoostWrapper(),
            CatBoostWrapper()
        ]

        suite = ModelBenchmarkingSuite(models=model_strategies)
        suite.run_benchmark(X_train, y_train, X_test, y_test)
        suite.print_leaderboard()

        # 6. Extract Champion Model
        best_name, best_model, primary_metric_score = suite.select_best_model(selection_metric="PR-AUC")
        print(f"\n🏆 CHAMPION MODEL SELECTION: {best_name}")
        print(f"-> Highest Validated PR-AUC Performance Score: {primary_metric_score:.4f}")

        # 7. Generate Production Permutation Metrics against the Champion Model
        suite.compute_and_print_permutation_importance(
            champion_wrapper=best_model,
            X_test=X_test,
            y_test=y_test,
            feature_names=transformed_feature_names
        )

    except Exception as e:
        logger.error(f"Production pipeline terminated abnormally: {str(e)}", exc_info=True)