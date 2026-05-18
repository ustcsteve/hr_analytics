import logging
import os
import re
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

# NLP Core Frameworks
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer

# Statistical & Sentiment Modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Force validation downloading of foundational NLTK packages if missing in runtime env
for resource in ['stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon']:
    try:
        nltk.data.find(f'corpora/{resource}' if 'lexicon' not in resource else f'sentiment/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Configure production logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeedbackDataIngestion:
    """Handles raw text collection and cross-field structural isolation."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_and_clean_df(self) -> pd.DataFrame:
        logger.info(f"Ingesting raw reviews from: {self.file_path}")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Target review file not found: {self.file_path}")
        
        df = pd.read_csv(self.file_path)
        
        # Enforce baseline requirement: Fill text voids with empty tokens
        df['Likes'] = df['Likes'].fillna("").astype(str)
        df['Dislikes'] = df['Dislikes'].fillna("").astype(str)
        
        # Consolidate text corpus while maintaining structural field indicators
        df['Combined_Feedback'] = df['Likes'] + " " + df['Dislikes']
        
        logger.info(f"Successfully loaded {len(df)} feedback records.")
        return df


class ProductionTextPreprocessor:
    """Advanced cleaning pipeline utilizing regulatory stopword filtering and lemmatization."""
    def __init__(self):
        self.tokenizer = WordPunctTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Construct domain-specific corporate stopwords
        base_stopwords = set(stopwords.words('english'))
        corporate_extensions = {'amazon', 'company', 'work', 'job', 'get', 'would', 'could', 'also', 'lot', 'even'}
        self.stop_words = base_stopwords.union(corporate_extensions)

    def clean_text(self, text: str) -> str:
        """Executes regex token normalization and POS-aligned lemmatization."""
        # Convert to lower case and drop structural layout noise (\n, \r, punctuation)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        tokens = self.tokenizer.tokenize(text)
        
        # Filter stopwords and enforce noun/verb normalization normalization
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return " ".join(cleaned_tokens)


class EmployeeSentimentAnalyzer:
    """Applies VADER Lexicon processing to map psychological variance vectors."""
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def compute_sentiment_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating fine-grained VADER sentiment parameters...")
        
        # Extract sentiment on the raw combined text block to preserve emotional punctuation symbols
        scores = df['Combined_Feedback'].apply(lambda x: self.vader.polarity_scores(x))
        
        df['Sentiment_Compound'] = [s['compound'] for s in scores]
        df['Sentiment_Positive'] = [s['pos'] for s in scores]
        df['Sentiment_Negative'] = [s['neg'] for s in scores]
        df['Sentiment_Neutral'] = [s['neu'] for s in scores]
        
        # Establish deterministic categorical sentiment assignments
        df['Sentiment_Label'] = pd.cut(
            df['Sentiment_Compound'],
            bins=[-1.0, -0.05, 0.05, 1.0],
            labels=['Negative', 'Neutral', 'Positive']
        )
        return df


class EmployeeTopicModeler:
    """Extracts hidden structural thematic topics using Latent Dirichlet Allocation (LDA)."""
    def __init__(self, n_topics: int = 4, random_state: int = 42):
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=3, stop_words='english')
        self.lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state, max_iter=15)

    def fit_transform_topics(self, text_series: pd.Series) -> Tuple[np.ndarray, List[str]]:
        logger.info(f"Fitting LDA matrix decomposition engine for {self.n_topics} components...")
        dtm = self.vectorizer.fit_transform(text_series)
        topic_distributions = self.lda.fit_transform(dtm)
        feature_names = self.vectorizer.get_feature_names_out()
        return topic_distributions, feature_names

    def get_top_topic_words(self, feature_names: List[str], n_top_words: int = 6) -> Dict[int, List[str]]:
        """Maps continuous distribution weight scores back to absolute top words."""
        topic_keywords = {}
        for topic_idx, topic in enumerate(self.lda.components_):
            top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
            topic_keywords[topic_idx] = [feature_names[i] for i in top_word_indices]
        return topic_keywords


# ==========================================
# EXECUTION & INSIGHTS ORCHESTRATION ENGINE
# ==========================================
if __name__ == "__main__":
    DATA_PATH = "Amazon_Reviews.csv"

    try:
        # 1. Load Data
        ingestor = FeedbackDataIngestion(file_path=DATA_PATH)
        raw_feedback_df = ingestor.load_and_clean_df()

        # 2. Extract Sentiment
        sentiment_engine = EmployeeSentimentAnalyzer()
        processed_df = sentiment_engine.compute_sentiment_scores(raw_feedback_df)

        # 3. Clean Text Content for Topic Modeling
        preprocessor = ProductionTextPreprocessor()
        logger.info("Executing tokenization and lemmatization on feedback text...")
        processed_df['Cleaned_Feedback'] = processed_df['Combined_Feedback'].apply(preprocessor.clean_text)

        # Remove rows that have zero valid text tokens post-cleaning to protect the alignment vectorizer
        valid_mask = processed_df['Cleaned_Feedback'].str.strip().str.len() > 0
        modeling_df = processed_df[valid_mask].reset_index(drop=True)

        # 4. Generate Latent Topics
        topic_engine = EmployeeTopicModeler(n_topics=4)
        topic_distributions, features = topic_engine.fit_transform_topics(modeling_df['Cleaned_Feedback'])
        
        # Append dominant topic configuration to primary dataframe
        modeling_df['Dominant_Topic'] = np.argmax(topic_distributions, axis=1)
        keywords_map = topic_engine.get_top_topic_words(features)

        # ==========================================
        # STRATEGIC EXECUTION ACTION REPORT
        # ==========================================
        print("\n" + "="*23 + " ENTERPRISE NLP WORKFORCE INSIGHT REPORT " + "="*23)
        
        # Print high-level sentiment architecture
        sentiment_distribution = modeling_df['Sentiment_Label'].value_counts(normalize=True) * 100
        print(f"OVERALL CORP SENTIMENT DISTRIBUTION:")
        for label, pct in sentiment_distribution.items():
            print(f" -> {label}: {pct:.2f}%")
        
        print("\n" + "-"*20 + " EXTRACTED LATENT THEMES & CORPORATE ACTION ITEMS " + "-"*20)
        
        # Aggregate sentiment metrics inside each structural topic to isolate friction items
        topic_pivot = modeling_df.groupby('Dominant_Topic').agg(
            Volume=('Dominant_Topic', 'count'),
            Avg_Compound_Sentiment=('Sentiment_Compound', 'mean'),
            Avg_Numeric_Rating=('Overall_rating', 'mean')
        ).reset_index()

        for idx, row in topic_pivot.iterrows():
            topic_num = int(row['Dominant_Topic'])
            words = ", ".join(keywords_map[topic_num])
            
            print(f"\nTHEME TOPIC #{topic_num}: Key Identifiers: [{words}]")
            print(f" -> Total Feedback Occurrences: {int(row['Volume'])} segments")
            print(f" -> Metric Averages: Sentiment: {row['Avg_Compound_Sentiment']:.3f} | Numerical Rating: {row['Avg_Numeric_Rating']:.2f}/5.0")
            
            # Actionable insight logic derived via programmatic semantic combinations
            if row['Avg_Compound_Sentiment'] < 0.10:
                print(" 🚨 LEADERSHIP ACTION REQUIRED: High friction identified in this thematic vector. Investigate regional policy controls.")
            else:
                print(" ✅ RETAIN AND SCALE: Favorable thematic alignment. Maintain standard operational procedures.")
                
        print("="*85)

    except Exception as e:
        logger.error(f"NLP Insight Generation Pipeline Failed: {str(e)}", exc_info=True)