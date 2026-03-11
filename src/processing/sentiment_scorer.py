"""
Sentiment Scorer
Computes daily sentiment scores from news headlines using FinBERT and VADER.
FinBERT is the primary scorer, VADER is retained for comparison.
Logic adapted from the original research paper implementation.
"""

import logging
from statistics import mean

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from config.settings import FINBERT_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

nltk.download("vader_lexicon", quiet=True)

# FinBERT pipeline is loaded once at module level to avoid reloading on every call
_finbert_pipeline = None


def get_finbert_pipeline():
    """
    Lazy-load the FinBERT pipeline on first call and cache it for reuse.
    Loading the model takes ~30 seconds so we only do it once per process.
    """
    global _finbert_pipeline
    if _finbert_pipeline is None:
        logger.info("Loading FinBERT model from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512
        )
        logger.info("FinBERT model loaded successfully")
    return _finbert_pipeline


def finbert_score_single(headline):
    """
    Score a single headline using FinBERT.
    Returns a float in range [-1, 1] where:
        positive label  -> +score
        neutral label   -> 0
        negative label  -> -score
    """
    if not headline or not headline.strip():
        return 0.0

    try:
        nlp = get_finbert_pipeline()
        result = nlp(headline[:512])[0]
        label = result["label"].lower()
        score = result["score"]

        if label == "positive":
            return round(score, 6)
        elif label == "negative":
            return round(-score, 6)
        else:
            return 0.0

    except Exception as e:
        logger.warning(f"FinBERT scoring failed for headline: {e}")
        return 0.0


def vader_score_single(headline):
    """
    Score a single headline using VADER.
    Returns a float in range [-1, 1].
    Positive dominant -> +pos score, negative dominant -> -neg score, neutral -> 0.
    """
    if not headline or not headline.strip():
        return 0.0

    try:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(headline)
        pos, neg, neu = scores["pos"], scores["neg"], scores["neu"]

        dominant = max(pos, neg, neu)
        if dominant == pos:
            return round(pos, 6)
        elif dominant == neg:
            return round(-neg, 6)
        else:
            return 0.0

    except Exception as e:
        logger.warning(f"VADER scoring failed for headline: {e}")
        return 0.0


def score_headlines(headlines):
    """
    Score a list of headlines and return the mean FinBERT and VADER scores for the day.
    Returns a tuple (finbert_mean, vader_mean).
    Returns (0.0, 0.0) if headlines list is empty.
    """
    if not headlines:
        return 0.0, 0.0

    finbert_scores = [finbert_score_single(h) for h in headlines]
    vader_scores   = [vader_score_single(h) for h in headlines]

    return round(mean(finbert_scores), 6), round(mean(vader_scores), 6)


def score_dataframe(df):
    """
    Score all headlines in a DataFrame that contains a 'headlines' column (list of strings).
    Adds 'finbert_score' and 'vader_score' columns in place.
    Used during offline feature building on the full historical dataset.
    """
    logger.info(f"Scoring sentiment for {len(df)} rows...")

    finbert_col = []
    vader_col   = []

    for idx, row in df.iterrows():
        headlines = row.get("headlines", [])
        if isinstance(headlines, str):
            headlines = [headlines]

        fb, vd = score_headlines(headlines)
        finbert_col.append(fb)
        vader_col.append(vd)

        if idx % 50 == 0:
            logger.info(f"  Scored {idx}/{len(df)} rows")

    df["finbert_score"] = finbert_col
    df["vader_score"]   = vader_col

    logger.info("Sentiment scoring complete")
    return df
