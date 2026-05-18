"""
features/preprocessor.py
Text normalisation pipeline: lowercasing, punctuation removal,
stop-word elimination, and Porter-stemming.

Corresponds to Section 3.3 (Feature Selection and Extraction) – preprocessing
sub-pipeline.
"""

import re
import string
import logging

import nltk
import os as _os

# Ensure the project-local NLTK data folder is on the search path.
# This is required on Render where nltk_data/ lives in the repo root.
_proj_root  = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_local_nltk = _os.path.join(_proj_root, "nltk_data")
if _os.path.isdir(_local_nltk) and _local_nltk not in nltk.data.path:
    nltk.data.path.insert(0, _local_nltk)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.config import (
    LOWERCASE, REMOVE_PUNCT, REMOVE_NUMBERS,
    REMOVE_STOPWORDS, APPLY_STEMMING
)

logger = logging.getLogger(__name__)

# Download required NLTK assets if not already present
for _resource in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{_resource}" if _resource != "punkt"
                       else f"tokenizers/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)


class TextPreprocessor:
    """
    Implements the multi-step text normalisation pipeline described in
    Section 3.3 of the methodology.

    Pipeline order:
        1. Lowercase conversion
        2. Remove URLs and e-mail addresses
        3. Remove HTML tags
        4. Remove punctuation
        5. Remove numeric digits
        6. Tokenise
        7. Remove stop-words
        8. Stem tokens (Porter Stemmer)
        9. Rejoin tokens into a single string
    """

    def __init__(
        self,
        lowercase: bool       = LOWERCASE,
        remove_punct: bool    = REMOVE_PUNCT,
        remove_numbers: bool  = REMOVE_NUMBERS,
        remove_stopwords: bool= REMOVE_STOPWORDS,
        apply_stemming: bool  = APPLY_STEMMING,
    ):
        self.lowercase        = lowercase
        self.remove_punct     = remove_punct
        self.remove_numbers   = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.apply_stemming   = apply_stemming

        self._stop_words = set(stopwords.words("english")) if remove_stopwords \
            else set()
        self._stemmer    = PorterStemmer() if apply_stemming else None
        self._punct_table = str.maketrans("", "", string.punctuation)

    # ── Public API ─────────────────────────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        """Remove URLs, HTML tags, punctuation, and numbers."""
        if not isinstance(text, str):
            return ""
        # URLs
        text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.I)
        # Email addresses
        text = re.sub(r"\S+@\S+", " ", text)
        # HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        if self.lowercase:
            text = text.lower()
        if self.remove_punct:
            text = text.translate(self._punct_table)
        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        """Split cleaned text into word tokens."""
        return text.split()

    def remove_stop_words(self, tokens: list[str]) -> list[str]:
        """Filter out NLTK English stop-words."""
        return [t for t in tokens if t not in self._stop_words and len(t) > 1]

    def stem_tokens(self, tokens: list[str]) -> list[str]:
        """Apply Porter stemming to each token."""
        if self._stemmer is None:
            return tokens
        return [self._stemmer.stem(t) for t in tokens]

    def preprocess_pipeline(self, text: str) -> str:
        """
        Execute the full preprocessing pipeline on a single email string.
        Returns a single space-joined string of processed tokens.
        """
        text   = self.clean_text(text)
        tokens = self.tokenize(text)
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        if self.apply_stemming:
            tokens = self.stem_tokens(tokens)
        return " ".join(tokens)

    def transform(self, texts) -> list[str]:
        """
        Apply preprocess_pipeline to an iterable of email texts.
        Returns a list of preprocessed strings.
        """
        return [self.preprocess_pipeline(t) for t in texts]
