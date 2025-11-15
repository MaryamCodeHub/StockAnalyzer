"""
SentimentAnalyzer (lightweight)

- Provides a simple lexicon-based sentiment scoring function to return a float in [-1, 1].
- Method:
    - score(texts) -> float
      Accepts a single string or a list of strings; returns the averaged sentiment score.
- Replace or extend with transformer-based models if you want higher accuracy.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Union

# Small lexicon for demo purposes. Extend as needed.
_POSITIVE = {
    "gain", "gains", "bull", "bullish", "up", "surge", "rally", "positive", "buy", "buys", "profit", "outperform", "strong"
}
_NEGATIVE = {
    "loss", "losses", "bear", "bearish", "down", "dump", "sell", "sells", "negative", "drop", "decline", "weak"
}


class SentimentAnalyzer:
    def __init__(self):
        # Optionally load custom lexicons or rules in future
        self.positive = _POSITIVE
        self.negative = _NEGATIVE

    def _clean(self, text: str) -> List[str]:
        text = text.lower()
        # simple tokenization
        tokens = re.findall(r"\b[a-z']+\b", text)
        return tokens

    def score(self, texts: Union[str, Iterable[str]]) -> float:
        """
        Returns a sentiment score between -1 (very negative) and +1 (very positive).
        - If texts is a string, analyze it directly.
        - If texts is a list, return the average score.
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if len(texts) == 0:
            return 0.0

        scores = []
        for t in texts:
            tokens = self._clean(t)
            if not tokens:
                scores.append(0.0)
                continue
            pos = sum(1 for w in tokens if w in self.positive)
            neg = sum(1 for w in tokens if w in self.negative)
            # simple normalized score
            score = (pos - neg) / max(1, len(tokens))
            # clamp to [-1,1]
            score = max(-1.0, min(1.0, score))
            scores.append(score)
        # average across texts
        avg = float(sum(scores) / len(scores))
        return avg