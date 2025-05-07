"""
This module contains minimal patches for compatibility between OpenAI and testing.
"""

import os
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr


class FixedEmbeddings(Embeddings):
    """
    Simple embeddings class that provides consistent vectors for tests.

    This class produces deterministic embeddings for common test cases
    to ensure tests can run without API access when needed.
    """

    def __init__(self, dimensions: int = 1536):
        """Initialize with the desired dimensions."""
        self.dimensions = dimensions

        # Create a small set of fixed embeddings for core test cases only
        self.special_cases = {
            # Basic words - with necessary relationships for tests
            "foo": self._normalized([0.9, 0.1, 0.1] + [0.1] * (self.dimensions - 3)),
            "bar": self._normalized([0.1, 0.9, 0.1] + [0.1] * (self.dimensions - 3)),
            "baz": self._normalized([0.7, 0.3, 0.1] + [0.1] * (self.dimensions - 3)),
            # Additional variation for MMR tests
            "bay": self._normalized([0.7, 0.2, 0.2] + [0.1] * (self.dimensions - 3)),
            "bax": self._normalized([0.6, 0.2, 0.3] + [0.1] * (self.dimensions - 3)),
            "baw": self._normalized([0.5, 0.3, 0.3] + [0.1] * (self.dimensions - 3)),
            "bav": self._normalized([0.4, 0.3, 0.4] + [0.1] * (self.dimensions - 3)),
            # Simple terms for basic tests
            "apple": self._normalized([0.9, 0.2, 0.1] + [0.1] * (self.dimensions - 3)),
            "orange": self._normalized([0.7, 0.4, 0.1] + [0.1] * (self.dimensions - 3)),
            "hammer": self._normalized([0.2, 0.3, 0.9] + [0.1] * (self.dimensions - 3)),
        }

    def _normalized(self, vector: List[float]) -> List[float]:
        """Normalize a vector to unit length for cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return [float(x / norm) for x in vector]
        return vector

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string."""
        # For exact matches of test terms, use the pre-defined vectors
        if text in self.special_cases:
            return self.special_cases[text]

        # For sentences that may contain our keywords, help tests that expect
        # specific content like "The cat is on the mat" to match patterns
        lower_text = text.lower()
        for key, vector in self.special_cases.items():
            if key in lower_text:
                # Return a slightly perturbed version of the vector
                # Use a more controlled perturbation that won't have integer issues
                seed_val = abs(hash(text)) % (2**32 - 1)
                np.random.seed(seed_val)
                perturb = 0.9 + 0.2 * np.random.rand()
                return self._normalized([x * perturb for x in vector])

        # For all other cases, create a deterministic but unique vector
        # This is a simple hash-based approach to create consistent vectors
        # Use a positive seed value within numpy's required range
        seed_val = abs(hash(text)) % (2**32 - 1)
        np.random.seed(seed_val)
        vector = np.random.rand(self.dimensions).tolist()
        return self._normalized(vector)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of documents."""
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Return embedding for a query."""
        return self._get_embedding(text)


def get_embeddings_for_tests() -> Embeddings:
    """Get appropriate embeddings for tests based on API key availability."""
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        # If no API key is available, use fixed embeddings
        return FixedEmbeddings(dimensions=1536)

    # If API key is available, try to use OpenAI embeddings
    try:
        from langchain_openai.embeddings.base import OpenAIEmbeddings

        # Return OpenAI embeddings with explicitly provided API key as SecretStr
        return OpenAIEmbeddings(api_key=SecretStr(api_key))
    except Exception:
        return FixedEmbeddings(dimensions=1536)
