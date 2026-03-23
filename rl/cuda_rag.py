"""
cuda_rag.py — BM25-based retrieval over cuda_best_practices.md sections.

No external dependencies — pure Python BM25 implementation.
Used during GRPO training to inject relevant CUDA optimization patterns
into turn 2+ feedback when the model is stuck.

Usage:
    rag = CudaRAG()                         # indexes cuda_best_practices.md
    sections = rag.retrieve(query, top_k=2)  # returns top-k relevant sections
"""

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Section:
    """A section of the best practices document."""
    title: str
    content: str       # full markdown content (title + body)
    tokens: list[str] = field(default_factory=list, repr=False)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing.

    Keeps C++ tokens intact: __shfl_down_sync, float4, data_ptr, etc.
    Strips markdown formatting characters.
    """
    # Remove markdown code fence markers but keep the code
    text = re.sub(r'```\w*', '', text)
    # Remove markdown formatting
    text = text.replace('**', '').replace('`', '').replace('#', '')
    # Split on whitespace and punctuation, keeping underscores and dots
    tokens = re.findall(r'[A-Za-z_][\w.]*(?:<[^>]*>)?', text.lower())
    return tokens


def _parse_sections(md_path: str) -> list[Section]:
    """Parse a markdown file into sections split by ## headers."""
    with open(md_path, 'r') as f:
        content = f.read()

    sections = []
    # Split on ## headers (level-2 headings)
    parts = re.split(r'^## ', content, flags=re.MULTILINE)

    for part in parts[1:]:  # skip preamble before first ##
        lines = part.strip().split('\n')
        title = lines[0].strip()
        body = '\n'.join(lines[1:]).strip()
        full = f"## {title}\n{body}"

        section = Section(
            title=title,
            content=full,
            tokens=_tokenize(full),
        )
        sections.append(section)

    return sections


class CudaRAG:
    """BM25 retriever over sections of cuda_best_practices.md.

    BM25 scoring:
        score(q, d) = Σ_{t ∈ q} IDF(t) · (tf(t,d) · (k1+1)) / (tf(t,d) + k1 · (1 - b + b · |d|/avgdl))

    Parameters:
        k1: term frequency saturation (1.2-2.0)
        b:  length normalization (0.75 default)
    """

    def __init__(self, md_path: str = None, k1: float = 1.5, b: float = 0.75):
        if md_path is None:
            md_path = os.path.join(os.path.dirname(__file__), "cuda_best_practices.md")

        self.sections = _parse_sections(md_path)
        self.k1 = k1
        self.b = b

        # Precompute BM25 components
        self._doc_lens = [len(s.tokens) for s in self.sections]
        self._avgdl = sum(self._doc_lens) / max(len(self._doc_lens), 1)
        self._doc_freqs: dict[str, int] = Counter()  # how many docs contain each term
        self._tf: list[Counter] = []  # term frequencies per document

        for section in self.sections:
            tf = Counter(section.tokens)
            self._tf.append(tf)
            for term in set(section.tokens):
                self._doc_freqs[term] += 1

        self._n_docs = len(self.sections)

    def _idf(self, term: str) -> float:
        """Inverse document frequency with smoothing."""
        df = self._doc_freqs.get(term, 0)
        return math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        """BM25 score for a single document against a query."""
        tf = self._tf[doc_idx]
        dl = self._doc_lens[doc_idx]
        score = 0.0

        for term in query_tokens:
            if term not in tf:
                continue
            term_freq = tf[term]
            idf = self._idf(term)
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, top_k: int = 2) -> list[Section]:
        """Retrieve the top-k most relevant sections for a query.

        Args:
            query: combined text of (prompt code + error message + generated code)
            top_k: number of sections to return

        Returns:
            list of Section objects, most relevant first
        """
        query_tokens = _tokenize(query)
        scores = [
            (self._score(query_tokens, i), i)
            for i in range(self._n_docs)
        ]
        scores.sort(reverse=True)

        results = []
        for score, idx in scores[:top_k]:
            if score > 0:
                results.append(self.sections[idx])

        return results

    def retrieve_text(self, query: str, top_k: int = 2, max_chars: int = 3000) -> str:
        """Retrieve relevant sections and return as formatted text.

        Returns a single string ready to inject into feedback, capped at
        max_chars to avoid flooding the context window.
        """
        sections = self.retrieve(query, top_k=top_k)
        if not sections:
            return ""

        parts = []
        total = 0
        for s in sections:
            content = s.content
            # Truncate individual sections if too long
            if total + len(content) > max_chars:
                remaining = max_chars - total
                if remaining < 200:
                    break
                content = content[:remaining] + "\n..."
            parts.append(content)
            total += len(content)

        if not parts:
            return ""

        return (
            "\n\n--- Relevant CUDA Pattern ---\n"
            + "\n\n".join(parts)
            + "\n--- End Pattern ---"
        )
