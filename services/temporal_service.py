"""
Temporal ranking service.

Provides utilities for time-aware RAG pipelines:
  - detect_temporal_query(): broad detection for date-sorting
  - detect_filter_query():   strict detection for server-side Pinecone filter
  - build_recency_filter():  build a Pinecone timestamp filter
  - sort_by_date():          sort Pinecone matches by metadata['date']

Two-tier detection design
--------------------------
Server-side filtering has a high cost when it misfires: it excludes
documents from the candidate pool entirely.  Date-sorting has a low cost
when it misfires: it only reorders already-retrieved documents.

Therefore two separate keyword sets are maintained:

  FILTER_KEYWORDS  (strict subset)
    Unambiguous recency terms — safe to narrow the Pinecone search window.
    Excludes \\blast\\b which also matches "recruiter last contacted me"
    (Named Entity, not temporal).

  SORT_KEYWORDS  (superset)
    Broader terms including \\blast\\b, 最後, 上一封 — tolerable for
    sort-only mode where a false positive only reorders results.
"""

import re
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Filter keywords — strict, low false-positive rate
# Used to trigger the server-side Pinecone timestamp filter.
# ---------------------------------------------------------------------------
_FILTER_KEYWORDS = [
    # Chinese — high-confidence recency terms
    r"最新", r"最近", r"近期", r"剛收到", r"最新的", r"最近的",
    # English — unambiguous recency terms
    r"\blatest\b", r"\bnewest\b", r"\blately\b",
    # English — "recent/most recent" (standalone; tolerable FP rate in email context)
    r"\brecent\b", r"\bmost\s+recent\b",
]

# ---------------------------------------------------------------------------
# Sort keywords — broader, superset of filter keywords
# Used to trigger date-sorting of already-retrieved results.
# Includes \blast\b and Chinese near-synonyms that carry higher FP risk
# for filtering but are acceptable for reordering.
# ---------------------------------------------------------------------------
_SORT_KEYWORDS = _FILTER_KEYWORDS + [
    # Chinese — slightly ambiguous (最後 can mean "finally"; 上一封 = "previous email")
    r"最後", r"上一封",
    # English — ambiguous (\blast\b also matches "recruiter last contacted me")
    r"\blast\b",
]

_FILTER_PATTERN = re.compile("|".join(_FILTER_KEYWORDS), re.IGNORECASE)
_SORT_PATTERN   = re.compile("|".join(_SORT_KEYWORDS),   re.IGNORECASE)


def detect_filter_query(query: str) -> bool:
    """
    Return True if the query contains a high-confidence temporal keyword
    that justifies applying a server-side Pinecone recency filter.

    Uses a strict keyword set to minimise false positives — a missed
    detection is safer than erroneously excluding valid documents.
    """
    return bool(_FILTER_PATTERN.search(query))


def detect_temporal_query(query: str) -> bool:
    """
    Return True if the query contains any temporal / recency-intent keyword.

    Uses a broader keyword set suitable for date-sorting (low misfire cost).
    """
    return bool(_SORT_PATTERN.search(query))


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------
_UTC = timezone.utc
_FALLBACK = datetime.min.replace(tzinfo=_UTC)  # oldest possible — sorts last


def _parse_date(date_str: str) -> datetime:
    """
    Parse an RFC 2822 date string (as stored in metadata['date']).

    Falls back to datetime.min (UTC) on any parse error so that
    emails with unparseable dates are sorted to the end.
    """
    if not date_str:
        return _FALLBACK
    try:
        dt = parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_UTC)
        return dt
    except Exception:
        return _FALLBACK


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_recency_filter(days: int = 30) -> dict:
    """
    Build a Pinecone metadata filter that restricts results to emails
    received within the last `days` days.

    Uses the 'timestamp' field (Unix epoch int) which supports Pinecone's
    numeric $gte operator.

    Args:
        days: Number of past days to include (default 30).

    Returns:
        Pinecone filter dict, e.g. {"timestamp": {"$gte": 1737590400}}
    """
    cutoff = int((datetime.now(tz=_UTC) - timedelta(days=days)).timestamp())
    return {"timestamp": {"$gte": cutoff}}


def sort_by_date(matches: list, ascending: bool = False) -> list:
    """
    Sort Pinecone matches by metadata['date'].

    Args:
        matches:   List of match dicts returned by Pinecone (each containing metadata).
        ascending: If False (default), newest first. If True, oldest first.

    Returns:
        A new list sorted by date; length unchanged.
    """
    return sorted(
        matches,
        key=lambda m: _parse_date(m.get("metadata", {}).get("date", "")),
        reverse=not ascending,
    )
