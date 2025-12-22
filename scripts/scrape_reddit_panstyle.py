#!/usr/bin/env python3

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import nltk
from nltk import tokenize
from langdetect import detect, LangDetectException

# make sure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ---------------------------------------------------------------------------
# LANGUAGE FILTERING

def detect_lang(text: str) -> Optional[str]:
    try:
        return detect(text)
    except LangDetectException:
        return None


def keep_sentence(sentence: str, *, allowed_langs: set[str], exclude_english: bool = True) -> bool:
    lang = detect_lang(sentence)
    if lang is None:
        return False
    if exclude_english and lang == "en":
        return False
    return lang in allowed_langs


# ---------------------------------------------------------------------------
# CONFIGURATION

DATASET_OUT_ROOT = Path("Data")  # repo-relative output folder
DATASET_ROOT_DANISH = DATASET_OUT_ROOT / "reddit_dataset_new_DANISH"
DATASET_ROOT_ITALIAN = DATASET_OUT_ROOT / "reddit_dataset_new_ITALIAN"
DATASET_ROOT_POLISH = DATASET_OUT_ROOT / "reddit_dataset_new_POLISH"

SUBREDDITS_CONFIG_DANISH = [
    ("Denmark", 100),
    ("dkfinance", 40),
    ("LegalDK", 30),
    ("dkpol", 30),
    ("Copenhagen", 40),
    ("dkloenseddel", 10),
    ("GossipDK", 8),
    ("InfluencergossipDK", 5),
    ("Aarhus", 40),
    ("BitcoinDK", 20),
]

SUBREDDITS_CONFIG_ITALIAN = [
    ("Italia", 140),
    ("CasualIT", 70),
    ("Roma", 10),
    ("Milano", 10),
    ("relazioni", 4),
    ("libri", 6),
    ("FotografiaItalia", 10),
    ("cucina", 4),
    ("ViaggiITA", 2),
    ("psicologia", 10),
    ("giardinaggioITA", 2),
    ("avvocati", 2),
    ("Toscana", 10),
    ("Napoli", 20),
    ("torino", 10),
]

SUBREDDITS_CONFIG_POLISH = [
    ("inwestowanie", 30),
    ("PolskaNaLuzie", 70),
    ("ksiazki", 30),
    ("Polska", 100),
    ("filozofia", 10),
    ("KryptowalutyPolska", 20),
    ("warszawa", 50),
    ("SwiatCzytnikow", 50),
    ("PolskaPolityka", 20),
    ("BushcraftPL", 30),
]

CATEGORY = "top"
TIME_FILTER = "all"

# target metrics per dataset
NUM_DOCUMENTS = 250
MIN_SENTENCES_PER_DOC = 4
MAX_SENTENCES_PER_DOC = 35
MIN_SENTENCE_WORDS = 4
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# language allowlists (langdetect codes)
ALLOWED_LANGS_DANISH = {"da"}
ALLOWED_LANGS_ITALIAN = {"it"}
ALLOWED_LANGS_POLISH = {"pl"}

EXCLUDE_ENGLISH = True


# ---------------------------------------------------------------------------
# RATE LIMIT SETTINGS

@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6
    base_delay_s: float = 0.8
    max_delay_s: float = 20.0
    jitter_s: float = 0.25


RETRY = RetryPolicy()

# Minimum spacing between *any* YARS network call
MIN_SECONDS_BETWEEN_REQUESTS = 2.0  # default; increase further in case of listing failures

# if a subreddit yields very few docs, we try a second pass with a larger limit
ENABLE_ADAPTIVE_SECOND_PASS = True
SECOND_PASS_MULTIPLIER = 3


class PaceLimiter:
    def __init__(self, min_interval_s: float):
        self.min_interval_s = float(min_interval_s)
        self._last = 0.0
    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last = time.monotonic()


PACE = PaceLimiter(MIN_SECONDS_BETWEEN_REQUESTS)


def _sleep_backoff(attempt: int, policy: RetryPolicy) -> None:
    delay = min(policy.max_delay_s, policy.base_delay_s * (2 ** (attempt - 1)))
    delay += random.uniform(0.0, policy.jitter_s)
    time.sleep(delay)


def call_with_retries(
    fn: Callable[[], Any],
    *,
    label: str,
    policy: RetryPolicy = RETRY,
    pace: Optional[PaceLimiter] = PACE,
    retry_if: Optional[Callable[[Any], bool]] = None,
) -> Any:
    last_exc: Optional[BaseException] = None
    last_result: Any = None
    for attempt in range(1, policy.max_attempts + 1):
        if pace is not None:
            pace.wait()
        try:
            result = fn()
            last_result = result
            if retry_if is not None and retry_if(result):
                print(f"[WARN] {label} attempt {attempt}/{policy.max_attempts} returned empty/invalid result; retrying...")
                if attempt < policy.max_attempts:
                    _sleep_backoff(attempt, policy)
                    continue
            return result
        except Exception as e:
            last_exc = e
            print(f"[WARN] {label} attempt {attempt}/{policy.max_attempts} failed: {e}")
            if attempt < policy.max_attempts:
                _sleep_backoff(attempt, policy)
    if last_exc is not None:
        raise RuntimeError(f"{label} failed after {policy.max_attempts} attempts") from last_exc
    return last_result


# ---------------------------------------------------------------------------
# IMPORT YARS

try:
    from yars.yars import YARS
except ImportError as e:
    raise ImportError(
        "Missing dependency: yars. Install with: pip install -r requirements.txt "
        "or pip install git+https://github.com/datavorous/yars.git"
    ) from e


# ---------------------------------------------------------------------------
# SCRAPING AND DOCUMENT CONSTRUCTION 

def collect_documents_from_subreddit(
    subreddit: str,
    miner: YARS,
    post_limit: int,
    *,
    category: str,
    time_filter: str,
    min_sentence_words: int,
    min_sentences: int,
    max_sentences: int,
    max_documents: int,
    allowed_langs: set[str],
    exclude_english: bool,
) -> list[tuple[list[str], list[str]]]:
    docs: list[tuple[list[str], list[str]]] = []

    # Listing fetch: retry on exceptions
    try:
        subreddit_posts = call_with_retries(
            lambda: miner.fetch_subreddit_posts(
                subreddit,
                limit=post_limit,
                category=category,
                time_filter=time_filter,
            ),
            label=f"fetch_subreddit_posts(r/{subreddit})",
            # YARS may return [] on failure without raising.
            retry_if=lambda r: not r,
        )
    except Exception as e:
        print(f"[ERROR] Could not fetch listing for r/{subreddit}: {e}")
        return docs

    if not subreddit_posts:
        print(f"[WARN] Empty listing returned for r/{subreddit} (limit={post_limit}).")
        return docs

    # Per post
    for post in subreddit_posts:
        if len(docs) >= max_documents:
            break

        if post.get("num_comments", 0) == 0:
            continue

        permalink = post.get("permalink")
        if not permalink:
            continue

        # Details fetch: retry on exceptions
        try:
            post_details = call_with_retries(
                lambda: miner.scrape_post_details(permalink),
                label=f"scrape_post_details({permalink})",
                # YARS may return None/{} on failure without raising.
                retry_if=lambda r: (not r) or (isinstance(r, dict) and not r.get("comments")),
            )
        except Exception as e:
            print(f"[ERROR] Could not scrape details for {permalink}: {e}")
            continue

        if not post_details:
            continue

        sent_author_pairs: list[tuple[str, str]] = []

        for comment in post_details.get("comments", []):
            body = (comment.get("body") or "").strip()
            author = comment.get("author") or "[deleted]"

            if not body or author in {"[deleted]", "AutoModerator"}:
                continue

            for sent in tokenize.sent_tokenize(body):
                sent = sent.replace("\n", " ").strip()
                if len(sent.split()) < min_sentence_words:
                    continue

                # Keep only sentences in the dataset language (and optionally exclude English)
                if not keep_sentence(sent, allowed_langs=allowed_langs, exclude_english=exclude_english):
                    continue

                sent_author_pairs.append((sent, author))

        n = len(sent_author_pairs)
        if n < min_sentences:
            continue

        max_len_allowed = min(max_sentences, n)
        doc_len = random.randint(min_sentences, max_len_allowed)

        start_max = n - doc_len
        start_idx = 0 if start_max <= 0 else random.randint(0, start_max)

        window = sent_author_pairs[start_idx : start_idx + doc_len]
        sentences = [s for s, _ in window]
        authors = [a for _, a in window]

        docs.append((sentences, authors))

    return docs


# ---------------------------------------------------------------------------
# STYLE CHANGE ARRAY

def compute_changes(author_sequence: list[str]) -> list[int]:
    return [0 if author_sequence[i] == author_sequence[i + 1] else 1 for i in range(len(author_sequence) - 1)]


# ---------------------------------------------------------------------------
# SAVING IN THE PAN-STYLE FORMAT

def save_documents(docs: Iterable[tuple[list[str], list[str]]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    for idx, (sentences, author_seq) in enumerate(docs, start=1):
        problem_id = f"problem-{idx}"

        text_path = output_root / f"{problem_id}.txt"
        truth_path = output_root / f"truth-{problem_id}.json"

        with open(text_path, "w", encoding="utf-8", newline="") as f_txt:
            for s in sentences:
                f_txt.write(s.strip() + "\n")

        changes = compute_changes(author_seq)
        num_authors = len(set(author_seq))

        truth = {
            "authors": int(num_authors),
            "changes": changes,
        }

        with open(truth_path, "w", encoding="utf-8") as f_json:
            json.dump(truth, f_json, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# DATASET RUNNER

def build_dataset(
    *,
    name: str,
    dataset_root: Path,
    subreddits_config: list[tuple[str, int]],
    allowed_langs: set[str],
) -> None:
    miner = YARS()

    all_docs: list[tuple[list[str], list[str]]] = []

    print("\n" + "=" * 80)
    print(f"Building dataset: {name}")
    print(f"Output: {dataset_root}")
    print(f"Target documents: {NUM_DOCUMENTS}")
    print("=" * 80)

    for subreddit, post_limit in subreddits_config:
        remaining = NUM_DOCUMENTS - len(all_docs)
        if remaining <= 0:
            break

        print(f"Collecting post-level documents from r/{subreddit} (POST_LIMIT={post_limit})...")

        docs = collect_documents_from_subreddit(
            subreddit,
            miner,
            post_limit=post_limit,
            category=CATEGORY,
            time_filter=TIME_FILTER,
            min_sentence_words=MIN_SENTENCE_WORDS,
            min_sentences=MIN_SENTENCES_PER_DOC,
            max_sentences=MAX_SENTENCES_PER_DOC,
            max_documents=remaining,
            allowed_langs=allowed_langs,
            exclude_english=EXCLUDE_ENGLISH,
        )

        # optional second pass if we got almost nothing (helps when first attempt hit transient failures)
        if ENABLE_ADAPTIVE_SECOND_PASS and len(docs) == 0 and remaining > 0:
            boosted = max(post_limit, 10) * SECOND_PASS_MULTIPLIER
            print(f"  -> 0 docs; trying second pass for r/{subreddit} with POST_LIMIT={boosted}...")
            docs = collect_documents_from_subreddit(
                subreddit,
                miner,
                post_limit=boosted,
                category=CATEGORY,
                time_filter=TIME_FILTER,
                min_sentence_words=MIN_SENTENCE_WORDS,
                min_sentences=MIN_SENTENCES_PER_DOC,
                max_sentences=MAX_SENTENCES_PER_DOC,
                max_documents=remaining,
                allowed_langs=allowed_langs,
                exclude_english=EXCLUDE_ENGLISH,
            )

        print(f"  -> collected {len(docs)} documents from r/{subreddit}.")
        all_docs.extend(docs)

    print(f"Total documents collected for {name}: {len(all_docs)} (target was {NUM_DOCUMENTS}).")

    if len(all_docs) < NUM_DOCUMENTS:
        print(
            "NOTE: Fewer than NUM_DOCUMENTS were created. "
            "This run likely encountered listing/detail fetch failures or filtering removed content. "
            "Consider: (1) increasing MIN_SECONDS_BETWEEN_REQUESTS, (2) authenticating YARS if supported, "
            "or (3) increasing POST_LIMITs."
        )

    print(f"Saving documents to: {dataset_root}")
    save_documents(all_docs, dataset_root)
    print(f"Done: {name}")


# ---------------------------------------------------------------------------
# PIPELINE

def main() -> None:
    build_dataset(
        name="DANISH",
        dataset_root=DATASET_ROOT_DANISH,
        subreddits_config=SUBREDDITS_CONFIG_DANISH,
        allowed_langs=ALLOWED_LANGS_DANISH,
    )

    build_dataset(
        name="ITALIAN",
        dataset_root=DATASET_ROOT_ITALIAN,
        subreddits_config=SUBREDDITS_CONFIG_ITALIAN,
        allowed_langs=ALLOWED_LANGS_ITALIAN,
    )

    build_dataset(
        name="POLISH",
        dataset_root=DATASET_ROOT_POLISH,
        subreddits_config=SUBREDDITS_CONFIG_POLISH,
        allowed_langs=ALLOWED_LANGS_POLISH,
    )


if __name__ == "__main__":
    main()

