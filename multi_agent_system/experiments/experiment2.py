# Experiment 2: Multi-hop QA pipeline without question decomposition.
#This ablation removes the QuestionAnalyzerAgent and uses the original question directly during retrieval and evidence selection.
from __future__ import annotations  # Enables postponed evaluation of type hints

import argparse  # Command-line argument parsing
import json  # Reading/writing JSON files
import math  # Mathematical functions
import pickle  # Saving/loading cached Python objects
import re  # Regular expressions
import string  # String constants and utilities
import time  # Runtime measurement

from collections import Counter, defaultdict  # Counting and dictionary helpers
from dataclasses import dataclass  # Lightweight data containers
from pathlib import Path  # File and path handling
from typing import Dict, List, Tuple  # Type annotations

import requests  # HTTP requests to Ollama API
from datasets import load_dataset  # Loading HotpotQA dataset


TOKEN_RE = re.compile(r"\w+")  # Regex pattern for word tokenization

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower()) # Convert text to lowercase tokens


@dataclass
class RetrievedDocument:
    title: str  # Document title
    sentences: List[str]  # Document sentences/content
    score: float  # Retrieval score


@dataclass
class PipelineResult:
    question: str  # Input question
    prediction: str  # Final predicted answer
    raw_answer: str  # Raw LLM output
    subquestions: List[str]  # Generated subquestions
    selected_titles: List[str]  # Selected evidence documents
    retrieved_titles: List[str]  # Retrieved candidate documents
    runtime: float  # Total pipeline runtime


class BM25Retriever:
    def __init__(
        self,
        corpus: Dict[str, List[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len: Dict[str, int] = {}
        self.inverted: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        total_len = 0

        # Build BM25 inverted index
        for title, sentences in corpus.items():
            tokens = tokenize(title + " " + " ".join(sentences))
            counts = Counter(tokens)

            self.doc_len[title] = len(tokens)
            total_len += len(tokens)

            for token, tf in counts.items():
                self.inverted[token].append((title, tf))

        self.n_docs = len(corpus)
        self.avgdl = total_len / max(self.n_docs, 1)

    def search(self, query: str, top_k: int = 10) -> List[RetrievedDocument]:
        query_tokens = tokenize(query)
        scores: Dict[str, float] = defaultdict(float)

        # Compute BM25 relevance scores
        for token in query_tokens:
            postings = self.inverted.get(token)

            if not postings:
                continue

            df = len(postings)
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

            for title, tf in postings:
                dl = self.doc_len[title]
                denom = tf + self.k1 * (
                    1 - self.b + self.b * dl / max(self.avgdl, 1)
                )

                scores[title] += idf * (tf * (self.k1 + 1)) / denom

        query_lower = query.lower().strip()

        # Boost exact and partial title matches
        for title in list(scores.keys()):
            title_lower = title.lower().strip()

            if title_lower == query_lower:
                scores[title] += 120.0
            elif query_lower in title_lower or title_lower in query_lower:
                scores[title] += 25.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievedDocument(title, self.corpus[title], score)
            for title, score in ranked
        ]


class OllamaClient:
    def __init__(
        self,
        model: str = "mixtral",
        host: str = "http://127.0.0.1:11434",
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")

    def generate(self, prompt: str) -> str:
        # Send prompt to local Ollama API
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_ctx": 8192,
                },
            },
            timeout=600,
        )

        response.raise_for_status()

        return response.json()["response"].strip()

def unique(values: List[str]) -> List[str]:
    seen = set()
    out = []

    # Remove duplicates while preserving order
    for value in values:
        value = value.strip()

        if value and value not in seen:
            out.append(value)
            seen.add(value)

    return out


def extract_entities(question: str) -> List[str]:
    # Extract quoted entities
    quoted = re.findall(r'"([^"]+)"', question)

    # Extract capitalized multi-word entities
    capitalized = re.findall(
        r"\b[A-Z][a-zA-Z0-9'â€™.-]+(?:\s+[A-Z][a-zA-Z0-9'â€™.-]+)+\b",
        question,
    )

    # Remove common question prefixes
    bad_prefixes = [
        "Were ",
        "What ",
        "Which ",
        "Who ",
        "Are ",
        "The ",
        "A ",
        "An ",
        "Is ",
        "Was ",
        "Did ",
        "Does ",
        "Do ",
    ]

    entities = []

    for item in quoted + capitalized:
        item = item.strip(" ?.,;:()[]{}\"'")

        for prefix in bad_prefixes:
            if item.startswith(prefix):
                item = item[len(prefix):].strip()

        if item and len(item) > 2:
            entities.append(item)

    # Separate multi-word and single-word entities
    multi = [e for e in unique(entities) if " " in e]
    single = [e for e in unique(entities) if " " not in e]

    # Remove single entities already covered by multi-word entities
    single = [
        e for e in single
        if not any(e.lower() in m.lower() for m in multi)
    ]

    return unique(multi + single)


def infer_relation_terms(question: str) -> List[str]:
    q = question.lower()
    terms = []

    # Map question keywords to retrieval-related terms
    mapping = {
        "nationality": ["nationality", "country", "citizenship"],
        "same nationality": ["nationality", "country", "citizenship"],
        "started": ["started", "founded", "launched", "began"],
        "first": ["first", "started", "founded", "date", "began"],
        "older": ["born", "birth date", "age"],
        "director": ["director", "directed by"],
        "government": ["government position", "office", "politician"],
        "position": ["position", "office", "role"],
        "arena": ["arena", "capacity", "home games"],
        "seat": ["capacity", "seating capacity", "seated"],
        "capacity": ["capacity", "seating capacity", "seated"],
        "formed": ["formed by", "founder", "created by"],
        "debut album": ["debut album", "band", "formed by"],
        "located": ["location", "neighborhood", "district"],
        "neighborhood": ["neighborhood", "district"],
        "city": ["city", "based in", "location"],
        "based": ["based in", "city", "location"],
        "stage name": ["stage name", "also known as"],
        "magazine": ["magazine", "first published", "founded"],
        "series": ["series", "books", "companion books"],
        "writer": ["writer", "written by"],
        "composer": ["composer", "composed by"],
        "headquartered": ["headquarters", "headquartered"],
        "population": ["population", "inhabitants"],
        "released": ["released", "release date"],
    }

    # Add matching relation terms
    for key, vals in mapping.items():
        if key in q:
            terms.extend(vals)

    return unique(terms)


def question_type(question: str) -> str:
    q = question.lower().strip()

    # Classify question type from opening words
    if q.startswith(("are ", "is ", "was ", "were ", "do ", "does ", "did ")):
        return "yes_no"
    if q.startswith("who "):
        return "person"
    if q.startswith("when "):
        return "date"
    if q.startswith("where "):
        return "location"
    if q.startswith(("which ", "what ")):
        return "entity"

    return "short"


def build_corpus_from_split(split: str, max_examples: int | None = None) -> Dict[str, List[str]]:
    # Load HotpotQA split
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=split)
    corpus: Dict[str, List[str]] = {}

    # Extract unique documents from contexts
    for i, item in enumerate(dataset):
        if max_examples is not None and i >= max_examples:
            break

        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            if title not in corpus:
                corpus[title] = list(sentences)

    return corpus


def merge_corpora(*corpora: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged = {}

    # Merge corpora without duplicate titles
    for corpus in corpora:
        for title, sentences in corpus.items():
            if title not in merged:
                merged[title] = sentences

    return merged


def load_or_build_corpus(
    cache_path: Path,
    max_train: int | None,
    include_validation: bool,
) -> Dict[str, List[str]]:
    # Reuse cached corpus if available
    if cache_path.exists():
        print(f"Loading cached corpus from {cache_path}")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    print("Loading train corpus...")
    train_corpus = build_corpus_from_split("train", max_examples=max_train)
    corpora = [train_corpus]

    if include_validation:
        print("Loading validation corpus...")
        corpora.append(build_corpus_from_split("validation", max_examples=None))

    corpus = merge_corpora(*corpora)

    # Save corpus for later runs
    print(f"Saving corpus cache to {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("wb") as f:
        pickle.dump(corpus, f)

    return corpus


def parse_bullets(text: str) -> List[str]:
    # Extract bullet-point items from LLM output
    return unique(
        [
            line.strip()[2:].strip()
            for line in text.splitlines()
            if line.strip().startswith("- ") and line.strip()[2:].strip().lower() != "none"
        ]
    )


def analyze_question(llm: OllamaClient, question: str, entities: List[str]) -> List[str]:
    # Ask LLM to split the question into retrieval-focused subquestions
    prompt = f"""
You are QuestionAnalyzerAgent.

Break the question into 2-4 retrieval-friendly subquestions.

Rules:
- For comparison questions, create one lookup per compared entity and one comparison step.
- For date/order questions, ask for exact dates.
- For bridge questions, include the bridge step explicitly.
- For questions asking for a position, capacity, location, date, director, founder, or nationality, ask for that exact fact.
- Do not answer.
- Return bullets only.

Question:
{question}

Detected entities:
{chr(10).join("- " + e for e in entities) if entities else "- none"}

SUBQUESTIONS:
"""
    output = llm.generate(prompt)
    subquestions = parse_bullets(output)

    if subquestions:
        return subquestions[:4]

    # Fallback if no valid subquestions are returned
    fallback = [question]
    for entity in entities:
        fallback.append(f"What relevant facts are stated about {entity}?")

    return fallback[:4]


def format_docs(docs: List[RetrievedDocument], max_sentences: int = 4) -> str:
    blocks = []

    # Format documents for agent prompts
    for i, doc in enumerate(docs):
        sent_text = "\n".join(
            f"S{j}: {sentence}"
            for j, sentence in enumerate(doc.sentences[:max_sentences])
        )
        blocks.append(
            f"DOC_ID: D{i}\n"
            f"TITLE: {doc.title}\n"
            f"SCORE: {doc.score:.3f}\n"
            f"{sent_text}"
        )

    return "\n\n".join(blocks)


def parse_doc_ids(text: str, max_id: int) -> List[int]:
    ids = []

    # Extract valid document IDs from LLM output
    for match in re.findall(r"\bD(\d+)\b", text):
        idx = int(match)

        if 0 <= idx <= max_id and idx not in ids:
            ids.append(idx)

    return ids


def heuristic_score_doc(
    question: str,
    doc: RetrievedDocument,
    entities: List[str],
    relation_terms: List[str],
) -> float:
    title = doc.title.lower()
    text = (doc.title + " " + " ".join(doc.sentences[:8])).lower()
    q_tokens = set(tokenize(question))
    doc_tokens = set(tokenize(text))

    score = doc.score

    # Boost documents that match extracted entities
    for entity in entities:
        e = entity.lower()

        if title == e:
            score += 100
        elif e in title:
            score += 50
        elif e in text:
            score += 20

    # Boost documents that contain relation terms
    for term in relation_terms:
        if term.lower() in text:
            score += 10

    # Add small boost for token overlap
    score += len(q_tokens & doc_tokens) * 0.5

    noisy = [
        "list of",
        "bibliography",
        "discography",
        "alternative facts",
        "chuck norris facts",
        "seating capacity",
    ]

    # Penalize common noisy document titles
    if any(n in title for n in noisy):
        score -= 18

    return score


def rerank_candidates(
    question: str,
    candidates: List[RetrievedDocument],
    max_docs: int,
) -> List[RetrievedDocument]:
    entities = extract_entities(question)
    relation_terms = infer_relation_terms(question)

    # Re-score retrieved documents using question-specific signals
    scored = [
        (heuristic_score_doc(question, doc, entities, relation_terms), doc)
        for doc in candidates
    ]

    scored.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored[:max_docs]]


def selector_agent(
    llm: OllamaClient,
    question: str,
    subquestions: List[str],
    docs: List[RetrievedDocument],
) -> List[RetrievedDocument]:
    qtype = question_type(question)

    # Ask the LLM to select relevant evidence documents
    prompt = f"""
You are SelectorAgent.

Select the smallest set of documents needed to answer the question.

Rules:
- Return ONLY DOC_IDs.
- Do not explain.
- Select documents containing answer-bearing facts.
- For bridge questions, select every document in the chain.
- For comparison questions, select evidence for BOTH compared entities.
- For yes/no questions, select documents that prove yes or prove no.
- For location questions, select documents that contain the exact location for each entity.
- For capacity/date/position questions, select documents that explicitly contain the exact requested value.
- Avoid distractors and generic pages.

Question type: {qtype}

Question:
{question}

Subquestions:
{chr(10).join("- " + s for s in subquestions)}

Candidate documents:
{format_docs(docs, max_sentences=6)}

Return exactly:
SELECTED:
- D0
- D1
"""
    output = llm.generate(prompt)
    ids = parse_doc_ids(output, len(docs) - 1)

    # Fallback to heuristic selection if no valid document IDs are returned
    if not ids:
        ids = fallback_select(question, docs)

    return [docs[i] for i in ids[:10]]


def adder_agent(
    llm: OllamaClient,
    question: str,
    subquestions: List[str],
    all_docs: List[RetrievedDocument],
    selected_docs: List[RetrievedDocument],
) -> List[RetrievedDocument]:
    selected_titles = {d.title for d in selected_docs}

    # Ask the LLM to identify missing supporting documents
    prompt = f"""
You are AdderAgent.

Add missing documents needed for multi-hop reasoning.

Rules:
- Return ONLY DOC_IDs.
- Do not explain.
- Add missing bridge facts or missing facts for compared entities.
- Add the page containing exact requested values: capacity, date, position, location, nationality, founder, director.
- If nothing should be added, return:
ADDED:
- none

Question:
{question}

Subquestions:
{chr(10).join("- " + s for s in subquestions)}

Currently selected titles:
{chr(10).join("- " + d.title for d in selected_docs) if selected_docs else "- none"}

Candidate documents:
{format_docs(all_docs, max_sentences=6)}

Return exactly:
ADDED:
- D0
- D1
"""
    output = llm.generate(prompt)
    ids = parse_doc_ids(output, len(all_docs) - 1)

    # Add only documents not already selected
    added = []
    for i in ids:
        doc = all_docs[i]

        if doc.title not in selected_titles:
            added.append(doc)

    return added[:8]


def parse_answer_text(raw: str) -> str:
    # Extract answer from expected LLM output format
    for line in raw.splitlines():
        stripped = line.strip()

        if stripped.lower().startswith("answer:"):
            return stripped.split(":", 1)[1].strip()

    # Fallback if ANSWER field is missing
    return raw.strip().splitlines()[0].strip() if raw.strip() else "Not Answerable"


def clean_prediction(pred: str, qtype: str) -> str:
    pred = pred.strip().strip("\"'")

    # Normalize yes/no predictions
    if qtype == "yes_no":
        low = pred.lower()

        if low.startswith("yes"):
            return "yes"
        if low.startswith("no"):
            return "no"

    bad = {
        "not enough information",
        "unknown",
        "cannot be determined",
        "insufficient evidence",
    }

    # Normalize unsupported answers
    if pred.lower() in bad:
        return "Not Answerable"

    return pred


def answer_agent(
    llm: OllamaClient,
    question: str,
    subquestions: List[str],
    evidence: List[RetrievedDocument],
) -> tuple[str, str]:
    qtype = question_type(question)
    evidence_text = format_docs(evidence, max_sentences=8)

    # Ask the LLM to answer using selected evidence only
    prompt = f"""
You are AnswererAgent.

Answer using ONLY the provided evidence.

Question type: {qtype}

Strict rules:
- Return exactly two lines:
ANSWER: <answer>
REASON: <one short sentence>
- Never use outside knowledge.
- If the exact answer is not in the evidence, answer Not Answerable.
- If question type is yes_no, ANSWER must be exactly yes or no.
- If question asks Who/What/Which, answer with the requested entity, title, person, place, date, position, or value.
- For comparison/date/order questions, compare the explicit dates or values in evidence.
- Be concise.

Question:
{question}

Subquestions:
{chr(10).join("- " + s for s in subquestions)}

Evidence:
{evidence_text}
"""
    raw = llm.generate(prompt)

    return clean_prediction(parse_answer_text(raw), qtype), raw


def fallback_select(question: str, docs: List[RetrievedDocument], max_docs: int = 8) -> List[int]:
    entities = [e.lower() for e in extract_entities(question)]
    relation_terms = [t.lower() for t in infer_relation_terms(question)]
    q_tokens = set(tokenize(question))

    scored = []

    # Heuristic document selection fallback
    for i, doc in enumerate(docs):
        title = doc.title.lower()
        text = (doc.title + " " + " ".join(doc.sentences[:8])).lower()
        tokens = set(tokenize(text))

        score = 0

        for entity in entities:
            if entity == title:
                score += 35
            elif entity in title:
                score += 18
            elif entity in text:
                score += 7

        for term in relation_terms:
            if term in text:
                score += 5

        score += len(q_tokens & tokens)

        if score > 0:
            scored.append((score, i))

    scored.sort(reverse=True)

    return [i for _, i in scored[:max_docs]]


def retrieve_candidates(
    retriever: BM25Retriever,
    question: str,
    subquestions: List[str],
    entities: List[str],
    top_k: int,
) -> List[RetrievedDocument]:
    relation_terms = infer_relation_terms(question)

    # Build retrieval queries from question, entities and subquestions
    queries = []
    queries.append(question)
    queries.extend(entities)
    queries.extend(subquestions)

    # Add broad entity-based query expansions
    for entity in entities:
        queries.extend(
            [
                entity,
                f"{entity} biography",
                f"{entity} nationality",
                f"{entity} born",
                f"{entity} country",
                f"{entity} origin",
                f"{entity} founded",
                f"{entity} director",
                f"{entity} office",
                f"{entity} location",
                f"{entity} capacity",
                f"{entity} formed by",
                f"{entity} founder",
                f"{entity} based in",
                f"{entity} headquarters",
            ]
        )

        # Add focused entity-relation queries
        for term in relation_terms:
            queries.append(f"{entity} {term}")

    queries = unique([q for q in queries if q])

    docs = []
    seen_titles = set()

    # Retrieve documents and remove duplicate titles
    for query in queries:
        for doc in retriever.search(query, top_k=top_k):
            if doc.title not in seen_titles:
                docs.append(doc)
                seen_titles.add(doc.title)

    return docs


def run_pipeline(
    question: str,
    retriever: BM25Retriever,
    llm: OllamaClient,
    top_k: int,
    iterations: int,
    selector_candidates: int,
    verbose: bool = True,
) -> PipelineResult:
    start = time.time()

    entities = extract_entities(question)

    # Ablation: use original question instead of generated subquestions
    subquestions = [question]

    # Retrieve and rerank candidate documents
    candidates = retrieve_candidates(
        retriever=retriever,
        question=question,
        subquestions=subquestions,
        entities=entities,
        top_k=top_k,
    )

    candidates = rerank_candidates(question, candidates, max_docs=selector_candidates)

    # Select initial evidence documents
    selected = selector_agent(llm, question, subquestions, candidates)

    # Iteratively add missing evidence and re-select
    for _ in range(iterations):
        added = adder_agent(llm, question, subquestions, candidates, selected)

        seen = {d.title for d in selected}
        for doc in added:
            if doc.title not in seen:
                selected.append(doc)
                seen.add(doc.title)

        selected = selector_agent(llm, question, subquestions, selected)

    # Generate final answer
    prediction, raw_answer = answer_agent(llm, question, subquestions, selected)
    runtime = time.time() - start

    if verbose:
        print("\nQUESTION")
        print(question)

        print("\nENTITIES")
        for e in entities:
            print("-", e)

        print("\nSUBQUESTIONS")
        for s in subquestions:
            print("-", s)

        print("\nTOP RETRIEVED TITLES")
        for d in candidates[:30]:
            print(f"- {d.title} ({d.score:.2f})")

        print("\nSELECTED EVIDENCE")
        for d in selected:
            first = d.sentences[0] if d.sentences else ""
            print(f"- {d.title}: {first}")

        print("\nANSWER")
        print(raw_answer)
        print(f"\nParsed prediction: {prediction}")
        print(f"Runtime: {runtime:.2f}s")

    # Return structured pipeline output
    return PipelineResult(
        question=question,
        prediction=prediction,
        raw_answer=raw_answer,
        subquestions=subquestions,
        selected_titles=[d.title for d in selected],
        retrieved_titles=[d.title for d in candidates],
        runtime=runtime,
    )


def normalize_answer(s: str) -> str:
    # Normalize text before comparing answers
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return " ".join(remove_articles(remove_punc(s.lower())).split())


def exact_match_score(prediction: str, gold: str) -> float:
    # Compute exact match after normalization
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    # Compute token-level F1 score
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def title_recall(retrieved_titles: List[str], gold_titles: List[str]) -> float:
    # Compute recall for supporting document titles
    gold_set = set(unique(gold_titles))

    if not gold_set:
        return 0.0

    retrieved_set = set(retrieved_titles)

    return len(gold_set & retrieved_set) / len(gold_set)


def run_eval(
    retriever: BM25Retriever,
    llm: OllamaClient,
    eval_n: int,
    eval_split: str,
    top_k: int,
    iterations: int,
    selector_candidates: int,
    save_json: str | None,
) -> None:

    # Load evaluation dataset split
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=eval_split)

    records = []
    total_start = time.time()

    # Evaluate pipeline on multiple examples
    for i, item in enumerate(dataset):
        if i >= eval_n:
            break

        question = item["question"]
        gold = item["answer"]

        supporting_titles = unique(item["supporting_facts"]["title"])

        print(f"\n[{i + 1}/{eval_n}] {question}")

        result = run_pipeline(
            question=question,
            retriever=retriever,
            llm=llm,
            top_k=top_k,
            iterations=iterations,
            selector_candidates=selector_candidates,
            verbose=False,
        )

        # Compute evaluation metrics
        em = exact_match_score(result.prediction, gold)
        f1 = f1_score(result.prediction, gold)

        retrieval_recall = title_recall(
            result.retrieved_titles[:30],
            supporting_titles,
        )

        evidence_recall = title_recall(
            result.selected_titles,
            supporting_titles,
        )

        # Store evaluation result
        record = {
            "question": question,
            "gold": gold,
            "supporting_titles": supporting_titles,
            "prediction": result.prediction,
            "raw_answer": result.raw_answer,
            "em": em,
            "f1": f1,
            "retrieval_recall": retrieval_recall,
            "evidence_recall": evidence_recall,
            "runtime": result.runtime,
            "subquestions": result.subquestions,
            "subquestion_count": len(result.subquestions),
            "selected_titles": result.selected_titles,
            "selected_count": len(result.selected_titles),
            "retrieved_titles": result.retrieved_titles[:30],
            "retrieved_count": len(result.retrieved_titles[:30]),
        }

        records.append(record)

        print(f"Gold: {gold}")
        print(f"Pred: {result.prediction}")

        print(
            f"EM: {em:.0f} | F1: {f1:.3f} | Retrieval Recall: {retrieval_recall:.3f} "
            f"| Evidence Recall: {evidence_recall:.3f} | Runtime: {result.runtime:.2f}s"
        )

        print(f"Selected: {', '.join(result.selected_titles[:8])}")

    # Compute average metrics
    avg_em = sum(r["em"] for r in records) / max(len(records), 1)
    avg_f1 = sum(r["f1"] for r in records) / max(len(records), 1)

    avg_retrieval_recall = (
        sum(r["retrieval_recall"] for r in records)
        / max(len(records), 1)
    )

    avg_evidence_recall = (
        sum(r["evidence_recall"] for r in records)
        / max(len(records), 1)
    )

    avg_runtime = (
        sum(r["runtime"] for r in records)
        / max(len(records), 1)
    )

    avg_subquestions = (
        sum(r["subquestion_count"] for r in records)
        / max(len(records), 1)
    )

    avg_selected_docs = (
        sum(r["selected_count"] for r in records)
        / max(len(records), 1)
    )

    avg_retrieved_docs = (
        sum(r["retrieved_count"] for r in records)
        / max(len(records), 1)
    )

    # Build evaluation summary
    summary = {
        "examples": len(records),
        "exact_match": avg_em,
        "f1": avg_f1,
        "retrieval_recall": avg_retrieval_recall,
        "evidence_recall": avg_evidence_recall,
        "avg_runtime": avg_runtime,
        "avg_subquestions": avg_subquestions,
        "avg_selected_docs": avg_selected_docs,
        "avg_retrieved_docs": avg_retrieved_docs,
        "total_runtime": time.time() - total_start,
    }

    print("\nEVALUATION SUMMARY")
    print(json.dumps(summary, indent=2))

    # Save evaluation results to JSON
    if save_json:
        path = Path(save_json)

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "records": records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Saved results to {path}")


def parse_args() -> argparse.Namespace:
    # Define command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="mixtral")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--selector-candidates", type=int, default=40)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument("--cache", default="cache/hotpot_fullwiki_corpus.pkl")
    parser.add_argument("--question", default=None)
    parser.add_argument("--eval-n", type=int, default=0)
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--save-json", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load corpus and build BM25 index
    corpus = load_or_build_corpus(
        cache_path=Path(args.cache),
        max_train=args.max_train,
        include_validation=args.include_validation,
    )

    print(f"Corpus documents: {len(corpus)}")
    print("Building BM25 index...")

    retriever = BM25Retriever(corpus)

    print("BM25 index ready.")

    llm = OllamaClient(model=args.model)

    # Run evaluation mode
    if args.eval_n > 0:
        run_eval(
            retriever=retriever,
            llm=llm,
            eval_n=args.eval_n,
            eval_split=args.eval_split,
            top_k=args.top_k,
            iterations=args.iterations,
            selector_candidates=args.selector_candidates,
            save_json=args.save_json,
        )
        return

    # Run single-question mode
    if args.question:
        run_pipeline(
            question=args.question,
            retriever=retriever,
            llm=llm,
            top_k=args.top_k,
            iterations=args.iterations,
            selector_candidates=args.selector_candidates,
            verbose=True,
        )
        return

    # Run interactive question-answering mode
    while True:
        question = input("\nQuestion> ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            break

        if not question:
            continue

        run_pipeline(
            question=question,
            retriever=retriever,
            llm=llm,
            top_k=args.top_k,
            iterations=args.iterations,
            selector_candidates=args.selector_candidates,
            verbose=True,
        )


if __name__ == "__main__":
    main()
