# Multi-hop QA Pipeline

## Overview

This project implements a multi-agent retrieval-augmented question answering pipeline for HotpotQA.

The system combines:

* BM25 document retrieval
* Question decomposition
* Query expansion
* Document reranking
* Evidence selection
* Multi-hop reasoning
* Source validation

The goal is to evaluate how different pipeline components affect retrieval quality, reasoning performance, and answer reliability in multi-hop question answering.

---

## Project Structure

```text
IKT469_Project/
│
├── multi_agent_system/
│   │
│   ├── experiments/
│   │   ├── experiment1.py
│   │   ├── experiment2.py
│   │   ├── experiment3.py
│   │   ├── experiment4.py
│   │   └── experiment5.py
│   │
│   ├── results/
│   │   ├── experiment1.json
│   │   ├── experiment2.json
│   │   ├── experiment3.json
│   │   ├── experiment4.json
│   │   └── experiment5.json
│   │
│   ├── ablation_analysis.ipynb
│   └── source_validation.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Experiments

| Experiment   | Description                                    |
| ------------ | ---------------------------------------------- |
| Experiment 1 | Full pipeline (baseline)                       |
| Experiment 2 | Removes question decomposition                 |
| Experiment 3 | Removes broad query expansion                  |
| Experiment 4 | Removes both decomposition and broad expansion |
| Experiment 5 | Removes source validation                      |

---

## Pipeline Components

* **BM25Retriever** — Retrieves candidate documents
* **QuestionAnalyzerAgent** — Creates subquestions
* **Query Expansion** — Expands retrieval queries
* **Heuristic Reranker** — Re-scores retrieved documents
* **SelectorAgent** — Selects relevant evidence
* **AdderAgent** — Adds missing supporting documents
* **AnswererAgent** — Generates final answers

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install and run Ollama:

```bash
ollama serve
```

Pull the model:

```bash
ollama pull mixtral
```

---

## Running Experiments

Example:

```bash
python multi_agent_system/experiments/experiment1.py \
  --eval-n 50 \
  --include-validation \
  --save-json results/experiment1.json
```

Run other experiments by replacing the filename.

---

## Analysis

* `ablation_analysis.ipynb`
  Compares Experiments 1–4.

* `source_validation.ipynb`
  Compares Experiment 1 and Experiment 5.

---

## Output

Each experiment produces a JSON file containing:

* Exact Match (EM)
* F1 score
* Retrieval recall
* Evidence recall
* Runtime statistics
* Per-question predictions
