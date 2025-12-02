# RAGentA System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RAGentA SYSTEM                                      │
│                                                                                  │
│  ┌──────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │  USER    │───>│ HYBRID RETRIEVER│───>│  MULTI-AGENT    │───>│   ANSWER    │ │
│  │ QUESTION │    │   (Pinecone +   │    │   PIPELINE      │    │   WITH      │ │
│  │          │    │  Elasticsearch) │    │  (4 Agents)     │    │  CITATIONS  │ │
│  └──────────┘    └─────────────────┘    └─────────────────┘    └─────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│                            ┌─────────────────┐                                  │
│                            │   User Query    │                                  │
│                            └────────┬────────┘                                  │
│                                     │                                           │
│                                     ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                        HYBRID RETRIEVER                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐ │  │
│  │  │                                                                      │ │  │
│  │  │   ┌─────────────────┐              ┌─────────────────┐              │ │  │
│  │  │   │    PINECONE     │              │  ELASTICSEARCH  │              │ │  │
│  │  │   │  (Vector DB)    │              │  (Keyword Search)│              │ │  │
│  │  │   │                 │              │                 │              │ │  │
│  │  │   │  ┌───────────┐  │              │  ┌───────────┐  │              │ │  │
│  │  │   │  │ e5-base-v2│  │              │  │   BM25    │  │              │ │  │
│  │  │   │  │ Embeddings│  │              │  │  Ranking  │  │              │ │  │
│  │  │   │  └───────────┘  │              │  └───────────┘  │              │ │  │
│  │  │   │                 │              │                 │              │ │  │
│  │  │   │ Semantic Score  │              │ Keyword Score   │              │ │  │
│  │  │   └────────┬────────┘              └────────┬────────┘              │ │  │
│  │  │            │                                │                       │ │  │
│  │  │            └───────────┬───────────────────┘                       │ │  │
│  │  │                        ▼                                            │ │  │
│  │  │            ┌─────────────────────┐                                  │ │  │
│  │  │            │   HYBRID SCORING    │                                  │ │  │
│  │  │            │                     │                                  │ │  │
│  │  │            │ Score = α×Semantic  │                                  │ │  │
│  │  │            │      + (1-α)×BM25   │                                  │ │  │
│  │  │            │                     │                                  │ │  │
│  │  │            │   (α = 0.65 default)│                                  │ │  │
│  │  │            └──────────┬──────────┘                                  │ │  │
│  │  │                       │                                             │ │  │
│  │  └───────────────────────┼─────────────────────────────────────────────┘ │  │
│  │                          │                                                │  │
│  │                          ▼                                                │  │
│  │                 ┌─────────────────┐                                       │  │
│  │                 │  Top-K Documents │                                      │  │
│  │                 │    (k = 20)      │                                      │  │
│  │                 └────────┬────────┘                                       │  │
│  └──────────────────────────┼────────────────────────────────────────────────┘  │
│                             │                                                    │
│                             ▼                                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Agent Pipeline (RAGentA)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          MULTI-AGENT PIPELINE                                    │
│                                                                                  │
│  Retrieved Documents (Top-K)                                                     │
│         │                                                                        │
│         ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         AGENT 1: PREDICTOR                               │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │  For each document:                                              │    │    │
│  │  │    • Generate candidate answer based on single document          │    │    │
│  │  │    • Input: Query + Document[i]                                  │    │    │
│  │  │    • Output: Candidate Answer[i]                                 │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  └──────────────────────────────────┬──────────────────────────────────────┘    │
│                                     │                                            │
│                                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                          AGENT 2: JUDGE                                  │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │  For each document:                                              │    │    │
│  │  │    • Calculate relevance score using log probabilities           │    │    │
│  │  │    • Score = log_prob("Yes") - log_prob("No")                    │    │    │
│  │  │                                                                  │    │    │
│  │  │  Adaptive Threshold:                                             │    │    │
│  │  │    • τq = mean(scores)                                           │    │    │
│  │  │    • adjusted_τq = τq - n × std(scores)                          │    │    │
│  │  │    • Filter: Keep docs where score >= adjusted_τq                │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  └──────────────────────────────────┬──────────────────────────────────────┘    │
│                                     │                                            │
│                                     ▼                                            │
│                          Filtered Documents                                      │
│                                     │                                            │
│                                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                     AGENT 3: FINAL PREDICTOR                             │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │  • Synthesize final answer from filtered documents               │    │    │
│  │  │  • Add citations in [X] format                                   │    │    │
│  │  │  • Input: Query + All Filtered Documents                         │    │    │
│  │  │  • Output: Answer with Citations                                 │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  └──────────────────────────────────┬──────────────────────────────────────┘    │
│                                     │                                            │
│                                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    AGENT 4: CLAIM JUDGE (Enhanced)                       │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │  1. Extract claims from answer                                   │    │    │
│  │  │  2. Analyze question structure (single/multi-component)          │    │    │
│  │  │  3. Map claims to question components                            │    │    │
│  │  │  4. Assess coverage (FULLY/PARTIALLY/NOT ANSWERED)               │    │    │
│  │  │  5. Generate follow-up questions for gaps                        │    │    │
│  │  │  6. Retrieve additional documents if needed                      │    │    │
│  │  │  7. Integrate new information into final answer                  │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  └──────────────────────────────────┬──────────────────────────────────────┘    │
│                                     │                                            │
│                                     ▼                                            │
│                         ┌───────────────────┐                                    │
│                         │   FINAL ANSWER    │                                    │
│                         │  WITH CITATIONS   │                                    │
│                         └───────────────────┘                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                           │
│                                                                                  │
│   INPUT                    PROCESSING                           OUTPUT          │
│                                                                                  │
│  ┌────────┐     ┌─────────────────────────────────────┐     ┌─────────────┐    │
│  │Question│────>│         Embedding Model              │────>│Query Vector │    │
│  │  Text  │     │        (e5-base-v2)                 │     │  (768-dim)  │    │
│  └────────┘     └─────────────────────────────────────┘     └──────┬──────┘    │
│                                                                     │           │
│                 ┌───────────────────────────────────────────────────┼──────┐    │
│                 │                                                   │      │    │
│                 ▼                                                   ▼      │    │
│          ┌─────────────┐                                    ┌─────────────┐│    │
│          │  Pinecone   │                                    │Elasticsearch││    │
│          │   Query     │                                    │   Query     ││    │
│          └──────┬──────┘                                    └──────┬──────┘│    │
│                 │                                                  │       │    │
│                 ▼                                                  ▼       │    │
│          ┌─────────────┐                                    ┌─────────────┐│    │
│          │  Semantic   │                                    │  Keyword    ││    │
│          │  Results    │                                    │  Results    ││    │
│          └──────┬──────┘                                    └──────┬──────┘│    │
│                 │                                                  │       │    │
│                 └──────────────────┬───────────────────────────────┘       │    │
│                                    │                                        │    │
│                                    ▼                                        │    │
│                            ┌─────────────┐                                  │    │
│                            │   Merge &   │                                  │    │
│                            │   Re-rank   │                                  │    │
│                            └──────┬──────┘                                  │    │
│                                   │                                         │    │
│                                   ▼                                         │    │
│                           ┌──────────────┐                                  │    │
│                           │ Top-K Docs   │                                  │    │
│                           │  + Scores    │                                  │    │
│                           └──────┬───────┘                                  │    │
│                                  │                                          │    │
│                                  ▼                                          │    │
│                          ┌──────────────┐                                   │    │
│                          │  LLM Agent   │                                   │    │
│                          │  Pipeline    │                                   │    │
│                          └──────┬───────┘                                   │    │
│                                 │                                           │    │
│                                 ▼                                           │    │
│                          ┌──────────────┐                                   │    │
│                          │ Final Answer │                                   │    │
│                          │ + Citations  │                                   │    │
│                          └──────────────┘                                   │    │
│                                                                             │    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure & Components

```
RAGentA/
│
├── Core Components
│   │
│   ├── RAGentA.py                 # Main multi-agent RAG system
│   │   ├── class RAGENTA          # Orchestrates 4-agent pipeline
│   │   └── class EnhancedAgent4   # Claim analysis & follow-ups
│   │
│   ├── local_agent.py             # Local LLM inference
│   │   └── class LLMAgent         # HuggingFace model wrapper
│   │       ├── generate()         # Text generation
│   │       ├── get_log_probs()    # Yes/No probability scoring
│   │       └── batch_process()    # Batch inference
│   │
│   └── api_agent.py               # API-based inference
│       └── class FalconAgent      # AI71 API wrapper
│
├── Retrievers
│   │
│   ├── hybrid_retriever.py        # AWS-based retriever (original)
│   │   └── class HybridRetriever  # Pinecone + OpenSearch (AWS)
│   │
│   └── local_hybrid_retriever.py  # Local retriever (no AWS)
│       └── class LocalHybridRetriever
│           ├── index_documents()  # Index docs to both DBs
│           ├── retrieve()         # Hybrid search
│           └── _embed_query()     # Generate embeddings
│
├── Runners
│   │
│   ├── run_RAGentA.py             # Full multi-agent RAG runner
│   ├── run_BASIC_RAG.py           # Simple RAG (no multi-agent)
│   └── run_local_rag.py           # Local RAG runner (no AWS)
│
├── Utilities
│   │
│   ├── index_documents.py         # Document indexing script
│   └── sample_documents.json      # Sample documents for testing
│
└── Evaluation
    │
    └── evaluation/
        ├── RAG_evaluation/        # Answer quality metrics
        └── retrieval_evaluation/  # Retrieval metrics (MRR, Recall)
```

---

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            TECHNOLOGY STACK                                      │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                          APPLICATION LAYER                               │    │
│  │                                                                          │    │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │    │
│  │   │ run_RAGentA  │  │run_BASIC_RAG │  │run_local_rag │                  │    │
│  │   │    .py       │  │    .py       │  │    .py       │                  │    │
│  │   └──────────────┘  └──────────────┘  └──────────────┘                  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                           CORE RAG LAYER                                 │    │
│  │                                                                          │    │
│  │   ┌──────────────────────────────────────────────────────────────┐      │    │
│  │   │                    RAGentA.py                                 │      │    │
│  │   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────────┐│      │    │
│  │   │  │ Agent 1 │ │ Agent 2 │ │ Agent 3 │ │ Agent 4 (Enhanced)  ││      │    │
│  │   │  │Predictor│ │  Judge  │ │ Final   │ │ Claim Analysis      ││      │    │
│  │   │  └─────────┘ └─────────┘ └─────────┘ └─────────────────────┘│      │    │
│  │   └──────────────────────────────────────────────────────────────┘      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                          RETRIEVAL LAYER                                 │    │
│  │                                                                          │    │
│  │   ┌─────────────────────────┐    ┌─────────────────────────┐            │    │
│  │   │   hybrid_retriever.py   │    │local_hybrid_retriever.py│            │    │
│  │   │      (AWS-based)        │    │    (Local/No AWS)       │            │    │
│  │   └─────────────────────────┘    └─────────────────────────┘            │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                          STORAGE LAYER                                   │    │
│  │                                                                          │    │
│  │   ┌─────────────────────────┐    ┌─────────────────────────┐            │    │
│  │   │       PINECONE          │    │     ELASTICSEARCH       │            │    │
│  │   │    (Vector Store)       │    │    (Keyword Index)      │            │    │
│  │   │                         │    │                         │            │    │
│  │   │  • Semantic Search      │    │  • BM25 Ranking         │            │    │
│  │   │  • Cosine Similarity    │    │  • Full-text Search     │            │    │
│  │   │  • 768-dim vectors      │    │  • Inverted Index       │            │    │
│  │   └─────────────────────────┘    └─────────────────────────┘            │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                            ML LAYER                                      │    │
│  │                                                                          │    │
│  │   ┌─────────────────────────┐    ┌─────────────────────────┐            │    │
│  │   │    EMBEDDING MODEL      │    │       LLM MODEL         │            │    │
│  │   │    (e5-base-v2)         │    │  (Falcon/Qwen/etc.)     │            │    │
│  │   │                         │    │                         │            │    │
│  │   │  • Query Embedding      │    │  • Text Generation      │            │    │
│  │   │  • Document Embedding   │    │  • Log Probabilities    │            │    │
│  │   │  • 768 dimensions       │    │  • Chat Templates       │            │    │
│  │   └─────────────────────────┘    └─────────────────────────┘            │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         FRAMEWORK LAYER                                  │    │
│  │                                                                          │    │
│  │   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐           │    │
│  │   │  PyTorch   │ │Transformers│ │ Sentence   │ │ Pinecone   │           │    │
│  │   │            │ │(HuggingFace│ │Transformers│ │  Client    │           │    │
│  │   └────────────┘ └────────────┘ └────────────┘ └────────────┘           │    │
│  │                                                                          │    │
│  │   ┌────────────┐ ┌────────────┐ ┌────────────┐                          │    │
│  │   │Elasticsearch│ │   NumPy   │ │    tqdm    │                          │    │
│  │   │   Client   │ │            │ │            │                          │    │
│  │   └────────────┘ └────────────┘ └────────────┘                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Sequence Diagram: Query Processing

```
User          Runner         Retriever       Pinecone     Elasticsearch      LLM
 │               │               │              │              │              │
 │  Question     │               │              │              │              │
 │──────────────>│               │              │              │              │
 │               │               │              │              │              │
 │               │  retrieve()   │              │              │              │
 │               │──────────────>│              │              │              │
 │               │               │              │              │              │
 │               │               │ embed_query()│              │              │
 │               │               │─────────────>│              │              │
 │               │               │              │              │              │
 │               │               │   query()    │              │              │
 │               │               │─────────────>│              │              │
 │               │               │<─────────────│              │              │
 │               │               │ semantic_results            │              │
 │               │               │              │              │              │
 │               │               │         search()            │              │
 │               │               │────────────────────────────>│              │
 │               │               │<────────────────────────────│              │
 │               │               │         keyword_results     │              │
 │               │               │              │              │              │
 │               │               │ merge & rank │              │              │
 │               │               │──────────────│              │              │
 │               │<──────────────│              │              │              │
 │               │  top_k_docs   │              │              │              │
 │               │               │              │              │              │
 │               │                    Agent 1: generate()      │              │
 │               │─────────────────────────────────────────────────────────>│
 │               │<─────────────────────────────────────────────────────────│
 │               │                    candidate_answers        │              │
 │               │               │              │              │              │
 │               │                    Agent 2: get_log_probs() │              │
 │               │─────────────────────────────────────────────────────────>│
 │               │<─────────────────────────────────────────────────────────│
 │               │                    relevance_scores         │              │
 │               │               │              │              │              │
 │               │  filter by    │              │              │              │
 │               │  threshold    │              │              │              │
 │               │───────────────│              │              │              │
 │               │               │              │              │              │
 │               │                    Agent 3: generate()      │              │
 │               │─────────────────────────────────────────────────────────>│
 │               │<─────────────────────────────────────────────────────────│
 │               │                    answer_with_citations    │              │
 │               │               │              │              │              │
 │               │                    Agent 4: analyze_claims()│              │
 │               │─────────────────────────────────────────────────────────>│
 │               │<─────────────────────────────────────────────────────────│
 │               │                    final_answer             │              │
 │               │               │              │              │              │
 │<──────────────│               │              │              │              │
 │ Final Answer  │               │              │              │              │
 │ + Citations   │               │              │              │              │
 │               │               │              │              │              │
```

---

## Key Algorithms

### 1. Hybrid Scoring
```
final_score = α × semantic_score + (1 - α) × keyword_score

Where:
  α = 0.65 (default) - weight for semantic search
  semantic_score = normalized cosine similarity from Pinecone
  keyword_score = normalized BM25 score from Elasticsearch
```

### 2. Adaptive Judge Threshold
```
τq = mean(all_document_scores)
σ = std(all_document_scores)
adjusted_τq = τq - n × σ

Where:
  n = 0.5 (default) - adjustment factor
  Documents with score >= adjusted_τq are kept
```

### 3. Relevance Scoring (Agent 2)
```
relevance_score = log_prob("Yes") - log_prob("No")

Prompt: "Is this document relevant to answering the question?"
Higher score = more relevant document
```

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.65 | Semantic vs keyword weight |
| `top_k` | 20 | Documents to retrieve |
| `n` | 0.5 | Threshold adjustment factor |
| `model` | Falcon-3-10B | LLM for generation |
| `embedding_model` | e5-base-v2 | Text embedding model |

---

## Deployment Options

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DEPLOYMENT OPTIONS                                      │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  OPTION 1: AWS-Based (Original)                                          │    │
│  │                                                                          │    │
│  │  • Pinecone (managed) + AWS OpenSearch                                   │    │
│  │  • Requires AWS credentials (SIGIR participant)                          │    │
│  │  • Use: hybrid_retriever.py + run_RAGentA.py                            │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  OPTION 2: Local Setup (Your Own Infrastructure)                         │    │
│  │                                                                          │    │
│  │  • Pinecone (free tier) + Local Elasticsearch                            │    │
│  │  • No AWS required                                                       │    │
│  │  • Use: local_hybrid_retriever.py + run_local_rag.py                    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  OPTION 3: Fully Local (Future)                                          │    │
│  │                                                                          │    │
│  │  • FAISS (local vectors) + Local Elasticsearch                           │    │
│  │  • Completely offline capable                                            │    │
│  │  • Requires custom implementation                                        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```
