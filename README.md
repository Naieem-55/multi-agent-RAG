# RAGentA: Multi-Agent Retrieval-Augmented Generation for Attributed Question Answering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.16988-b31b1b.svg)](https://arxiv.org/abs/2506.16988)

RAGentA is a multi-agent retrieval-augmented generation (RAG) framework for attributed question answering. With the goal of trustworthy answer generation, RAGentA focuses on optimizing answer correctness, defined by coverage and relevance to the question and faithfulness, which measures the extent to which answers are grounded in retrieved documents.

## Quick Start

Get RAGentA running in 5 minutes with this minimal example:

```bash
# 1. Clone the repository
git clone https://github.com/Naieem-55/multi-agent-RAG.git
cd multi-agent-RAG

# 2. Create virtual environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# 3. Install dependencies (choose one)
pip install -r requirements.txt          # For API-based usage
pip install -r requirements_local.txt    # For local model inference

# 4. Set environment variables
export AWS_PROFILE=sigir-participant
export AWS_REGION=us-east-1

# 5. Run a simple query
python run_RAGentA.py --single_question "What is machine learning?"
```

### Requirements File Guide

| File | Use Case | Description |
|------|----------|-------------|
| `requirements.txt` | API-based inference | Lighter dependencies, uses external APIs for LLM |
| `requirements_local.txt` | Local GPU inference | Full dependencies including PyTorch CUDA support |

## Features

- **Multi-Agent Architecture**: Uses multiple specialized agents for document retrieval, relevance judgment, answer generation, and claim analysis
- **Hybrid Retrieval**: Combines semantic (dense) and keyword (sparse) search for better document retrieval
- **Citation Tracking**: Automatically tracks citations in generated answers to ensure factual accuracy
- **Claim Analysis**: Analyzes individual claims in answers to ensure relevance and identify knowledge gaps
- **Follow-Up Processing**: Generates follow-up questions for unanswered aspects and integrates additional knowledge
- **Evaluation Metrics**: Includes standard RAG evaluation metrics like MRR, Recall, Precision, and F1

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- CUDA-compatible GPU (recommended)
- AWS account with access to OpenSearch and Pinecone (for hybrid retrieval)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Naieem-55/multi-agent-RAG.git
cd multi-agent-RAG
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### AWS Configuration

RAGentA uses AWS services for document retrieval. You'll need to set up AWS credentials:

1. Create AWS credentials file:
```bash
mkdir -p ~/.aws
```

2. Add your credentials to `~/.aws/credentials`:
```
[sigir-participant]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

3. Add your region to `~/.aws/config`:
```
[profile sigir-participant]
region = us-east-1
output = json
```

### Environment Variables

Set the following environment variables:
```bash
export AWS_PROFILE=sigir-participant
export AWS_REGION=us-east-1
export HUGGING_FACE_HUB_TOKEN=your_hf_token  # If needed for accessing models
```

## Running RAGentA

RAGentA can be run on a single question or a batch of questions from a JSON/JSONL file.

### Process a Single Question
```bash
python run_RAGentA.py --model tiiuae/Falcon3-10B-Instruct --n 0.5 --alpha 0.65 --top_k 20 --single_question "Your question here?"
```

### Process Questions from a Dataset
```bash
python run_RAGentA.py --model tiiuae/Falcon3-10B-Instruct --n 0.5 --alpha 0.65 --top_k 20 --data_file your_questions.jsonl --output_format jsonl
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model name or path | `tiiuae/falcon-3-10b-instruct` |
| `--n` | Adjustment factor for adaptive judge bar | `0.5` |
| `--alpha` | Weight for semantic vs keyword search (0-1) | `0.65` |
| `--top_k` | Number of documents to retrieve | `20` |
| `--data_file` | File containing questions (JSON/JSONL) | - |
| `--single_question` | Process a single question | - |
| `--output_format` | Output format: json, jsonl, or debug | `jsonl` |
| `--output_dir` | Directory to save results | `results` |

## Input/Output Format

### Input
```json
{"id": "question_id", "question": "The question text?"}
```

### Output
```json
{
  "id": "question_id",
  "question": "The question text?",
  "passages": [{"passage": "Document content...", "doc_IDs": ["doc_id1"]}],
  "final_prompt": "Final prompt used for generation...",
  "answer": "Generated answer..."
}
```

## System Architecture

RAGentA uses a sophisticated multi-agent architecture:

```
Query → Hybrid Retrieval → Agent 1 (Predictor) → Agent 2 (Judge) → Agent 3 (Final-Predictor) → Agent 4 (Claim Judge) → Answer
              ↓                    ↓                   ↓                      ↓                        ↓
        Pinecone + OpenSearch   Per-doc answers    Relevance scoring    Cited answer           Gap analysis + Follow-up
```

### Agent Overview

| Agent | Purpose | Input | Output |
|-------|---------|-------|--------|
| Agent 1 (Predictor) | Generate candidate answers | Query + document | Document-specific answer |
| Agent 2 (Judge) | Evaluate document relevance | Query + document + answer | Relevance score |
| Agent 3 (Final-Predictor) | Generate comprehensive answer | Query + filtered docs | Answer with citations |
| Agent 4 (Claim Judge) | Analyze claims & detect gaps | Answer + claims | Improved answer + follow-ups |

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Pinecone connection failed** | Verify API key in environment variables; check network connectivity |
| **OpenSearch timeout** | Ensure AWS credentials are valid; check region configuration |
| **Out of memory** | Reduce `--top_k` or use a smaller model; try `requirements.txt` with API mode |
| **Model download fails** | Set `HUGGING_FACE_HUB_TOKEN`; check disk space (models ~10-20GB) |
| **CUDA not available** | Install PyTorch with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu118` |

### Validating Setup

Run the setup validation script:
```bash
python -c "
from retrieval.hybrid_retriever import HybridRetriever
retriever = HybridRetriever()
print('Setup validated successfully!')
"
```

## Evaluation

To evaluate RAG performance:
```python
from RAG_evaluation import evaluate_corpus_rag_mrr, evaluate_corpus_rag_recall

mrr_score = evaluate_corpus_rag_mrr(retrieved_docs_list, golden_docs_list, k=5)
recall_score = evaluate_corpus_rag_recall(retrieved_docs_list, golden_docs_list, k=20)
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{Besrour2025RAGentA,
  author       = {Ines Besrour and Jingbo He and Tobias Schreieder and Michael Färber},
  title        = {{RAGentA: Multi-Agent Retrieval-Augmented Generation for Attributed Question Answering}},
  year         = {2025},
  eprint       = {2506.16988},
  archivePrefix= {arXiv},
  primaryClass = {cs.IR},
  url          = {https://arxiv.org/abs/2506.16988},
}
```

## Acknowledgments

RAGentA draws inspiration from the [MAIN-RAG framework](https://arxiv.org/abs/2501.00332) by Chang et al.
