"""
Local RAG Runner
Runs RAGentA with your own Pinecone account + local Elasticsearch.
No AWS dependencies required.
"""

import argparse
import json
import time
import datetime
import os
import logging
import random
import string


def get_unique_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"logs/local_rag_{timestamp}_{random_str}.log"


# Create logs directory
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(get_unique_log_filename()), logging.StreamHandler()],
)
logger = logging.getLogger("LocalRAG_Runner")


class LocalBasicRAG:
    """Basic RAG implementation using local hybrid retriever."""

    def __init__(self, retriever, agent_model=None, top_k=10):
        self.retriever = retriever
        self.top_k = top_k

        # Initialize LLM agent
        if isinstance(agent_model, str):
            from local_agent import LLMAgent
            self.agent = LLMAgent(agent_model)
            logger.info(f"Using local LLM agent with model {agent_model}")
        else:
            self.agent = agent_model
            logger.info("Using pre-initialized agent")

    def _create_rag_prompt(self, query, documents):
        """Create prompt for the LLM with retrieved documents."""
        docs_text = "\n\n".join(
            [f"Document {i+1}: {doc_text}" for i, (doc_text, _) in enumerate(documents)]
        )

        return f"""You are an accurate and helpful AI assistant. Answer the question based ONLY on the information provided in the documents below. Include citations like [1], [2] to reference the documents you used. If the documents don't contain the necessary information, state that you don't have enough information.

Documents:
{docs_text}

Question: {query}

Answer:"""

    def answer_query(self, query):
        """Process a query using basic RAG approach."""
        logger.info(f"Processing query: {query}")

        # Step 1: Retrieve documents
        logger.info(f"Retrieving top-{self.top_k} documents...")
        retrieved_docs = self.retriever.retrieve(query, top_k=self.top_k)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        # Step 2: Create prompt with documents
        prompt = self._create_rag_prompt(query, retrieved_docs)

        # Step 3: Generate answer
        logger.info("Generating answer...")
        answer = self.agent.generate(prompt)

        debug_info = {
            "raw_answer": answer,
            "retrieved_docs": retrieved_docs,
            "prompt": prompt
        }

        return answer, debug_info


def load_questions(file_path):
    """Load questions from JSON or JSONL file."""
    is_jsonl = file_path.lower().endswith(".jsonl")

    try:
        questions = []

        if is_jsonl:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        question = json.loads(line)
                        if "id" not in question:
                            question["id"] = line_num
                        questions.append(question)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    questions = data
                elif isinstance(data, dict):
                    questions = data.get("questions", [data])

                for i, q in enumerate(questions):
                    if "id" not in q:
                        q["id"] = i + 1

        logger.info(f"Loaded {len(questions)} questions")
        return questions

    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        return []


def format_result(result):
    """Format a result to match the required schema."""
    passages = []
    for doc_text, doc_id in result.get("retrieved_docs", []):
        passages.append({"passage": doc_text, "doc_IDs": [doc_id]})

    return {
        "id": result.get("id", 0),
        "question": result.get("question", ""),
        "passages": passages,
        "final_prompt": result.get("prompt", ""),
        "answer": result.get("model_answer", "")
    }


def main():
    parser = argparse.ArgumentParser(description="Local RAG with Pinecone + Elasticsearch")

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="HuggingFace model name (use smaller models for testing)"
    )

    # Retriever settings
    parser.add_argument(
        "--pinecone-api-key",
        type=str,
        default=None,
        help="Pinecone API key (or set PINECONE_API_KEY env var)"
    )
    parser.add_argument(
        "--pinecone-index",
        type=str,
        default="ragenta-index",
        help="Pinecone index name"
    )
    parser.add_argument(
        "--elasticsearch-url",
        type=str,
        default="http://localhost:9200",
        help="Elasticsearch URL"
    )
    parser.add_argument(
        "--elasticsearch-index",
        type=str,
        default="ragenta-docs",
        help="Elasticsearch index name"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Weight for semantic search (0-1)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )

    # Input/Output settings
    parser.add_argument(
        "--single_question",
        type=str,
        default=None,
        help="Process a single question"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="File containing questions (JSON or JSONL)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )

    args = parser.parse_args()

    # Initialize retriever
    logger.info("Initializing local hybrid retriever...")
    from local_hybrid_retriever import LocalHybridRetriever

    retriever = LocalHybridRetriever(
        pinecone_api_key=args.pinecone_api_key,
        pinecone_index_name=args.pinecone_index,
        elasticsearch_url=args.elasticsearch_url,
        elasticsearch_index_name=args.elasticsearch_index,
        alpha=args.alpha,
        top_k=args.top_k,
    )

    # Check if documents are indexed
    stats = retriever.get_stats()
    if stats["pinecone_vectors"] == 0 or stats["elasticsearch_docs"] == 0:
        logger.warning("No documents indexed! Run index_documents.py first.")
        logger.warning("Example: py index_documents.py --documents sample_documents.json")
        return

    logger.info(f"Found {stats['pinecone_vectors']} vectors in Pinecone, {stats['elasticsearch_docs']} docs in Elasticsearch")

    # Initialize RAG
    logger.info(f"Initializing RAG with model: {args.model}")
    rag = LocalBasicRAG(retriever, agent_model=args.model, top_k=args.top_k)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process single question
    if args.single_question:
        logger.info(f"\nProcessing: {args.single_question}")
        start_time = time.time()

        try:
            answer, debug_info = rag.answer_query(args.single_question)
            process_time = time.time() - start_time

            print("\n" + "=" * 60)
            print("QUESTION:", args.single_question)
            print("=" * 60)
            print("\nRETRIEVED DOCUMENTS:")
            for i, (text, doc_id) in enumerate(debug_info["retrieved_docs"], 1):
                print(f"\n[{i}] {doc_id}: {text[:200]}...")
            print("\n" + "-" * 60)
            print("ANSWER:", answer)
            print("-" * 60)
            print(f"Processing time: {process_time:.2f}s")

            # Save result
            result = {
                "id": "single_question",
                "question": args.single_question,
                "model_answer": answer,
                "retrieved_docs": debug_info["retrieved_docs"],
                "prompt": debug_info["prompt"],
                "process_time": process_time
            }

            output_file = os.path.join(args.output_dir, f"local_rag_result_{timestamp}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(format_result(result), f, indent=2)
            logger.info(f"Result saved to {output_file}")

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)

        return

    # Process questions from file
    if args.data_file:
        questions = load_questions(args.data_file)
        if not questions:
            logger.error("No questions found.")
            return

        results = []
        for i, item in enumerate(questions):
            logger.info(f"\nProcessing {i+1}/{len(questions)}: {item['question']}")
            start_time = time.time()

            try:
                answer, debug_info = rag.answer_query(item["question"])
                process_time = time.time() - start_time

                result = {
                    "id": item.get("id", i + 1),
                    "question": item["question"],
                    "model_answer": answer,
                    "retrieved_docs": debug_info["retrieved_docs"],
                    "prompt": debug_info["prompt"],
                    "process_time": process_time
                }
                results.append(result)

                logger.info(f"Answer: {answer[:100]}...")
                logger.info(f"Time: {process_time:.2f}s")

            except Exception as e:
                logger.error(f"Error: {e}")

        # Save results
        if results:
            output_file = os.path.join(args.output_dir, f"local_rag_results_{timestamp}.jsonl")
            with open(output_file, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(format_result(result)) + "\n")
            logger.info(f"Results saved to {output_file}")

        return

    # Interactive mode if no input specified
    print("\n" + "=" * 60)
    print("LOCAL RAG - Interactive Mode")
    print("Type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            question = input("\nYour question: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                break
            if not question:
                continue

            answer, debug_info = rag.answer_query(question)

            print("\nRetrieved documents:")
            for i, (text, doc_id) in enumerate(debug_info["retrieved_docs"], 1):
                print(f"  [{i}] {text[:100]}...")

            print(f"\nAnswer: {answer}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
