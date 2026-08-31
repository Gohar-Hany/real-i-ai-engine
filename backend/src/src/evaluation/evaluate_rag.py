import os
import json
import sys
import time
import math
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

# Ensure the src directory is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.config import get_settings
from controllers.NLPController import NLPController
from models.ProjectModel import ProjectModel
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser

# Standard pure-Python ROUGE-L LCS implementation to avoid heavy dependency errors
def lcs(x, y):
    m, n = len(x), len(y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif x[i-1] == y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]

def calculate_rouge_l(gen_ans: str, gt_ans: str) -> float:
    # Tokenize simple split
    gen_tokens = gen_ans.lower().split()
    gt_tokens = gt_ans.lower().split()
    if not gen_tokens or not gt_tokens:
        return 0.0
    lcs_len = lcs(gen_tokens, gt_tokens)
    prec = lcs_len / len(gen_tokens)
    rec = lcs_len / len(gt_tokens)
    if (prec + rec) == 0:
        return 0.0
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def get_llm_score(llm_client, system_prompt: str, user_prompt: str) -> float:
    try:
        response = llm_client.generate_text(
            prompt=f"{system_prompt}\n\nUser Input:\n{user_prompt}",
            chat_history=[],
            max_output_tokens=50,
            temperature=0.0
        )
        # Find any number between 0 and 1 or scale of 0 to 10 in the response
        match = re.search(r"([0-9]\.[0-9]+|[0-9]+)", response)
        if match:
            score = float(match.group(1))
            if score > 1.0 and score <= 10.0:
                score /= 10.0
            return min(max(score, 0.0), 1.0)
        return 0.5
    except Exception as e:
        print(f"Error calling LLM-as-a-judge: {e}")
        return 0.5

import re

def main():
    print("Initializing Evaluation System...")
    load_dotenv()
    settings = get_settings()

    # Load test dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"Error: dataset file {dataset_path} does not exist. Please run generate_dataset.py first.")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_dataset = json.load(f)

    print(f"Loaded {len(test_dataset)} Q&A pairs for evaluation.")

    # Initialize components
    llm_factory = LLMProviderFactory(settings)
    vectordb_factory = VectorDBProviderFactory(settings)

    generation_client = llm_factory.create(provider=settings.GENERATION_BACKEND)
    generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    embedding_client = llm_factory.create(provider=settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID, 
        embedding_size=settings.EMBEDDING_MODEL_SIZE
    )

    vectordb_client = vectordb_factory.create(provider=settings.VECTOR_DB_BACKEND)
    vectordb_client.connect()

    print("Loading reranker BGE model...")
    reranker_client = CrossEncoder('BAAI/bge-reranker-v2-m3')

    template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )

    # NLPController instances (one with reranker, one without for comparison!)
    nlp_controller_rerank = NLPController(
        vectordb_client=vectordb_client,
        generation_client=generation_client,
        embedding_client=embedding_client,
        template_parser=template_parser,
        reranker_client=reranker_client
    )

    nlp_controller_no_rerank = NLPController(
        vectordb_client=vectordb_client,
        generation_client=generation_client,
        embedding_client=embedding_client,
        template_parser=template_parser,
        reranker_client=None
    )

    # Need Mongo to fetch Project object representation
    import pymongo
    mongo_client = pymongo.MongoClient(settings.MONGODB_URL)
    db = mongo_client[settings.MONGODB_DATABASE]
    proj_doc = db["projects"].find_one({"project_id": "testproject1"})
    
    # Wrap in Project Pydantic model
    from models.db_schemes import Project
    project = Project(
        project_id=proj_doc["project_id"]
    )
    # Set internal DB _id for operations that query MongoDB using it
    project.id = proj_doc["_id"]

    results = []

    # LLM-as-a-judge prompts
    faithfulness_prompt = """You are an evaluator. Rate the FAITHFULNESS of a generated answer on a scale from 0.0 to 1.0.
Faithfulness measures if the generated answer is strictly supported by the provided retrieved context without hallucinating external information.
Output ONLY a single numerical score (e.g. 0.95 or 0.8) without explanations.

Context:
{context}

Generated Answer:
{answer}"""

    relevancy_prompt = """You are an evaluator. Rate the ANSWER RELEVANCY of a generated answer on a scale from 0.0 to 1.0.
Answer Relevancy measures if the generated answer directly addresses the user's question. It should not penalize factual incorrectness, only how relevant the topic of the answer is to the question.
Output ONLY a single numerical score (e.g. 0.95 or 0.8) without explanations.

Question:
{question}

Generated Answer:
{answer}"""

    context_relevancy_prompt = """You are an evaluator. Rate the CONTEXT RELEVANCY of the retrieved chunks on a scale from 0.0 to 1.0.
Context Relevancy measures if the retrieved context is relevant, clean, and directly helps answer the user's question, containing minimal redundant or irrelevant information.
Output ONLY a single numerical score (e.g. 0.9) without explanations.

Question:
{question}

Retrieved Context:
{context}"""

    # Cumulative metrics variables
    retrieval_stats = {
        "with_rerank": {"hit@1": 0, "hit@3": 0, "hit@5": 0, "mrr": 0.0, "ndcg@3": 0.0, "ndcg@5": 0.0},
        "no_rerank": {"hit@1": 0, "hit@3": 0, "hit@5": 0, "mrr": 0.0, "ndcg@3": 0.0, "ndcg@5": 0.0}
    }
    
    generation_stats = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_relevancy": [],
        "rouge_l": [],
        "semantic_similarity": []
    }

    raw_llm_stats = {
        "rouge_l": [],
        "semantic_similarity": []
    }

    latency_stats = {
        "retrieval": [],
        "generation": [],
        "total": []
    }

    print("\nStarting evaluation of Q&A dataset...")
    for idx, item in enumerate(test_dataset):
        q_id = item["id"]
        question = item["question"]
        gt_answer = item["ground_truth_answer"]
        gt_chunk = item["chunk_text"]

        print(f"\n[{idx+1}/{len(test_dataset)}] Evaluating: '{question[:50]}...'")

        # ─── RETRIEVAL ───
        # Measure Latency of retrieval with rerank
        start_ret = time.time()
        docs_with_rerank = nlp_controller_rerank.search_vector_db_collection(project=project, text=question, limit=5) or []
        ret_latency = time.time() - start_ret
        latency_stats["retrieval"].append(ret_latency)

        docs_no_rerank = nlp_controller_no_rerank.search_vector_db_collection(project=project, text=question, limit=5) or []

        # Find target rank (1-indexed)
        def find_rank(retrieved_docs, target_text):
            for rank_idx, doc in enumerate(retrieved_docs):
                # exact or substring match
                if target_text.strip() in doc.text.strip() or doc.text.strip() in target_text.strip():
                    return rank_idx + 1
            return 0

        rank_with_rerank = find_rank(docs_with_rerank, gt_chunk)
        rank_no_rerank = find_rank(docs_no_rerank, gt_chunk)

        # Calculate metrics function
        def calc_retrieval_metrics(rank):
            hit1 = 1 if rank == 1 else 0
            hit3 = 1 if (0 < rank <= 3) else 0
            hit5 = 1 if (0 < rank <= 5) else 0
            mrr = 1.0 / rank if rank > 0 else 0.0
            ndcg3 = 1.0 / math.log2(rank + 1) if (0 < rank <= 3) else 0.0
            ndcg5 = 1.0 / math.log2(rank + 1) if (0 < rank <= 5) else 0.0
            return hit1, hit3, hit5, mrr, ndcg3, ndcg5

        # With rerank stats
        h1, h3, h5, mrr, ndcg3, ndcg5 = calc_retrieval_metrics(rank_with_rerank)
        retrieval_stats["with_rerank"]["hit@1"] += h1
        retrieval_stats["with_rerank"]["hit@3"] += h3
        retrieval_stats["with_rerank"]["hit@5"] += h5
        retrieval_stats["with_rerank"]["mrr"] += mrr
        retrieval_stats["with_rerank"]["ndcg@3"] += ndcg3
        retrieval_stats["with_rerank"]["ndcg@5"] += ndcg5

        # No rerank stats
        h1_nr, h3_nr, h5_nr, mrr_nr, ndcg3_nr, ndcg5_nr = calc_retrieval_metrics(rank_no_rerank)
        retrieval_stats["no_rerank"]["hit@1"] += h1_nr
        retrieval_stats["no_rerank"]["hit@3"] += h3_nr
        retrieval_stats["no_rerank"]["hit@5"] += h5_nr
        retrieval_stats["no_rerank"]["mrr"] += mrr_nr
        retrieval_stats["no_rerank"]["ndcg@3"] += ndcg3_nr
        retrieval_stats["no_rerank"]["ndcg@5"] += ndcg5_nr

        print(f"  Retrieval Rank: With Reranker = {rank_with_rerank}, Without Reranker = {rank_no_rerank}")

        # ─── GENERATION ───
        # Full RAG Pipeline Call
        start_gen = time.time()
        answer, full_prompt, chat_history = nlp_controller_rerank.answer_rag_question(project=project, query=question, limit=5)
        gen_latency = time.time() - start_gen
        latency_stats["generation"].append(gen_latency)
        latency_stats["total"].append(ret_latency + gen_latency)

        # Baseline: Raw LLM Answer (without RAG context)
        raw_prompt = f"Answer the following question directly and concisely based on your general knowledge. Question: {question}"
        raw_answer = generation_client.generate_text(prompt=raw_prompt, chat_history=[], max_output_tokens=200)

        # Calculate scores
        # 1. ROUGE-L F1
        rouge_rag = calculate_rouge_l(answer, gt_answer)
        rouge_raw = calculate_rouge_l(raw_answer, gt_answer)
        generation_stats["rouge_l"].append(rouge_rag)
        raw_llm_stats["rouge_l"].append(rouge_raw)

        # 2. Semantic Similarity via GTE Embeddings
        try:
            emb_gen = embedding_client.embed_text(answer)
            emb_gt = embedding_client.embed_text(gt_answer)
            emb_raw = embedding_client.embed_text(raw_answer)
            
            similarity_rag = cosine_similarity(emb_gen, emb_gt)
            similarity_raw = cosine_similarity(emb_raw, emb_gt)
        except Exception as e:
            similarity_rag = 0.0
            similarity_raw = 0.0
            print(f"  Embedding error: {e}")

        generation_stats["semantic_similarity"].append(similarity_rag)
        raw_llm_stats["semantic_similarity"].append(similarity_raw)

        # 3. LLM-as-a-Judge Evaluation (only for RAG output)
        context_str = "\n\n".join([f"Chunk {i+1}: {doc.text}" for i, doc in enumerate(docs_with_rerank)])
        
        faithfulness_score = get_llm_score(
            generation_client, 
            faithfulness_prompt.format(context=context_str, answer=answer), 
            ""
        )
        relevancy_score = get_llm_score(
            generation_client, 
            relevancy_prompt.format(question=question, answer=answer), 
            ""
        )
        context_relevancy_score = get_llm_score(
            generation_client, 
            context_relevancy_prompt.format(question=question, context=context_str), 
            ""
        )

        generation_stats["faithfulness"].append(faithfulness_score)
        generation_stats["answer_relevancy"].append(relevancy_score)
        generation_stats["context_relevancy"].append(context_relevancy_score)

        print(f"  RAG Scores: Faithfulness={faithfulness_score:.2f}, Relevancy={relevancy_score:.2f}, ROUGE-L={rouge_rag:.2f}, Semantic Sim={similarity_rag:.2f}")
        print(f"  Raw LLM Scores: ROUGE-L={rouge_raw:.2f}, Semantic Sim={similarity_raw:.2f}")
        print(f"  Latency: Retrieval={ret_latency:.2f}s, Generation={gen_latency:.2f}s")

        results.append({
            "id": q_id,
            "question": question,
            "ground_truth_answer": gt_answer,
            "rag_answer": answer,
            "raw_answer": raw_answer,
            "retrieval": {
                "rank_with_rerank": rank_with_rerank,
                "rank_no_rerank": rank_no_rerank,
                "retrieved_docs_count": len(docs_with_rerank)
            },
            "scores": {
                "faithfulness": faithfulness_score,
                "answer_relevancy": relevancy_score,
                "context_relevancy": context_relevancy_score,
                "rouge_l_rag": rouge_rag,
                "rouge_l_raw": rouge_raw,
                "semantic_similarity_rag": similarity_rag,
                "semantic_similarity_raw": similarity_raw
            },
            "latency": {
                "retrieval": ret_latency,
                "generation": gen_latency,
                "total": ret_latency + gen_latency
            }
        })

    # Average the metrics
    num_queries = len(test_dataset)
    
    # Retrieval averages
    for config in ["with_rerank", "no_rerank"]:
        for metric in ["hit@1", "hit@3", "hit@5", "mrr", "ndcg@3", "ndcg@5"]:
            retrieval_stats[config][metric] /= num_queries

    final_evaluation = {
        "total_queries_evaluated": num_queries,
        "averages": {
            "retrieval": retrieval_stats,
            "rag_generation": {
                "faithfulness": np.mean(generation_stats["faithfulness"]),
                "answer_relevancy": np.mean(generation_stats["answer_relevancy"]),
                "context_relevancy": np.mean(generation_stats["context_relevancy"]),
                "rouge_l": np.mean(generation_stats["rouge_l"]),
                "semantic_similarity": np.mean(generation_stats["semantic_similarity"])
            },
            "raw_llm_generation": {
                "rouge_l": np.mean(raw_llm_stats["rouge_l"]),
                "semantic_similarity": np.mean(raw_llm_stats["semantic_similarity"])
            },
            "latency": {
                "avg_retrieval_seconds": np.mean(latency_stats["retrieval"]),
                "avg_generation_seconds": np.mean(latency_stats["generation"]),
                "avg_total_seconds": np.mean(latency_stats["total"]),
                "p95_total_seconds": np.percentile(latency_stats["total"], 95)
            }
        },
        "queries_detail": results
    }

    # Save final results
    results_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_evaluation, f, indent=4, ensure_ascii=False)

    print(f"\nEvaluation Complete! Results saved to {results_path} ✓")
    
    # Print high-level overview
    print("\n" + "="*50)
    print("              EVALUATION SUMMARY")
    print("="*50)
    print("RETRIEVAL PERFORMANCE:")
    print(f"  Metric       | With BGE Reranker | Without Reranker")
    print(f"  Hit Rate@1   | {retrieval_stats['with_rerank']['hit@1']*100:16.1f}% | {retrieval_stats['no_rerank']['hit@1']*100:15.1f}%")
    print(f"  Hit Rate@3   | {retrieval_stats['with_rerank']['hit@3']*100:16.1f}% | {retrieval_stats['no_rerank']['hit@3']*100:15.1f}%")
    print(f"  Hit Rate@5   | {retrieval_stats['with_rerank']['hit@5']*100:16.1f}% | {retrieval_stats['no_rerank']['hit@5']*100:15.1f}%")
    print(f"  MRR          | {retrieval_stats['with_rerank']['mrr']:.4f}           | {retrieval_stats['no_rerank']['mrr']:.4f}")
    print(f"  NDCG@5       | {retrieval_stats['with_rerank']['ndcg@5']:.4f}           | {retrieval_stats['no_rerank']['ndcg@5']:.4f}")
    print("-" * 50)
    print("GENERATION QUALITY:")
    print(f"  Faithfulness (LLM-Judge)    : {final_evaluation['averages']['rag_generation']['faithfulness']*100:.1f}%")
    print(f"  Answer Relevancy (LLM-Judge): {final_evaluation['averages']['rag_generation']['answer_relevancy']*100:.1f}%")
    print(f"  Context Relevancy (LLM-Judge): {final_evaluation['averages']['rag_generation']['context_relevancy']*100:.1f}%")
    print(f"  ROUGE-L RAG vs Raw LLM      : {final_evaluation['averages']['rag_generation']['rouge_l']:.4f} vs {final_evaluation['averages']['raw_llm_generation']['rouge_l']:.4f}")
    print(f"  Semantic Similarity RAG/Raw : {final_evaluation['averages']['rag_generation']['semantic_similarity']:.4f} vs {final_evaluation['averages']['raw_llm_generation']['semantic_similarity']:.4f}")
    print("-" * 50)
    print("LATENCY & SPEED:")
    print(f"  Avg Retrieval Latency       : {final_evaluation['averages']['latency']['avg_retrieval_seconds']:.3f} seconds")
    print(f"  Avg Generation Latency      : {final_evaluation['averages']['latency']['avg_generation_seconds']:.3f} seconds")
    print(f"  Avg Total Latency           : {final_evaluation['averages']['latency']['avg_total_seconds']:.3f} seconds")
    print(f"  P95 Total Latency           : {final_evaluation['averages']['latency']['p95_total_seconds']:.3f} seconds")
    print("="*50)

if __name__ == "__main__":
    main()
