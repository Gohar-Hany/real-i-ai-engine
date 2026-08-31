import os
import json
import sys
import re
from pymongo import MongoClient
from dotenv import load_dotenv

# Ensure the src directory is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory

def parse_llm_response(text: str) -> dict:
    text = text.strip()
    
    # Try 1: Find JSON block in markdown
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            # Normalize keys
            question = data.get("question")
            gt_answer = data.get("ground_truth_answer") or data.get("answer") or data.get("ground-truth") or data.get("ground_truth")
            
            if question and gt_answer:
                return {
                    "question": question.strip(),
                    "ground_truth_answer": gt_answer.strip()
                }
        except Exception:
            pass

    # Try 2: Parse using Regex for Question & Answer headers
    # Look for patterns like **Question:** ... **Answer:** ...
    q_match = re.search(r"(?:\*\*Question:\*\*|Question:|\d+\.\s+\*\*Question:\*\*)\s*(.*?)(?=\n\s*(?:\*\*Answer:\*\*|Answer:|\d+\.\s+\*\*Answer:\*\*)|$)", text, re.IGNORECASE | re.DOTALL)
    a_match = re.search(r"(?:\*\*Answer:\*\*|Answer:|\d+\.\s+\*\*Answer:\*\*)\s*(.*?)$", text, re.IGNORECASE | re.DOTALL)
    
    if q_match and a_match:
        q_text = q_match.group(1).strip(" *:\n\t")
        a_text = a_match.group(1).strip(" *:\n\t")
        if q_text and a_text:
            return {
                "question": q_text,
                "ground_truth_answer": a_text
            }
            
    # Try 3: Basic split if headers aren't matchable but exist
    parts = re.split(r"(?:\*\*Question:\*\*|Question:|\*\*Answer:\*\*|Answer:)", text, flags=re.IGNORECASE)
    if len(parts) >= 3:
        return {
            "question": parts[1].strip(" *:\n\t"),
            "ground_truth_answer": parts[2].strip(" *:\n\t")
        }

    raise ValueError("Could not parse Question and Answer from LLM response")

def main():
    print("Initializing Settings...")
    load_dotenv()
    settings = get_settings()

    # 1. Connect to MongoDB
    print(f"Connecting to MongoDB database: {settings.MONGODB_DATABASE}...")
    client = MongoClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DATABASE]

    # Find the target test project
    project = db["projects"].find_one({"project_id": "testproject1"})
    if not project:
        print("Error: 'testproject1' project not found in MongoDB. Please run seed_db.py or ensure the project exists.")
        sys.exit(1)

    project_oid = project["_id"]
    print(f"Found testproject1 with Object ID: {project_oid}")

    # Retrieve chunks for this project
    chunks_col = db["chunks"]
    # We focus on the Math Session PDF since it has actual course content
    query = {
        "chunk_project_id": project_oid,
        "chunk_metadata.source": "lha1b6s0pc30_Math_Session_1.pdf"
    }
    chunks = list(chunks_col.find(query).sort("chunk_metadata.chunk_index", 1))
    print(f"Retrieved {len(chunks)} chunks from lha1b6s0pc30_Math_Session_1.pdf.")

    if len(chunks) < 15:
        print(f"Warning: Only found {len(chunks)} chunks, selecting all of them instead of 15.")
        selected_chunks = chunks
    else:
        # Select 15 evenly-spaced chunks to cover the whole document
        indices = [int(i * (len(chunks) - 1) / 14) for i in range(15)]
        selected_chunks = [chunks[i] for i in indices]

    print(f"Selected {len(selected_chunks)} chunks for Q&A generation.")

    # 2. Initialize LLM Client
    print(f"Initializing LLM generation client: {settings.GENERATION_MODEL_ID}...")
    llm_factory = LLMProviderFactory(settings)
    llm_client = llm_factory.create(provider=settings.GENERATION_BACKEND)
    llm_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    test_dataset = []

    for idx, chunk in enumerate(selected_chunks):
        chunk_text = chunk["chunk_text"]
        chunk_id = str(chunk["_id"])
        source_doc = chunk.get("chunk_metadata", {}).get("source", "unknown")
        
        print(f"[{idx+1}/{len(selected_chunks)}] Generating Q&A for chunk ID: {chunk_id[:8]}... (length: {len(chunk_text)})")

        prompt = f"""You are a professional teacher designing high-quality exam questions based on a textbook.
Based ONLY on the following textbook chunk, generate:
1. A clear, direct, and specific question (in English) that can be answered completely and solely using the text. Avoid vague questions like "What is discussed in the text?".
2. A precise, accurate, and complete ground-truth answer based ONLY on the text.

Textbook Chunk:
\"\"\"{chunk_text}\"\"\"

Your output MUST be a JSON object with the exact keys "question" and "ground_truth_answer". Do not write any markdown code block wrappers (like ```json) or explanation outside the JSON.
Example:
{{"question": "What is the difference between nominal and ordinal variables?", "ground_truth_answer": "Nominal variables have no inherent order among categories, whereas ordinal variables have a clear ordered scale."}}
"""
        try:
            # Generate Q&A
            raw_response = llm_client.generate_text(prompt=prompt, chat_history=[], max_output_tokens=300)
            parsed_data = parse_llm_response(raw_response)

            test_dataset.append({
                "id": idx + 1,
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "question": parsed_data["question"],
                "ground_truth_answer": parsed_data["ground_truth_answer"],
                "source_document": source_doc
            })
            print(f"  Q: {parsed_data['question'][:60]}...")
            print(f"  A: {parsed_data['ground_truth_answer'][:60]}...")
        except Exception as e:
            print(f"  Error generating Q&A for chunk: {e}")
            print("  Raw Response was:")
            print(raw_response)
            continue

    # Create evaluation directory if not exists
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    
    # Save the dataset
    output_path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, indent=4, ensure_ascii=False)

    print(f"\nSuccessfully generated and saved {len(test_dataset)} Q&A pairs to {output_path} ✓")

if __name__ == "__main__":
    main()
