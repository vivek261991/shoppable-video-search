import os
import json
import argparse
from pathlib import Path
import tiktoken
import yaml
from dotenv import load_dotenv

from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from pymilvus import connections

load_dotenv()

MAX_TOKENS = 100
MAX_DURATION = 25.0  # in seconds

tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

def load_api_key():
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    key = config.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("error config.yaml.")
    return key

def count_tokens(text):
    return len(tokenizer.encode(text))

def merge_segments(segments):
    chunks = []
    current = {
        "start": None,
        "end": None,
        "text": "",
        "tokens": 0
    }

    for seg in segments:
        seg_tokens = count_tokens(seg["text"])
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"]

        if current["start"] is None:
            current["start"] = seg_start

        chunk_duration = seg_end - current["start"]
        if current["tokens"] + seg_tokens > MAX_TOKENS or chunk_duration > MAX_DURATION:
            chunks.append({
                "start": current["start"],
                "end": current["end"],
                "text": current["text"].strip()
            })
            current = {
                "start": seg_start,
                "end": seg_end,
                "text": seg_text + " ",
                "tokens": seg_tokens
            }
        else:
            current["end"] = seg_end
            current["text"] += seg_text + " "
            current["tokens"] += seg_tokens

    if current["text"].strip():
        chunks.append({
            "start": current["start"],
            "end": current["end"],
            "text": current["text"].strip()
        })

    return chunks

def embed_chunks(chunks, model="text-embedding-3-small"):
    from openai import OpenAI
    client = OpenAI(api_key=load_api_key())
    embeddings = []
    for chunk in chunks:
        print(f"Embedding: [{chunk['start']:.2f}s - {chunk['end']:.2f}s]...")
        response = client.embeddings.create(
            input=chunk["text"],
            model=model
        )
        embedding = response.data[0].embedding
        embeddings.append({
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"],
            "embedding": embedding
        })
    return embeddings

def save_to_milvus(embeddings, collection_name="video_chunks"):
    print("🧠 Saving embeddings to Milvus...")

    # Connect to Milvus
    connections.connect(host="host.docker.internal", port="19530")

    texts = [e["text"] for e in embeddings]
    metadatas = [{"start": e["start"], "end": e["end"]} for e in embeddings]

    documents = [Document(page_content=text, metadata=meta)
                 for text, meta in zip(texts, metadatas)]


    api_key = load_api_key()
    vector_store = Milvus.from_documents(
        documents,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key),
        connection_args={"host": "host.docker.internal", "port": "19530"},
        collection_name=collection_name
    )

    print("Embeddings saved to Milvus.")

def enrich(video_name, model="text-embedding-3-small"):
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    input_path = data_dir / f"{video_name}.json"
    output_path = data_dir / f"{video_name}_embeddings.json"

    with open(input_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    print(f"Merging segments from {input_path.name}...")
    chunks = merge_segments(segments)

    print(f"Generating embeddings using {model}...")
    embeddings = embed_chunks(chunks, model=model)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2)

    print(f"Saved embeddings locally to {output_path}")

    save_to_milvus(embeddings)
    return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Name of the input transcript file (e.g., clip.mp4)")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model to use")
    args = parser.parse_args()
    enrich(args.video, model=args.model)
