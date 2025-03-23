import os
import json
import argparse
from pathlib import Path
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pymilvus import connections

load_dotenv()

def load_api_key():
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config" / "config.yaml"
    with open(config_path, "r") as f:
        import yaml
        config = yaml.safe_load(f)
    key = config.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY not found in config.yaml.")
    return key

def query_milvus(video_name, query, k=3):
    print("🔍 Querying Milvus...")

    api_key = load_api_key()

    # Connect to Milvus
    connections.connect(host="host.docker.internal", port="19530")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vector_store = Milvus(
        embedding_function=embedding_model,
        collection_name="video_chunks",
        connection_args={"host": "host.docker.internal", "port": "19530"},
    )

    docs = vector_store.similarity_search(query, k=k)
    return docs

def load_prompt_template():
    root_dir = Path(__file__).resolve().parent.parent
    prompt_path = root_dir / "scripts" / "prompt_template.txt"
    with open(prompt_path, "r") as f:
        return PromptTemplate.from_template(f.read())

def generate_answer_with_llm(context_chunks, query, model="gpt-3.5-turbo"):
    api_key = load_api_key()
    prompt = load_prompt_template()

    print(prompt)

    chain = prompt | ChatOpenAI(model=model, temperature=0, openai_api_key=api_key)

    return chain.invoke({
        "context": context_chunks.strip(),
        "question": query
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Video name without extension (e.g., clip.mp4)")
    parser.add_argument("--query", required=True, help="User query to ask about the video")
    args = parser.parse_args()

    docs = query_milvus(args.video, args.query)
    print("\n Top Relevant Chunks:\n")
    for doc in docs:
        print(f"[{doc.metadata.get('start', 0.0):.2f}s - {doc.metadata.get('end', 0.0):.2f}s] {doc.page_content}")

    context = "\n".join([
        f"[{doc.metadata.get('start', 0.0):.2f}s - {doc.metadata.get('end', 0.0):.2f}s] {doc.page_content}"
        for doc in docs
    ])

    answer = generate_answer_with_llm(context, args.query)
    print("\n Answer:")
    print(answer)
