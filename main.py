from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def get_obsidian_vault_path() -> Path:
    path = Path.home() / "Documents" / "Obsidian"
    if not path.exists():
        raise FileNotFoundError("Obsidian vault not found")
    return path

def get_markdown_files(vault_path: Path) -> list[Path]:
    ignore_dirs = ["00 System", "20 Obsidian", "53.41 LeetCode", "91 Finance", ".trash"]

    markdown_files = []
    
    # Traverse the vault directory
    for path in vault_path.rglob("*.md"):
        # Check if any part of the path is in the ignore list
        if not any(ignore_dir in path.parts for ignore_dir in ignore_dirs):
            markdown_files.append(path)

    return markdown_files

def parse_markdown_file(file: Path) -> str:
    with open(file, "r") as f:
        return f.read()

def create_embeddings(model: SentenceTransformer, files: list[Path]) -> tuple[list[list[float]], list[str]]:
    docs = [parse_markdown_file(file) for file in files]
    return model.encode(docs), docs

def create_index(embeddings: list[list[float]]) -> faiss.Index:
    faiss_index = faiss.IndexFlatL2(len(embeddings[0]))
    faiss_index.add(embeddings)
    return faiss_index

def search_index(index: faiss.Index, query: str, model: SentenceTransformer, top_k: int = 10) -> tuple[list[float], list[int]]:
    """Search the index for the query and return the top_k results."""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=top_k)
    return distances[0], indices[0]

def generate_answer(query: str, model: pipeline, retrieved_docs: list[str]) -> str:
    context = "\n".join(retrieved_docs)
    result = model(question=query, context=context)
    return result["answer"]

if __name__ == "__main__":

    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    vault_path = get_obsidian_vault_path()
    markdown_files = get_markdown_files(vault_path)
    print(f"Found {len(markdown_files)} markdown files")

    embeddings, docs = create_embeddings(model, markdown_files)
    assert len(docs) == len(markdown_files), "Number of documents does not match"
    assert len(embeddings) == len(docs), "Number of embeddings does not match"
    print(f"Created {len(embeddings)} embeddings")

    faiss_index = create_index(embeddings)
    print(f"Created index")

    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        distances, indices = search_index(faiss_index, query, model, top_k=10)
        retrieved_docs = [docs[i] for i in indices]
        answer = generate_answer(query, qa_model, retrieved_docs)
        print(answer)
