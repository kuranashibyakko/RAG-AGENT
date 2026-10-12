import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq

# 1. Environment & Client Setup
load_dotenv()
groq_client = Groq()

# Distance threshold: distance <= 0.6 corresponds to similarity >= 0.4
DISTANCE_THRESHOLD = 0.6
TOP_K = 2

# 2. Text Chunking Helper
def chunk_text(text: str, chunk_size: int = 50, chunk_overlap: int = 10) -> list[str]:
    words = text.split()
    chunks = []
    stride = chunk_size - chunk_overlap

    for i in range(0, len(words), stride):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))

    return chunks

# 3. Knowledge Base & Query
sample_text = """
The refund policy allows customers to request a full refund within 30 days of purchase.
To qualify, the item must be unused, in its original packaging, and accompanied by a valid receipt.
Digital downloads and gift cards are strictly non-refundable under all circumstances.
Refunds take approximately 5 to 10 business days to reflect on the original payment method once approved.
"""

user_query = "What items are not eligible for a refund?"

# 4. Initialize ChromaDB (In-Memory) & Embedding Function
chroma_client = chromadb.Client()

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create an ephemeral collection configured for cosine distance
collection = chroma_client.create_collection(
    name="refund_policy",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# 5. Populate Collection
chunks = chunk_text(sample_text, chunk_size=30, chunk_overlap=8)
chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

# ChromaDB automatically converts these documents to embeddings upon insertion
collection.add(
    documents=chunks,
    ids=chunk_ids
)

# 6. Automatic Nearest-Neighbor Query
results = collection.query(
    query_texts=[user_query],
    n_results=TOP_K
)

retrieved_docs = results["documents"][0]
retrieved_distances = results["distances"][0]

print("--- ChromaDB Retrieval Results ---")
for doc, dist in zip(retrieved_docs, retrieved_distances):
    print(f"Distance: {dist:.4f} (Approx Sim: {1 - dist:.4f}) -> {doc}")
print("----------------------------------\n")

# Filter by threshold (lower distance = closer match)
qualifying_chunks = [
    doc for doc, dist in zip(retrieved_docs, retrieved_distances)
    if dist <= DISTANCE_THRESHOLD
]

# 7. Generation via Groq
if qualifying_chunks:
    combined_context = "\n---\n".join(qualifying_chunks)

    prompt = f"""You are an assistant who answers questions based strictly on the provided context.
If the answer cannot be found in the context, say "I don't know".

Context:
{combined_context}

Question:
{user_query}

Answer:"""

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
    )
    response = chat_completion.choices[0].message.content
else:
    response = "I couldn't find relevant information in the documents to answer that."

print("LLM Response:")
print(response)