import os
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Environment & Client Setup
load_dotenv()
client = Groq()
model = SentenceTransformer("all-MiniLM-L6-v2")

SIMILARITY_THRESHOLD = 0.4
TOP_K = 2  # Retrieve the top 2 most relevant chunks

# 2. Core Helper Functions
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def chunk_text(text: str, chunk_size: int = 50, chunk_overlap: int = 10) -> list[str]:
    words = text.split()
    chunks = []
    stride = chunk_size - chunk_overlap

    for i in range(0, len(words), stride):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))

    return chunks

# 3. Sample Knowledge Base & User Query
sample_text = """
The refund policy allows customers to request a full refund within 30 days of purchase.
To qualify, the item must be unused, in its original packaging, and accompanied by a valid receipt.
Digital downloads and gift cards are strictly non-refundable under all circumstances.
Refunds take approximately 5 to 10 business days to reflect on the original payment method once approved.
"""

user_query = "What items are not eligible for a refund?"

# 4. Chunking & Embedding
chunks = chunk_text(sample_text, chunk_size=30, chunk_overlap=8)
chunk_embeddings = model.encode(chunks)
query_embedding = model.encode(user_query)

# 5. Top-K Retrieval
scores = [cosine_similarity(query_embedding, chunk_vec) for chunk_vec in chunk_embeddings]

# Get indices of the top-k highest similarity scores
top_indices = np.argsort(scores)[-TOP_K:][::-1]

print("--- Retrieval Results ---")
for idx in top_indices:
    print(f"Chunk {idx} (Score: {scores[idx]:.4f}): {chunks[idx]}")
print("-------------------------\n")

# Filter chunks that meet the minimum threshold
qualifying_chunks = [chunks[i] for i in top_indices if scores[i] >= SIMILARITY_THRESHOLD]

# 6. Guardrail Check & Generation
if qualifying_chunks:
    combined_context = "\n---\n".join(qualifying_chunks)

    prompt = f"""You are an assistant who answers questions based strictly on the provided context.
If the answer cannot be found in the context, say "I don't know".

Context:
{combined_context}

Question:
{user_query}

Answer:"""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
    )
    response = chat_completion.choices[0].message.content
else:
    response = "I couldn't find relevant information in the documents to answer that."

print("LLM Response:")
print(response)