from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2"
)

vector_store = FAISS.load_local(
    "rag/vector_stores/authoritative_faiss",
    embeddings,
    allow_dangerous_deserialization=True,
)

results = vector_store.similarity_search(
    "What is retrieval-augmented generation?",
    k=5,
)

for i, doc in enumerate(results, 1):
    print(f"\nResult {i}")
    print(doc.metadata)
    print(doc.page_content[:300])