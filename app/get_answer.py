from sentence_transformers import SentenceTransformer, util

from db.db import ChromaWork


model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
chroma_collection = ChromaWork().init_db()


def format_query_results(question, query_embedding, documents, metadatas):
    """Format and print the search results with similarity scores"""

    print(f"Question: {question}\n")

    for i, doc in enumerate(documents):
        doc_embedding = model.encode([doc])
        similarity = util.cos_sim(query_embedding, doc_embedding)[0][0].item()
        source = metadatas[i].get('document', 'Unknown')

        print(f"Result {i + 1} (similarity: {similarity:.3f}):")
        print(f"Document: {source}")
        print(f"Content: {doc[:300]}...")
        print()


def query_knowledge_base(question, n_results=2):
    """Query the knowledge base with natural language"""

    query_embedding = model.encode([question])

    results = chroma_collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    format_query_results(question, query_embedding, documents, metadatas)


query_knowledge_base("How do if-else statements work in Python?")
