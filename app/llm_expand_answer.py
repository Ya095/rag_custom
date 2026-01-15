from chromadb import Collection
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSerializable, Output
from sentence_transformers import SentenceTransformer
from torch import Tensor

from db.db import ChromaWork

import logging

logger = logging.getLogger(__name__)


model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
chroma_collection = ChromaWork().init_db()


############### Расширенная генерация ответов с Open Source LLMs ###############
# Initialize the local LLM
llm = OllamaLLM(model="llama3.2:3b", temperature=0.1)
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a Python programming expert. Based on the provided documentation, answer the question clearly and accurately.

    Documentation:
    {context}

    Question: {question}

    Answer (be specific about syntax, keywords, and provide examples when helpful):""",
)

# call processing
chain: RunnableSerializable = prompt_template | llm


def retrieve_context(
    question: str,
    question_answer_model: SentenceTransformer,
    collection: Collection,
    n_results: int = 5,
) -> tuple[str, list[str]]:
    """Retrieve relevant context using embeddings"""

    query_embeddings: Tensor = question_answer_model.encode(question)
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents: list[str] = results["documents"][0]
    context: str = "\n\n---SECTION---\n\n".join(documents)

    return context, documents


def get_llm_answer(question: str, context: str) -> Output:
    """Generate answer using retrieved context"""

    answer: Output = chain.invoke(
        {
            "context": context[:2000],
            "question": question,
        }
    )

    return answer


def format_response(question: str, answer: str, source_chunks) -> str:
    """Format the final response with sources"""

    response = f"**Question:** {question}\n\n"
    response += f"**Answer:** {answer}\n\n"
    response += "**Sources:**\n"

    for i, chunk in enumerate(source_chunks[:3], 1):
        preview = chunk[:100].replace("\n", " ") + "..."
        response += f"{i}. {preview}\n"

    return response


def stream_llm_answer(question: str, context: str):
    """Stream LLM answer generation token by token"""

    for chunk in chain.stream(
        {
            "context": context[:2000],
            "question": question,
        }
    ):
        yield getattr(chunk, "content", str(chunk))


def enhanced_query_with_llm(
    question: str,
    question_answer_model: SentenceTransformer,
    collection: Collection,
    n_results: int = 5,
) -> str:
    """Query function combining retrieval with LLM generation"""

    context, documents = retrieve_context(
        question=question,
        question_answer_model=question_answer_model,
        collection=collection,
        n_results=n_results,
    )
    answer: str = get_llm_answer(question, context)

    return format_response(question, answer, documents)


if __name__ == "__main__":
    # Test the enhanced query system
    # question = "How do if-else statements work in Python?"
    # enhanced_response = enhanced_query_with_llm(
    #     question=question,
    #     question_answer_model=model,
    #     collection=chroma_collection,
    # )
    # print(enhanced_response)

    import time

    # Test the streaming functionality
    question = "What are Python loops?"
    context, documents = retrieve_context(
        question,
        question_answer_model=model,
        collection=chroma_collection,
        n_results=3,
    )

    print("Question:", question)
    print("Answer: ", end="", flush=True)

    # Stream the answer token by token
    for token in stream_llm_answer(question, context):
        print(token, end="", flush=True)
        time.sleep(0.05)  # Simulate real-time typing effect
