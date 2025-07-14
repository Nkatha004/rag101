from fastapi import FastAPI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from db import KnowledgeBase

# app = FastAPI()

embeddingModel = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="finance_101", embedding_function=embeddingModel)
llm = LlamaCpp(
    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=128,
    top_p=1,
    n_ctx=2048,
    n_batch=64,
    verbose=False
)
prompt_template = """
    You are a helpful financial assistant. Use the following extracted information from trusted sources to answer the question.
    If the answer is not contained within the text below, say "I don't know".


    IMPORTANT: This is not financial advice.

    Context:
    {context}

    Question:
    {question}

    Answer:"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

def search_chroma_db(question):
    # embeddingModel = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # db = Chroma(persist_directory="finance_101", embedding_function=embeddingModel)
    # query = embeddingModel.embed_query(question)

    results = db.similarity_search(question, k=3)

    return results

# @app.get("/seed")
def seed_db():
    knowledgeBase = KnowledgeBase()
    knowledgeBase.persist_embeddings_in_chroma()
    return {"message": "Knowledge base seeded successfully."}

# @app.get("/")
def interact():
    print("🤖 Financial Assistant is ready. Type your questions or 'exit' to quit.\n")
    try:

        while True:
            question = input("Enter your query: ")
            if question.lower() in ["exit", "quit", "stop", "bye"]:
                break
            
            # Using the chroma database to search for relevant information directly
            # results = search_chroma_db(question)
            # question = "What is a stock exchange?"
            # results = search_chroma_db(question)
            # if not results:
            #     return {"message": "No relevant information found."}
            # else:
            #     print("Results found:")
            #     for result in results:
            #         print(f"Document: {result.page_content}\nMetadata: {result.metadata}\n")

            # Model
            response = qa_chain.invoke({"query": question})

            print("\n👤 You: ")
            print(question)
            print("\n💬 Answer: ")
            print(response["result"])

            # return response
    except KeyboardInterrupt:
        print("\n👋 Exiting by keyboard interrupt. Bye!")

if __name__ == "__main__":
    interact()