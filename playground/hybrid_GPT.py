import os
import warnings
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage

warnings.filterwarnings("ignore")  # Suppress warnings

os.environ["OPENAI_API_KEY"] = "YOUR API KEY HERE"

# File paths
persist_directory = "DATABASE DIRECTORY"  # Path to combined vector store

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Step 1: Load vector store
if os.path.exists(persist_directory):
    print("Reusing existing combined database...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    raise FileNotFoundError(f"Vector store not found at {persist_directory}. Ensure the data is preprocessed and saved.")

# Step 2: Initialize retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 documents for context

# Step 3: Initialize LLM
llm = ChatOpenAI(model="gpt-4o")  # GPT model

# Hybrid chain with fallback
def hybrid_chain(query, retriever, llm, chat_history, max_length=4000):
    """
    Hybrid chain combining RAG with fallback to GPT general knowledge.

    Parameters:
    - query: User query.
    - retriever: Retriever object for vector database.
    - llm: GPT model.
    - chat_history: List of previous interactions.
    - max_length: Maximum character length for retrieved context.

    Returns:
    - Answer string (text content only).
    """
    # Step 1: Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(query)

    if retrieved_docs:
        # Combine retrieved documents into context
        context = "\n".join([f"{doc.metadata['gene_name']}: {doc.page_content}" for doc in retrieved_docs])
        context = context[:max_length]  # Ensure the context is within LLM limits

        # print(context)

        # Create a prompt with retrieved context
        messages = [
            SystemMessage(content="You are an expert assistant with access to gene and cancer knowledge."),
            HumanMessage(content=f"Using the following context, provide the most accurate and relevant answer to the question. "
"Prioritize the provided context, but if the context does not contain enough information to fully address the question, "
"use your best general knowledge to complete the answer:\n\n"
            f"{context}\n\n"
            f"Question: {query}")
        ]
        response = llm(messages)  # Pass structured messages
        final_response = f"Document-Grounded Answer:\n{response.content}"
    else:
        # Fallback to GPT's general knowledge
        messages = [
            SystemMessage(content="You are an expert in cancer genomics and bioinformatics."),
            HumanMessage(content=f"Answer the following question based on your general knowledge:\n\nQuestion: {query}")
        ]
        response = llm(messages)  # Pass structured messages
        final_response = f"General Knowledge Answer:\n{response.content}"

    return final_response

# Step 4: Chat loop
chat_history = []
print("Type 'quit', 'q', or 'exit' to end the chat.")

while True:
    query = input("Prompt: ")
    if query.lower() in ['quit', 'q', 'exit']:
        print("Goodbye!")
        break

    # Get the hybrid response
    answer = hybrid_chain(query, retriever, llm, chat_history)
    print("Answer:", answer)

    # Update chat history
    chat_history.append((query, answer))
