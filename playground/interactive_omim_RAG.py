"""
Populate a Chroma-based vector store with chunked JSON data from OMIM database or loading existing persisted database.
Then, use a hybrid chain to answer user queries by retrieving relevant documents from the vector store and combining them with the LLM's general knowledge.
"""

import os
import warnings
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document, HumanMessage, SystemMessage
import json
import sys
sys.path.insert(0, ".")
import constants

warnings.filterwarnings("ignore")  # Suppress warnings
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API

# Enable persistence to save the database to disk
PERSIST = True

# File paths
data_path = constants.OMIM_DATA_FILE
persist_directory_base = constants.OMIM_PERSIST_DIRECTORY
persist_directory = persist_directory_base  # Directory for the vector store

# Step 1: Load chunked data from both sources
print("Loading chunked JSON data from both sources...")
documents = []

# Load scraped OMIM data
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        documents.append(entry)

print(f"Loaded {len(documents)} total chunks from omim database.")

# Step 2: Initialize embeddings
embeddings = OpenAIEmbeddings()

# Step 3: Create or load the OMIM-based vector store
if PERSIST and os.path.exists(persist_directory):
    print("Reusing existing database...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    print("Creating a new database...")
    # Wrap each entry into a Document object
    documents_wrapped = [
        Document(page_content=doc['content'], metadata=doc['metadata']) for doc in documents
    ]
    vectorstore = Chroma.from_documents(
        documents=documents_wrapped,  # Use the wrapped documents
        embedding=embeddings,
        persist_directory=persist_directory
    )
    if PERSIST:
        vectorstore.persist()  # Save the combined database to disk

# Step 4: Initialize retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 documents for context

# Step 5: Initialize LLM
llm = ChatOpenAI(model="gpt-4o")  # GPT model

# Step 6: Define hybrid chain with fallback on pretrained knowledge
def hybrid_chain(query, retriever, llm, chat_history, max_length=4000):
    """
    Hybrid chain combining RAG with fallback to pretrained knowledge.

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
            SystemMessage(content="You are an expert assistant in cancer genomics and bioinformatics."),
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
            SystemMessage(content="You are an expert assistant in cancer genomics and bioinformatics."),
            HumanMessage(content=f"Answer the following question based on your general knowledge:\n\nQuestion: {query}")
        ]
        response = llm(messages)  # Pass structured messages
        final_response = f"General Knowledge Answer:\n{response.content}"

    return final_response

# Step 7: Chat loop
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
