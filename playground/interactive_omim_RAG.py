"""
Chroma-based RAG model for gene-expert GPT on omim.org with chunking.

USE WITHIN OPENAI LIMIT: String with maximum length 1048576
"""

import os
import json
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import Document  # Import the Document class

import warnings

warnings.filterwarnings("ignore")  # Suppress deprecation warnings

os.environ["OPENAI_API_KEY"] = "YOUR KEY HERE"

# Enable persistence to save the database to disk
PERSIST = True

# File paths
data_path = "DATA JSON PATH"
persist_directory_base = "DATABASE PATH"
persist_directory = persist_directory_base  # Directory for the combined vector store

# Step 1: Load chunked data from both sources
print("Loading chunked data from both sources...")
documents = []

# Load gene data
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        documents.append(entry)

# Load cancer data
# with open(cancer_json, "r", encoding="utf-8") as f:
#     for line in f:
#         entry = json.loads(line)
#         documents.append(entry)

print(f"Loaded {len(documents)} total chunks from omim database.")

# Step 2: Initialize embeddings
embeddings = OpenAIEmbeddings()

# Step 3: Create or load the combined vector store
if PERSIST and os.path.exists(persist_directory):
    print("Reusing existing combined database...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    print("Creating a new combined database...")
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

# # Step 4: Set up the retrieval chain with memory
# # memory buffer feature incompatible with conversation retrieval chain (see notes).
# chat = ChatOpenAI(model="gpt-4o")
# memory = ConversationSummaryBufferMemory(llm=chat, max_token_limit=5000)  # Summarization memory
# retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # Retrieve top k most relevant answers
# chain = ConversationalRetrievalChain.from_llm(
#     llm=chat,
#     retriever=retriever,
#     memory=memory,
# )

# # Step 5: Chat loop
# print("Type 'quit', 'q', or 'exit' to end the chat.")
# while True:
#     query = input("Prompt: ")
#     if query.lower() in ['quit', 'q', 'exit']:
#         print("Goodbye!")
#         break
#
#     # Run query through the chain
#     result = chain({"question": query})
#     print("Answer:", result['answer'])


# Step 4: Set up the retrieval chain without memory
chat = ChatOpenAI(model="gpt-4o")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top k most relevant answers

chain = ConversationalRetrievalChain.from_llm(
    llm=chat,
    retriever=retriever
)

# Step 5: Chat loop using a simple chat_history list
chat_history = []  # Initialize chat history as a list of (user, assistant) tuples
print("Type 'quit', 'q', or 'exit' to end the chat.")

while True:
    query = input("Prompt: ")
    if query.lower() in ['quit', 'q', 'exit']:
        print("Goodbye!")
        break

    # Run query through the chain
    result = chain({"question": query, "chat_history": chat_history})
    print("Answer:", result['answer'])

    # Update chat history
    chat_history.append((query, result['answer']))
