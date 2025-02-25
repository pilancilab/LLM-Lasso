
"""
Simple demo for a mini RAG model based on George Washington's wiki description using OpenAI API queries. 
"""
import os
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
import sys
sys.path.insert(0, ".")
import constants

import warnings
warnings.filterwarnings("ignore")

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API

PERSIST = False

loader = TextLoader('playground/prez.txt')  # loading data from example txt file
embeddings = OpenAIEmbeddings()
index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])

# Create a conversational retrieval chain: this is hybrid in itself
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 3}), # retrieve top 3 most relevant documents
)

# Initialize chat history
chat_history = []

# Simple chat loop
print("This is a simple demo for a mini RAG model based on George Washington's wiki description using OpenAI API queries. ")
print("Type 'quit', 'q', or 'exit' to end the chat.")
while True:
    query = input("Prompt: ")
    if query.lower() in ['quit', 'q', 'exit']:
        print("Goodbye!")
        break

    # Get the response from the conversational chain
    result = chain({"question": query, "chat_history": chat_history})  # save chat history into memory
    print("Answer:", result['answer'])

    # Update the chat history for context in future queries
    chat_history.append((query, result['answer']))