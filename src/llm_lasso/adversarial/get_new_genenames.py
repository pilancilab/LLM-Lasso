import os
import warnings
from tqdm import tqdm
import pickle as pkl
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from Expert_RAG.utils import *
from Expert_RAG.helper import *
from openai import OpenAI
from Expert_RAG.omim_RAG_process import *
from adversarial.utils import replace_top, replace_random


warnings.filterwarnings("ignore")  # Suppress warnings
os.environ["OPENAI_API_KEY"] = "YOUR API KEY HERE"

# Define paths and settings
persist_directory = "DATABASE DIRECTORY"  # Vector store path
data_file = "DATABASE DIRECTORY"  # Path to JSON data file
PERSIST = True  # Enable persistence

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Initialize vector store and embeddings
if os.path.exists(persist_directory):
    print("Reusing existing combined database...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    raise FileNotFoundError(f"Vector store not found at {persist_directory}. Ensure data is preprocessed and saved.")

def get_new_genenames(
    category, genenames, save_dir, vectorstore=vectorstore,
    max_replace=200,  replace_top_genes=True,
):
    if replace_top_genes:
        new_genenames = replace_top(
            genenames, category, vectorstore.as_retriever(search_kwargs={"k": 2000}),
            max_replace, min_doc_count=0
        )
    else:
        new_genenames = replace_random(genenames, max_replace)

    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir + "/new_genenames.pkl", "wb") as f:
        pkl.dump(new_genenames, f)
    with open(save_dir + "/new_genenames.txt", "w") as f:
        f.write(str(new_genenames))