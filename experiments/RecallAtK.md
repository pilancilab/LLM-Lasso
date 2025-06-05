# LLM-Lasso Recall@k experiment

## Setup

To perform recall@k, you need to set up the OMIM RAG database.
For setting up OMIM RAG, refer to **`examples/omim_rag_tutorial.ipynb`**.

As setting up the full OMIM vector database is quite time-intensive, feel free to set it up with a limited number of genes (e.g., for the genes in the Lung cancer dataset).

## Recall@k Experiment
To run the recall@k experiment after setting up the OMIM RAG database, run
```
./shell_scripts/recall_at_k.sh
```