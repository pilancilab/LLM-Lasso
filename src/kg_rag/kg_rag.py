"""
Implementation of a simple knowledge graph retriever for RAG (retrieval-augmented generation) tasks.
"""

from langchain_graph_retriever import GraphRetriever
from langchain.schema import Document


class NetworkXGraphStore:
    def __init__(self, graph):
        self.graph = graph

    def search(self, query, top_k=5):
        """Simple search: match official_name substring."""
        matches = []
        for node, data in self.graph.nodes(data=True):
            if query.lower() in data.get("official_name", "").lower():
                matches.append(node)
            if len(matches) >= top_k:
                break
        return matches

    def get(self, node_id):
        """Return node metadata."""
        if node_id not in self.graph:
            return None
        node_data = self.graph.nodes[node_id]
        return {
            "name": node_id,
            "type": node_data.get("type", "Unknown"),
            "subtype": node_data.get("subtype", ""),
            "official_name": node_data.get("official_name", ""),
            "common_name": node_data.get("common_name", ""),
        }
    


if __name__ == "__main__":
    import networkx as nx
    # Create a simple graph
    G = nx.Graph()
    G.add_node("GeneA", type="gene", official_name="Gene A", common_name="Gene A")
    G.add_node("GeneB", type="gene", official_name="Gene B", common_name="Gene B")
    G.add_edge("GeneA", "GeneB")
    graph_store = NetworkXGraphStore(G)

    retriever = GraphRetriever(
        graph_store=graph_store,
        search_depth=1,      # how many hops
        search_type="similarity",  # or "keyword" (you can tweak)
    )

    # Now you can use it like a normal retriever!
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),  # or whatever LLM you are using
        retriever=retriever,
        chain_type="stuff",
    )

    # Run a query
    response = qa_chain.run("Which genes are associated with glioblastoma?")
    print(response)
