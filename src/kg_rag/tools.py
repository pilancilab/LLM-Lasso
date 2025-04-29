"""
Extra tools for KG-RAG.
"""

from neo4j import GraphDatabase
from tqdm import tqdm
import networkx as nx
import pickle

def bulk_load_networkx_to_neo4j(gpickle_path, uri, user, password):
    """
    Load a NetworkX MultiGraph into Neo4j.

    Args:
        gpickle_path (str): Path to saved NetworkX .gpickle file.
        uri (str): Neo4j bolt URI, e.g., 'bolt://localhost:7687'
        user (str): Username, usually 'neo4j'
        password (str): Your Neo4j password
    """
    # Load NetworkX graph
    print("Loading NetworkX graph...")
    with open(gpickle_path, 'rb') as f:
        G = pickle.load(f)

    print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")

    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # Optional: Wipe database first (be careful)
        print("Cleaning Neo4j database...")
        session.run("MATCH (n) DETACH DELETE n")

        # Insert nodes
        print("Inserting nodes...")
        for node_id, node_data in tqdm(G.nodes(data=True), desc="Nodes"):
            session.run(
                """
                MERGE (n:Entity {id: $id})
                SET n += $properties
                """,
                id=node_id,
                properties=node_data
            )

        # Insert edges
        print("Inserting edges...")
        for node1, node2, edge_data in tqdm(G.edges(data=True), desc="Edges"):
            session.run(
                """
                MATCH (a:Entity {id: $node1})
                MATCH (b:Entity {id: $node2})
                MERGE (a)-[r:RELATED_TO]->(b)
                SET r += $properties
                """,
                node1=node1,
                node2=node2,
                properties=edge_data
            )

    driver.close()
    print("Finished bulk loading into Neo4j!")