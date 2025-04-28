
"""
Loading graph data from a JSON file into a knowledge graph to build a retreival database.
Current script implements a variety of graph data construction functions.
"""
import os
import json
import networkx as nx
from tqdm import tqdm
import igraph as ig
import torch
from arango import ArangoClient
from torch_geometric.data import Data
from constants import ARANGO_DB_USERNAME, ARANGO_DB_PASSWORD, ARANGO_DB_NAME

def load_graph_data_netx(load_dir, save_dir, save_filename='ikraph_graph.gpickle'):
    """
    Load iKraph knowledge graph data into a NetworkX MultiGraph and save it.

    Args:
        load_dir (str): Directory where the iKraph JSON files are located.
        save_dir (str): Directory where the constructed graph will be saved.
        save_filename (str): Filename for the saved graph (default: 'ikraph_graph.gpickle').

    Returns:
        G (networkx.MultiGraph): Constructed NetworkX graph object.
    """
    # Initialize empty undirected MultiGraph
    G = nx.MultiGraph()

    # File paths
    load_dir = os.path.join(load_dir, 'iKraph_full')
    node_file = os.path.join(load_dir, 'NER_ID_dict_cap_final.json')
    pubmed_rel_file = os.path.join(load_dir, 'PubMedList.json')
    db_rel_file = os.path.join(load_dir, 'DBRelations.json')

    print("Loading nodes...")
    # Load and add nodes
    with open(node_file, 'r') as f:
        node_data = json.load(f)

    for node in tqdm(node_data, desc="Adding Nodes"):
        G.add_node(
            node["biokdeid"],
            type=node.get("type", "Unknown"),
            subtype=node.get("subtype", "NA"),
            external_id=node.get("id", None),
            species=node.get("species", None),
            official_name=node.get("official name", ""),
            common_name=node.get("common name", "")
        )

    print(f"Loaded {len(G.nodes)} nodes.")

    print("Loading PubMed abstract relationships...")
    # Load and add edges from PubMed relationships
    with open(pubmed_rel_file, 'r') as f:
        pubmed_relations = json.load(f)

    for rel in tqdm(pubmed_relations, desc="Adding PubMed Edges"):
        rel_key = rel.get("id", "")
        rel_list = rel.get("list", [])
        parts = rel_key.split('.')
        if len(parts) != 6:
            continue  # Skip malformed entries
        node1, node2, rel_id, corr_id, direction, method = parts

        for entry in rel_list:
            if len(entry) != 4:
                continue
            score, doc_id, prob, novelty = entry

            if node1 in G and node2 in G:
                G.add_edge(
                    node1,
                    node2,
                    source='PubMed',
                    relation_id=rel_id,
                    correlation_id=corr_id,
                    direction=direction,
                    method=method,
                    score=float(score),
                    probability=float(prob),
                    novelty=int(novelty),
                    doc_id=doc_id
                )

    print(f"Graph now has {len(G.edges)} edges after adding PubMed relationships.")

    print("Loading Database-inferred relationships...")
    # Load and add edges from database relationships
    with open(db_rel_file, 'r') as f:
        db_relations = json.load(f)

    for rel in tqdm(db_relations, desc="Adding DB Edges"):
        node1 = rel.get("node_one_id")
        node2 = rel.get("node_two_id")
        if node1 and node2 and node1 in G and node2 in G:
            G.add_edge(
                node1,
                node2,
                source='Database',
                relationship_type=rel.get("relationship_type", "Unknown"),
                prob=float(rel.get("prob", 1.0)),
                score=float(rel.get("score", 1.0)),
                db_source=rel.get("source", "Unknown")
            )

    print(f"Graph now has {len(G.edges)} total edges after adding database relationships.")

    # Save graph
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_filename)
    nx.write_gpickle(G, save_path)
    print(f"Graph saved to {save_path}")

    return G

def load_graph_data_igraph(load_dir, save_dir, save_filename='ikraph_graph_igraph.pickle'):
    """
    Load iKraph into an igraph Graph and save it.
    """
    print("Initializing igraph graph...")
    g = ig.Graph(directed=True)

    node_id_mapping = {}  # biokdeid -> igraph integer ID
    node_counter = 0

    # Load nodes
    load_dir = os.path.join(load_dir, 'iKraph_full')
    with open(os.path.join(load_dir, 'NER_ID_dict_cap_final.json')) as f:
        node_data = json.load(f)
    
    print("Adding nodes...")
    for node in tqdm(node_data, desc="Nodes"):
        g.add_vertex(
            name=node["biokdeid"],
            type=node.get("type", "Unknown"),
            official_name=node.get("official name", ""),
            subtype=node.get("subtype", ""),
            species=node.get("species", ""),
            external_id=node.get("id", "")
        )
        node_id_mapping[node["biokdeid"]] = node_counter
        node_counter += 1
    
    # Load edges
    src_list, dst_list = [], []

    # PubMedList edges
    with open(os.path.join(load_dir, 'PubMedList.json')) as f:
        pubmed_relations = json.load(f)

    print("Adding PubMed edges...")
    for rel_key, rel_list in tqdm(pubmed_relations.items(), desc="PubMed edges"):
        parts = rel_key.split('.')
        if len(parts) < 6:
            continue
        node1, node2 = parts[0], parts[1]
        if node1 in node_id_mapping and node2 in node_id_mapping:
            src_list.append(node_id_mapping[node1])
            dst_list.append(node_id_mapping[node2])

    # DBRelations edges
    with open(os.path.join(load_dir, 'DBRelations.json')) as f:
        db_relations = json.load(f)

    print("Adding Database edges...")
    for rel in tqdm(db_relations, desc="DB edges"):
        node1 = rel.get("node_one_id")
        node2 = rel.get("node_two_id")
        if node1 in node_id_mapping and node2 in node_id_mapping:
            src_list.append(node_id_mapping[node1])
            dst_list.append(node_id_mapping[node2])

    g.add_edges(zip(src_list, dst_list))
    print(f"Graph now has {g.vcount()} nodes and {g.ecount()} edges.")

    # Save graph
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_filename)
    g.write_pickle(save_path)

    print(f"Graph saved to {save_path}")

    return g

def load_graph_data_arangodb(load_dir, db_name, username, password, graph_name='ikraph', vertex_col_name='nodes', edge_col_name='edges'):
    """
    Load iKraph data into ArangoDB as a graph database.

    Returns:
        db: ArangoDB database connection object
        graph: ArangoDB graph object
    """
    client = ArangoClient()
    sys_db = client.db('_system', username=username, password=password)

    if not sys_db.has_database(db_name):
        sys_db.create_database(db_name)

    db = client.db(db_name, username=username, password=password)

    if not db.has_collection(vertex_col_name):
        db.create_collection(vertex_col_name)

    if not db.has_collection(edge_col_name):
        db.create_collection(edge_col_name, edge=True)

    if not db.has_graph(graph_name):
        graph = db.create_graph(graph_name)
        graph.create_vertex_collection(vertex_col_name)
        graph.create_edge_definition(
            edge_collection=edge_col_name,
            from_vertex_collections=[vertex_col_name],
            to_vertex_collections=[vertex_col_name],
        )
    else:
        graph = db.graph(graph_name)

    nodes_collection = graph.vertex_collection(vertex_col_name)
    edges_collection = graph.edge_collection(edge_col_name)

    print("Loading nodes...")
    with open(os.path.join(load_dir, 'NER_ID_dict_cap_final.json')) as f:
        node_data = json.load(f)

    node_mapping = {}
    for node in tqdm(node_data, desc="Adding nodes"):
        doc = {
            "_key": node["biokdeid"],
            "type": node.get("type", "Unknown"),
            "subtype": node.get("subtype", "NA"),
            "external_id": node.get("id", ""),
            "species": node.get("species", ""),
            "official_name": node.get("official name", ""),
            "common_name": node.get("common name", "")
        }
        nodes_collection.insert(doc, overwrite=True)
        node_mapping[node["biokdeid"]] = f"{vertex_col_name}/{node['biokdeid']}"

    print(f"Loaded {len(node_mapping)} nodes.")

    print("Loading edges...")
    with open(os.path.join(load_dir, 'PubMedList.json')) as f:
        pubmed_relations = json.load(f)

    for rel_key, rel_list in tqdm(pubmed_relations.items(), desc="Adding PubMed edges"):
        parts = rel_key.split('.')
        if len(parts) < 6:
            continue
        node1, node2 = parts[0], parts[1]
        if node1 in node_mapping and node2 in node_mapping:
            for entry in rel_list:
                score, doc_id, prob, novelty = entry
                edge_doc = {
                    "_from": node_mapping[node1],
                    "_to": node_mapping[node2],
                    "source": "PubMed",
                    "relation_id": parts[2],
                    "correlation_id": parts[3],
                    "direction": parts[4],
                    "method": parts[5],
                    "score": float(score),
                    "probability": float(prob),
                    "novelty": int(novelty),
                    "doc_id": doc_id
                }
                edges_collection.insert(edge_doc)

    with open(os.path.join(load_dir, 'DBRelations.json')) as f:
        db_relations = json.load(f)

    for rel in tqdm(db_relations, desc="Adding DB edges"):
        node1 = rel.get("node_one_id")
        node2 = rel.get("node_two_id")
        if node1 in node_mapping and node2 in node_mapping:
            edge_doc = {
                "_from": node_mapping[node1],
                "_to": node_mapping[node2],
                "source": "Database",
                "relationship_type": rel.get("relationship_type", "Unknown"),
                "probability": float(rel.get("prob", 1.0)),
                "score": float(rel.get("score", 1.0)),
                "db_source": rel.get("source", "")
            }
            edges_collection.insert(edge_doc)

    print(f"Finished loading graph into ArangoDB: {graph_name}")

    return db, graph


def load_graph_data_pyg(load_dir, save_dir, save_filename='ikraph_graph_pyg.pt'):
    """
    Load iKraph into PyTorch Geometric Data object and save it.
    """
    node_id_mapping = {}
    node_counter = 0

    print("Loading nodes...")
    load_dir = os.path.join(load_dir, 'iKraph_full')
    with open(os.path.join(load_dir, 'NER_ID_dict_cap_final.json')) as f:
        node_data = json.load(f)

    for node in tqdm(node_data, desc="Nodes"):
        node_id_mapping[node["biokdeid"]] = node_counter
        node_counter += 1

    print(f"Loaded {len(node_id_mapping)} nodes.")

    src_list, dst_list = [], []

    # Load edges from PubMedList
    with open(os.path.join(load_dir, 'PubMedList.json')) as f:
        pubmed_relations = json.load(f)

    print("Adding PubMed edges...")
    for rel_key, rel_list in tqdm(pubmed_relations.items(), desc="PubMed edges"):
        parts = rel_key.split('.')
        if len(parts) < 6:
            continue
        node1, node2 = parts[0], parts[1]
        if node1 in node_id_mapping and node2 in node_id_mapping:
            src_list.append(node_id_mapping[node1])
            dst_list.append(node_id_mapping[node2])

    # Load edges from DBRelations
    with open(os.path.join(load_dir, 'DBRelations.json')) as f:
        db_relations = json.load(f)

    print("Adding Database edges...")
    for rel in tqdm(db_relations, desc="DB edges"):
        node1 = rel.get("node_one_id")
        node2 = rel.get("node_two_id")
        if node1 in node_id_mapping and node2 in node_id_mapping:
            src_list.append(node_id_mapping[node1])
            dst_list.append(node_id_mapping[node2])

    print(f"Total edges: {len(src_list)}")

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Create a Data object
    data = Data(edge_index=edge_index)

    print(f"Built PyG graph with {data.num_nodes} nodes and {data.num_edges} edges.")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(data, save_path)

    print(f"Graph saved to {save_path}")

    return data


def load_graph_data(load_dir, save_dir, save_filename='ikraph_graph.gpickle', method = 'networkx'):
    """
    Load iKraph data into a graph using the specified method and save it.

    Args:
        method (str): Method to use for loading the graph ('networkx', 'igraph', 'pyg', 'arangodb').
        load_dir (str): Directory where the iKraph JSON files are located.
        save_dir (str): Directory where the constructed graph will be saved.
        save_filename (str): Filename for the saved graph (default: 'ikraph_graph.gpickle').

    Returns:
        G: Constructed graph object.
    """
    if method == 'networkx':
        return load_graph_data_netx(load_dir, save_dir, save_filename)
    elif method == 'igraph':
        return load_graph_data_igraph(load_dir, save_dir, save_filename)
    elif method == 'pyg':
        return load_graph_data_pyg(load_dir, save_dir, save_filename)
    elif method == 'arangodb':
        # Example usage, replace with actual credentials
        username = ARANGO_DB_USERNAME
        password = ARANGO_DB_PASSWORD
        db_name = ARANGO_DB_NAME
        return load_graph_data_arangodb(load_dir, db_name, username, password)
    else:
        raise ValueError("Invalid method. Choose from 'networkx', 'igraph', or 'pyg'.")
    


if __name__ == "__main__":
    load_dir = 'retrieval_database'
    save_dir = 'retrieval_database'
    save_filename = 'ikraph_graph.gpickle'
    load_graph_data(load_dir, save_dir, save_filename = save_filename)