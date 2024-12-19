import os
import re
import csv
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import logging
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================
# Logging Configuration
# ============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler()
    ]
)

# ============================
# Documentation
# ============================
"""
This script is designed for processing, embedding, and evaluating textual data using Sentence-BERT.
It includes functionality for:
1. Loading and embedding documents from a folder.
2. Extracting numerical identifiers from filenames.
3. Retrieving and evaluating relevant documents for queries.
4. Computing precision and recall for information retrieval tasks.
5. Implementing an inverted index with clustering for efficient search and retrieval.

Requirements:
- numpy
- pandas
- sentence-transformers
- scikit-learn

How to Run:
1. Set `folder_path` to the folder containing your text documents.
2. Provide the path to query CSV files in `query_file_path`.
3. Adjust parameters (e.g., `top_k`) as needed for evaluation.
"""

# ============================
# Functions
# ============================
def load_documents_from_folder(folder_path):
    """
    Loads all .txt documents from the specified folder.

    Args:
        folder_path (str): Path to the folder containing text files.

    Returns:
        list: List of document contents.
        list: List of corresponding filenames.
    """
    logging.info(f"Loading documents from folder: {folder_path}")
    documents = []
    file_names = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
                file_names.append(file_name)
    logging.info(f"Loaded {len(documents)} documents.")
    return documents, file_names

def extract_numbers_from_filenames(file_names):
    """
    Extracts numerical identifiers from filenames.

    Args:
        file_names (list): List of filenames.

    Returns:
        list: List of extracted numbers (or None if no number found).
    """
    logging.info("Extracting numbers from filenames.")
    file_numbers = []
    for filename in file_names:
        match = re.search(r'\d+', filename)
        file_numbers.append(int(match.group()) if match else None)
    logging.info("Finished extracting numbers from filenames.")
    return file_numbers

def build_inverted_index(doc_embeddings, n_clusters=10):
    """
    Builds an inverted index using clustering.

    Args:
        doc_embeddings (numpy.ndarray): Document embeddings.
        n_clusters (int): Number of clusters to create.

    Returns:
        dict: Mapping of cluster index to document indices.
        numpy.ndarray: Centroids of the clusters.
    """
    logging.info(f"Building inverted index with {n_clusters} clusters.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(doc_embeddings)
    centroids = kmeans.cluster_centers_

    inverted_index = {}
    for idx, label in enumerate(cluster_labels):
        if label not in inverted_index:
            inverted_index[label] = []
        inverted_index[label].append(idx)

    logging.info("Inverted index built successfully.")
    return inverted_index, centroids

def retrieve_from_inverted_index(query_embedding, centroids, inverted_index, doc_embeddings, file_numbers, top_k=10):
    """
    Retrieves documents using the inverted index.

    Args:
        query_embedding (numpy.ndarray): Query embedding vector.
        centroids (numpy.ndarray): Cluster centroids.
        inverted_index (dict): Inverted index mapping clusters to document indices.
        doc_embeddings (numpy.ndarray): Document embeddings.
        file_numbers (list): List of document identifiers.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: Filenames of the top-k documents.
    """
    logging.info("Retrieving documents from the inverted index.")
    similarities = cosine_similarity(query_embedding.reshape(1, -1), centroids)
    top_clusters = np.argsort(similarities[0])[::-1][:top_k]

    candidate_docs = []
    for cluster in top_clusters:
        candidate_docs.extend(inverted_index.get(cluster, []))

    candidate_embeddings = doc_embeddings[candidate_docs]
    similarities = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings)
    top_indices = np.argsort(similarities[0])[::-1][:top_k]

    logging.info("Retrieved documents successfully.")
    return [file_numbers[candidate_docs[idx]] for idx in top_indices]

def save_or_load_embeddings(file_path, texts, model, batch_size, device):
    """
    Saves or loads embeddings to/from a file.

    Args:
        file_path (str): Path to save or load embeddings.
        texts (list): List of texts to encode.
        model (SentenceTransformer): Embedding model.
        batch_size (int): Batch size for encoding.
        device (torch.device): Device to run embeddings on.

    Returns:
        numpy.ndarray: Normalized embeddings for the texts.
    """
    if os.path.exists(file_path):
        logging.info(f"Loading embeddings from {file_path}.")
        embeddings = torch.load(file_path, weights_only=False)
        if isinstance(embeddings, np.ndarray):
            logging.info("Embeddings are already numpy arrays.")
        else:
            logging.info("Converting PyTorch tensor to numpy array.")
            embeddings = embeddings.cpu().numpy()
    else:
        logging.info(f"Computing and saving embeddings to {file_path}.")
        embeddings = model.encode(texts, batch_size=batch_size, convert_to_tensor=True, device=device).cpu().numpy()
        torch.save(embeddings, file_path)

    # Normalize embeddings
    embeddings = normalize(embeddings, norm='l2', axis=1)
    return embeddings

def save_retrieved_results_to_csv(retrieved_dic, output_file):
    """
    Saves the retrieved documents dictionary to a CSV file.

    Args:
        retrieved_dic (dict): Dictionary where keys are Query IDs and values are lists of Document Numbers.
        output_file (str): Path to the output CSV file.
    """
    logging.info(f"Saving retrieved results to {output_file}")
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Query_number", "doc_number"])

        # Write data
        for query_id, doc_numbers in retrieved_dic.items():
            for doc_number in doc_numbers:
                writer.writerow([query_id, doc_number])

    logging.info("Retrieved results saved successfully.")

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    logging.info("Starting the document retrieval system.")

    folder_path = "full_docs"
    query_file_path = './query_test/queries.csv'
    output_file = "result.csv"

    # Load documents
    documents, file_names = load_documents_from_folder(folder_path)

    file_numbers = extract_numbers_from_filenames(file_names)

    # Load queries
    logging.info("Loading queries.")
    queries_df = pd.read_csv(query_file_path, sep='\t')
    queries = queries_df['Query'].tolist()
    query_ids = queries_df['Query number'].tolist()

    # Load relevant documents
    logging.info("Loading relevant documents.")

    # Initialize model
    logging.info("Initializing embedding model.")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # Compute or load document embeddings
    doc_embeddings = save_or_load_embeddings("doc_embeddings.pt", documents, model, batch_size=512, device=device)

    # Compute or load query embeddings
    query_embeddings = save_or_load_embeddings("query_embeddings.pt", queries, model, batch_size=512, device=device)

    # Build inverted index
    inverted_index, centroids = build_inverted_index(doc_embeddings, n_clusters=10)

    # Evaluate system using inverted index
    logging.info("Evaluating system using inverted index")
    retrieved_dic = {}

    # Step 1: Retrieve once with k=10
    k_max = 10
    for query_id, query_embedding in zip(query_ids, query_embeddings):
        retrieved_docs = retrieve_from_inverted_index(
            query_embedding, centroids, inverted_index, doc_embeddings, file_numbers, top_k=k_max
        )
        retrieved_dic[query_id] = retrieved_docs

    # Call the function to save the retrieved results
    save_retrieved_results_to_csv(retrieved_dic, output_file)
