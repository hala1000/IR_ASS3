import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
    documents = []
    file_names = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
                file_names.append(file_name)
    return documents, file_names

def extract_numbers_from_filenames(file_names):
    """
    Extracts numerical identifiers from filenames.

    Args:
        file_names (list): List of filenames.

    Returns:
        list: List of extracted numbers (or None if no number found).
    """
    file_numbers = []
    for filename in file_names:
        match = re.search(r'\d+', filename)
        file_numbers.append(int(match.group()) if match else None)
    return file_numbers

def retrieve_documents_with_filenames(query_embedding, doc_embeddings, file_numbers, top_k=10):
    """
    Retrieves the top-k document filenames based on cosine similarity to the query embedding.

    Args:
        query_embedding (numpy.ndarray): Query embedding vector.
        doc_embeddings (numpy.ndarray): Document embeddings matrix.
        file_numbers (list): List of document identifiers.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: Indices of top-k documents.
        list: Filenames of top-k documents.
    """
    query_embedding_np = query_embedding.cpu().numpy()
    doc_embeddings_np = doc_embeddings.cpu().numpy()

    similarities = cosine_similarity(query_embedding_np.reshape(1, -1), doc_embeddings_np)
    top_indices = np.argsort(similarities[0])[::-1][:top_k]

    top_filenames = [file_numbers[idx] for idx in top_indices]
    return top_indices, top_filenames

def evaluate_system(relevant_docs, query_embeddings, query_ids, doc_embeddings, file_numbers, top_k=10):
    """
    Evaluates the retrieval system using precision and recall metrics.

    Args:
        relevant_docs (dict): Mapping of query IDs to relevant document numbers.
        query_embeddings (numpy.ndarray): Query embeddings matrix.
        query_ids (list): List of query IDs.
        doc_embeddings (numpy.ndarray): Document embeddings matrix.
        file_numbers (list): List of document identifiers.
        top_k (int): Number of top results to consider for evaluation.

    Returns:
        float: Mean precision.
        float: Mean recall.
    """
    precisions = []
    recalls = []

    for q_idx, query_embedding in zip(query_ids, query_embeddings):
        top_indices, top_filenames = retrieve_documents_with_filenames(query_embedding, doc_embeddings, file_numbers, top_k)

        retrieved_relevant = len(set(top_filenames).intersection(set(relevant_docs.get(q_idx, []))))

        precision = retrieved_relevant / top_k
        recall = retrieved_relevant / len(relevant_docs.get(q_idx, [])) if q_idx in relevant_docs else 0

        precisions.append(precision)
        recalls.append(recall)

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    return mean_precision, mean_recall

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    folder_path = "full_docs_small"
    query_file_path = './query_small/dev_small_queries - dev_small_queries.csv'
    relevant_docs_file = './query_small/dev_query_results_small (1).csv'

    # Load documents
    documents, file_names = load_documents_from_folder(folder_path)
    print(f"Loaded {len(documents)} documents.")

    file_numbers = extract_numbers_from_filenames(file_names)

    # Load queries
    queries_df = pd.read_csv(query_file_path)
    queries = queries_df['Query'].tolist()
    query_ids = queries_df['Query number'].tolist()

    # Load relevant documents
    relevant_docs_df = pd.read_csv(relevant_docs_file)
    relevant_docs = relevant_docs_df.groupby('Query_number')['doc_number'].apply(list).to_dict()

    # Initialize model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # Compute embeddings
    doc_embeddings = model.encode(documents, convert_to_tensor=True)
    query_embeddings = model.encode(queries, convert_to_tensor=True)

    # Evaluate system
    for k in [10, 5, 3, 1]:
        mean_precision, mean_recall = evaluate_system(relevant_docs, query_embeddings, query_ids, doc_embeddings, file_numbers, top_k=k)
        print(f"Mean Precision@{k}: {mean_precision:.4f}")
        print(f"Mean Recall@{k}: {mean_recall:.4f}")
