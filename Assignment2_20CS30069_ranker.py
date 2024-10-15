

import pandas as pd
import pickle
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import sys
from scipy.sparse import dok_matrix

# Load inverted index from .bin file using pickle
def load_index_from_file(file_path):
    with open(file_path, 'rb') as file_handle:
        return pickle.load(file_handle)

# Load query file from XML
def load_queries_from_xml(query_file):
    query_dict = {}
    tree = ET.parse(query_file)
    root = tree.getroot()
    for topic in root.findall('topic'):
        query_id = int(topic.get('number'))
        query_content = topic.find('query').text or ""
        query_dict[query_id] = query_content.strip()
    return query_dict

# Clean and process text
def clean_and_process_text(raw_text):
    stop_words_set = set(stopwords.words('english'))
    lemmatizer_tool = WordNetLemmatizer()
    token_list = word_tokenize(raw_text.lower())
    cleaned_tokens = [lemmatizer_tool.lemmatize(token) for token in token_list 
                      if token not in stop_words_set and token not in string.punctuation]
    return ' '.join(cleaned_tokens)

# Load documents from CSV and clean abstracts
def load_documents_from_csv(csv_file):
    # df = pd.read_csv(csv_file)
    df = pd.read_csv(csv_file, encoding='ISO-8859-1', on_bad_lines='skip')
    documents = {}
    for _, row in df.iterrows():
        doc_id = row.get('cord_uid', None)
        abstract = row.get('abstract', '')
        if isinstance(abstract, str):
            cleaned_abstract = clean_and_process_text(abstract)
            documents[doc_id] = cleaned_abstract
        else:
            documents[doc_id] = clean_and_process_text(str(abstract)) if pd.notna(abstract) else ''
    return documents

# Calculate the Document Frequency for each term in the vocabulary
def calculate_document_frequencies(inverted_index):
    df = {}
    for term, postings in inverted_index.items():
        df[term] = len(postings)
    return df

# Calculate the Term Frequency for each term in each document
def calculate_term_frequencies(documents):
    tf = defaultdict(lambda: defaultdict(int))
    for doc_id, content in documents.items():
        for term in content.split():
            tf[doc_id][term] += 1
    return tf

# Calculate the TF-IDF weights for each document
def calculate_tf_idf(tf, df, total_documents):
    tf_idf = defaultdict(lambda: defaultdict(float))
    for doc_id, terms in tf.items():
        for term, term_freq in terms.items():
            if term in df:
                idf = np.log10(total_documents / df[term])
                tf_idf[doc_id][term] = term_freq * idf
    return tf_idf

# Build the Vocabulary (V)
# def build_vocabulary(documents, queries):
#     vocab = set()
#     for content in documents.values():
#         vocab.update(content.split())
#     for content in queries.values():
#         vocab.update(content.split())
#     return sorted(vocab)  # Return sorted vocabulary for consistent indexing

# Build the Vocabulary (V) and Count Term Frequencies

from collections import Counter

def build_vocabulary(documents, queries):
    term_counter = Counter()  # Counter to count term frequencies
    vocab = set()  # Set to hold unique terms
    
    # Count terms in documents
    for content in documents.values():
        terms = content.split()
        vocab.update(terms)
        term_counter.update(terms)
    
    # Count terms in queries
    for content in queries.values():
        terms = content.split()
        vocab.update(terms)
        term_counter.update(terms)
    
    return sorted(vocab), term_counter  # Return sorted vocabulary and term counter

from collections import defaultdict
import numpy as np

# Convert documents and queries into |V|-dimensional vectors
def vectorize(tf_idf, vocab):
    vector_dict = {}  # Change to a regular dict
    vocab_index = {term: idx for idx, term in enumerate(vocab)}  # Map terms to indices

    for doc_id, terms in tf_idf.items():
        vector_dict[doc_id] = {}  # Initialize a new dict for each document
        for term, weight in terms.items():
            if term in vocab_index:
                vector_dict[doc_id][term] = weight  # Directly map term to its weight
    return vector_dict

def apply_lnc_ltc(tf_idf, vocab):
    """ lnc.ltc scheme: logarithmic term frequency, no document length normalization for documents; 
        logarithmic term frequency for queries with cosine normalization """

    # Create a dictionary that maps each term in vocab to its index for faster lookup
    vocab_dict = {term: idx for idx, term in enumerate(vocab)}
    
    lnc_ltc_vectors = {}  # Change to a regular dict

    for doc_id, terms in tf_idf.items():
        lnc_ltc_vectors[doc_id] = {}  # Initialize a new dict for each document
        # Apply lnc to documents: log(1 + tf) with no length normalization
        for term, weight in terms.items():
            if term in vocab_dict:
                lnc_ltc_vectors[doc_id][term] = 1 + np.log10(weight)

        # Normalize for cosine similarity (vector length)
        norm = np.linalg.norm(list(lnc_ltc_vectors[doc_id].values()))
        if norm > 0:
            for term in lnc_ltc_vectors[doc_id]:
                lnc_ltc_vectors[doc_id][term] /= norm

    return lnc_ltc_vectors


def apply_lnc_Ltc(tf_idf, vocab):
    """ lnc.Ltc scheme: logarithmic term frequency with document length normalization """

    vocab_dict = {term: idx for idx, term in enumerate(vocab)}
    lnc_Ltc_vectors = {}  # Change to a regular dict

    for doc_id, terms in tf_idf.items():
        lnc_Ltc_vectors[doc_id] = {}  # Initialize a new dict for each document
        # Apply lnc for queries and log + normalize for documents
        for term, weight in terms.items():
            if term in vocab_dict:
                lnc_Ltc_vectors[doc_id][term] = (1 + np.log10(weight)) * np.log10(weight)

        # Normalize for cosine similarity
        norm = np.linalg.norm(list(lnc_Ltc_vectors[doc_id].values()))
        if norm > 0:
            for term in lnc_Ltc_vectors[doc_id]:
                lnc_Ltc_vectors[doc_id][term] /= norm

    return lnc_Ltc_vectors

def apply_anc_apc(tf_idf, vocab):
    """ anc.apc scheme: augmented frequency with cosine normalization for both queries and documents """
    vocab_dict = {term: idx for idx, term in enumerate(vocab)}
    anc_apc_vectors = {}  # Change to a regular dict

    for doc_id, terms in tf_idf.items():
        anc_apc_vectors[doc_id] = {}  # Initialize a new dict for each document
        # Apply anc to documents and queries
        max_tf = max(terms.values())
        for term, weight in terms.items():
            if term in vocab_dict:
                anc_apc_vectors[doc_id][term] = 0.5 + 0.5 * (weight / max_tf)

        # Normalize for cosine similarity
        norm = np.linalg.norm(list(anc_apc_vectors[doc_id].values()))
        if norm > 0:
            for term in anc_apc_vectors[doc_id]:
                anc_apc_vectors[doc_id][term] /= norm

    return anc_apc_vectors


# Compute cosine similarity between two vectors
def cosine_similarity(vector1, vector2):
    terms1 = set(vector1.keys())
    terms2 = set(vector2.keys())
    common_terms = terms1.intersection(terms2)

    dot_product = sum(vector1[term] * vector2[term] for term in common_terms)
    norm1 = np.linalg.norm(list(vector1.values()))
    norm2 = np.linalg.norm(list(vector2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


# Rank documents for each query based on cosine similarity
def rank_documents(query_vectors, document_vectors):
    ranked_results = defaultdict(list)

    for query_id, query_vector in query_vectors.items():
        similarity_scores = []

        for doc_id, doc_vector in document_vectors.items():
            similarity = cosine_similarity(query_vector, doc_vector)
            similarity_scores.append((doc_id, similarity))

        # Sort documents by descending cosine similarity and get top 50
        ranked_documents = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:50]
        ranked_results[query_id] = [doc_id for doc_id, _ in ranked_documents]

    return ranked_results


# Write results to the file
def write_ranked_results_to_file(ranked_results, file_name):
    with open(file_name, 'w') as f:
        for query_id, doc_ids in ranked_results.items():
            f.write(f"{query_id} : {' '.join(doc_ids)}\n")


# Main processing
def main(inverted_index_file, query_file, csv_file):
    inverted_index = load_index_from_file(inverted_index_file)
    queries = load_queries_from_xml(query_file)
    cleaned_queries = {qid: clean_and_process_text(content) for qid, content in queries.items()}
    documents = load_documents_from_csv(csv_file)

    df = calculate_document_frequencies(inverted_index)
    tf = calculate_term_frequencies(documents)
    total_documents = len(documents)
    tf_idf = calculate_tf_idf(tf, df, total_documents)

    # Build vocabulary from documents and queries
    vocab, term_counter = build_vocabulary(documents, cleaned_queries)

    # Remove the last 300 terms from the vocabulary if there are enough terms
    # num_terms_to_remove = 300
    # if len(vocab) > num_terms_to_remove:
    #     terms_to_remove = vocab[-num_terms_to_remove:]  # Get the last 300 terms
    #     filtered_vocab = vocab[:-num_terms_to_remove]  # Filter the vocabulary
    #     # Remove terms from the frequency counter
    #     for term in terms_to_remove:
    #         del term_counter[term]  # Remove from frequency counter
    # else:
    #     filtered_vocab = vocab

    
    filtered_vocab = vocab
    # Count terms with frequencies of 1 and 2
    count_freq_1 = sum(1 for freq in term_counter.values() if freq == 1)
    count_freq_2 = sum(1 for freq in term_counter.values() if freq == 2)

    # Print the counts
    # print(f"Number of terms with frequency 1: {count_freq_1}")
    # print(f"Number of terms with frequency 2: {count_freq_2}")
    
    print(f"Vocabulary length: {len(filtered_vocab)}")

    # Convert documents and queries into vectors
    document_vectors = vectorize(tf_idf, filtered_vocab)
    
    query_tf = calculate_term_frequencies(cleaned_queries)
    query_tf_idf = calculate_tf_idf(query_tf, df, total_documents)
    query_vectors = vectorize(query_tf_idf, filtered_vocab)
    

    # Apply different weighting schemes
    lnc_ltc_doc_vectors = apply_lnc_ltc(tf_idf, filtered_vocab)

    # Vectorize queries using the same vocab and schemes
    lnc_ltc_query_vectors = apply_lnc_ltc(query_tf_idf, filtered_vocab)

    # Rank documents for each query for each scheme
    lnc_ltc_ranks = rank_documents(lnc_ltc_query_vectors, lnc_ltc_doc_vectors)

    # Write ranked results to files
    write_ranked_results_to_file(lnc_ltc_ranks, 'Assignment2_20CS30069_ranked_list_A.txt')
    print("Written to A.txt")
    
    # Release memory for the variables
    del lnc_ltc_doc_vectors
    del lnc_ltc_query_vectors
    del lnc_ltc_ranks
    
    lnc_Ltc_doc_vectors = apply_lnc_Ltc(tf_idf, filtered_vocab)
    lnc_Ltc_query_vectors = apply_lnc_Ltc(query_tf_idf, filtered_vocab)
    lnc_Ltc_ranks = rank_documents(lnc_Ltc_query_vectors, lnc_Ltc_doc_vectors)
    write_ranked_results_to_file(lnc_Ltc_ranks, 'Assignment2_20CS30069_ranked_list_B.txt')
    print("Written to B.txt")
    
    del lnc_Ltc_doc_vectors
    del lnc_Ltc_query_vectors
    del lnc_Ltc_ranks
    
    
    anc_apc_doc_vectors = apply_anc_apc(tf_idf, filtered_vocab)
    anc_apc_query_vectors = apply_anc_apc(query_tf_idf, filtered_vocab)
    anc_apc_ranks = rank_documents(anc_apc_query_vectors, anc_apc_doc_vectors)
    write_ranked_results_to_file(anc_apc_ranks, 'Assignment2_20CS30069_ranked_list_C.txt')
    print("Written to C.txt")
    
    del anc_apc_doc_vectors
    del anc_apc_query_vectors
    del anc_apc_ranks

    return document_vectors, query_vectors, filtered_vocab

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python Assignment2_<ROLL_NO>_ranker.py <path_to_data_folder> <path_to_model_queries.bin>")
        sys.exit(1)

    # Get the command line arguments
    csv_file = sys.argv[1]
    inverted_index_file = sys.argv[2]
    
    # Construct file paths
    query_file = "query_file.xml"
    
    # Call the main function with command-line arguments
    document_vectors, query_vectors, vocab = main(inverted_index_file, query_file, csv_file)



# if __name__ == "__main__":
#     # inverted_index_file = 'dummy_model.bin'
#     # query_file = 'query_file.xml'
#     # csv_file = 'dummy.csv'
    
#     inverted_index_file = 'model_queries_20CS30069.bin'
#     query_file = 'query_file.xml'
#     csv_file = "cord-19_2020-03-27\\2020-03-27\\metadata.csv"
    
#     document_vectors, query_vectors, vocab = main(inverted_index_file, query_file, csv_file)
