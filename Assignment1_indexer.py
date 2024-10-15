from datetime import date
import os
import sys
import nltk
import pickle
import csv
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to export the inverted index as a text file
def export_index_as_text(inverted_idx, file_name):
    with open(file_name, 'w') as output_file:
        for term, doc_list in inverted_idx.items():
            doc_ids_str = ' '.join(map(str, doc_list))
            output_file.write(f"{term}: {doc_ids_str}\n")
    print("Text file created successfully.")

# Function to extract documents from the CSV file
def extract_documents(file_location):
    doc_collection = {}
    # with open(file_location, 'r', newline='', encoding='utf-8') as input_file:
    with open(file_location, 'r', newline='', encoding='ISO-8859-1') as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            doc_identifier = row['cord_uid'].strip()  # Using 'cord_uid' as the document identifier
            # document_content = f"{row['title']} {row['abstract']}"  # Combining title and abstract as the document content
            document_content = row['abstract'].strip() 
            doc_collection[doc_identifier] = document_content.strip()
    return doc_collection

# Function to clean and tokenize text
def clean_text(text_input):
    stop_words_set = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text_input.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words_set and token not in string.punctuation]
    return tokens

# Function to construct the inverted index
def construct_inverted_index(docs):
    inv_index = defaultdict(list)
    for doc_id, content in docs.items():
        tokens = clean_text(content)
        for term in set(tokens):
            inv_index[term].append(doc_id)
    return inv_index

# Function to persist the inverted index to a binary file
def persist_inverted_index(inv_idx, file_name):
    with open(file_name, 'wb') as binary_file:
        pickle.dump(inv_idx, binary_file)

# Main execution block
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Assignment2_20CS30069_indexer.py <path to dataset file>")
        sys.exit(1)

    csv_input_file = sys.argv[1]
    
    # Extract documents from the CSV file
    documents = extract_documents(csv_input_file)
    
    # Construct the inverted index
    inverted_idx = construct_inverted_index(documents)
    
    # Persist the inverted index
    persist_inverted_index(inverted_idx, 'model_queries_20CS30069.bin')
    # persist_inverted_index(inverted_idx, 'dummy_model.bin')
    print("\nInverted index created and saved successfully.")
