import numpy as np

def read_ranked_list(file_name):
    """Reads the ranked list file and returns a dictionary of query IDs and their ranked document IDs."""
    ranked_list = {}
    with open(file_name, 'r') as f:
        for line in f:
            parts = line.strip().split(':')  # Split on the colon
            if len(parts) != 2:
                continue  # Skip any malformed lines
            
            query_id = parts[0].strip()  # Get the query ID
            ranked_docs = parts[1].strip().split()  # Get the ranked documents and split them into a list
            ranked_list[query_id] = ranked_docs  # Store in dictionary

    return ranked_list

def read_relevance_judgments(file_name):
    """Reads the relevance judgments file and returns a dictionary of query IDs and their relevant documents."""
    relevance = {}
    with open(file_name, 'r') as f:
        for line in f:
            parts = line.strip().split()
            query_id = parts[0]
            doc_id = parts[2]
            judgment = int(parts[3])  # Judgment: 0, 1, or 2
            
            if query_id not in relevance:
                relevance[query_id] = {}
            relevance[query_id][doc_id] = judgment
    return relevance

def average_precision(ranked_docs, relevant_docs, k):
    """Calculates Average Precision (AP) at k."""
    # print(len(ranked_docs))
    if k > len(ranked_docs):
        k = len(ranked_docs)
    
    relevant_count = 0
    precision_sum = 0.0
    
    for i in range(k):
        # if ranked_docs[i] in relevant_docs and relevant_docs[ranked_docs[i]] > 0:
        if ranked_docs[i] in relevant_docs:
            # print("Present")
            if relevant_docs[ranked_docs[i]] > 0:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
    
    # print(f'Relevant count: {relevant_count}')
    if relevant_count == 0:
        return 0.0
    return precision_sum / relevant_count

def ndcg(ranked_docs, relevant_docs, k):
    """Calculates Normalized Discounted Cumulative Gain (NDCG) at k."""
    dcg = 0.0
    idcg = 0.0
    
    # Calculate DCG
    for i in range(min(k, len(ranked_docs))):
        relevance = relevant_docs.get(ranked_docs[i], 0)  # Default to 0 if doc not found
        dcg += (2 ** relevance - 1) / np.log2(i + 2)
    
    # Calculate IDCG (Ideal DCG)
    ideal_relevances = sorted(relevant_docs.values(), reverse=True)
    for i in range(min(k, len(ideal_relevances))):
        idcg += (2 ** ideal_relevances[i] - 1) / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg



def evaluate_ranked_list(ranked_list, relevance_judgments):
    """Evaluate the ranked list for each query and return metrics."""
    metrics = {}
    for query_id, ranked_docs in ranked_list.items():
        relevant_docs = relevance_judgments.get(query_id, {})
        
        # print(len(relevant_docs))
        
        # Debug prints
        # print(f"Evaluating Query ID: {query_id}")
        # print(f"Ranked Documents: {ranked_docs}")
        # print(f"Relevant Documents: {relevant_docs}")
        
        ap_10 = average_precision(ranked_docs[:10], relevant_docs, 10)
        ap_20 = average_precision(ranked_docs[:20], relevant_docs, 20)
        ndcg_10 = ndcg(ranked_docs[:10], relevant_docs, 10)
        ndcg_20 = ndcg(ranked_docs[:20], relevant_docs, 20)
        
        # Print all metrics in one line
        # print(f"Metrics for Query ID {query_id}: AP@10={ap_10:.4f}, AP@20={ap_20:.4f}, NDCG@10={ndcg_10:.4f}, NDCG@20={ndcg_20:.4f}")
        
        metrics[query_id] = {
            'AP@10': ap_10,
            'AP@20': ap_20,
            'NDCG@10': ndcg_10,
            'NDCG@20': ndcg_20
        }
    
    return metrics


def calculate_mean_metrics(metrics):
    """Calculate Mean Average Precision and average NDCG from metrics."""
    ap_10_sum = 0.0
    ap_20_sum = 0.0
    ndcg_10_sum = 0.0
    ndcg_20_sum = 0.0
    query_count = len(metrics)
    
    for query_id, metric in metrics.items():
        ap_10_sum += metric['AP@10']
        ap_20_sum += metric['AP@20']
        ndcg_10_sum += metric['NDCG@10']
        ndcg_20_sum += metric['NDCG@20']
    
    return {
        'mAP@10': ap_10_sum / query_count,
        'mAP@20': ap_20_sum / query_count,
        'averNDCG@10': ndcg_10_sum / query_count,
        'averNDCG@20': ndcg_20_sum / query_count
    }

def write_metrics_to_file(metrics, file_name):
    """Writes the evaluation metrics to a file."""
    with open(file_name, 'w') as f:
        for query_id, metric in metrics.items():
            f.write(f"Query ID: {query_id}\n")
            f.write(f"AP@10: {metric['AP@10']:.4f}\n")
            f.write(f"AP@20: {metric['AP@20']:.4f}\n")
            f.write(f"NDCG@10: {metric['NDCG@10']:.4f}\n")
            f.write(f"NDCG@20: {metric['NDCG@20']:.4f}\n")
            f.write("\n")
        
        # Write averages
        mean_metrics = calculate_mean_metrics(metrics)
        f.write("Overall Metrics:\n")
        f.write(f"Mean AP@10: {mean_metrics['mAP@10']:.4f}\n")
        f.write(f"Mean AP@20: {mean_metrics['mAP@20']:.4f}\n")
        f.write(f"Average NDCG@10: {mean_metrics['averNDCG@10']:.4f}\n")
        f.write(f"Average NDCG@20: {mean_metrics['averNDCG@20']:.4f}\n")

import sys

def main():
    # Check for correct command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python Assignment2_<ROLL_NO>_evaluator.py <path_to_gold_standard_ranked_list.txt> <path_to_Assignment2_<ROLL_NO>_ranked_list_<K>.txt>")
        return

    # Get file paths from command-line arguments
    relevance_file = sys.argv[1]
    ranked_list_file = sys.argv[2]

    # Read relevance judgments
    relevance_judgments = read_relevance_judgments(relevance_file)

    # Read ranked list
    ranked_list = read_ranked_list(ranked_list_file)

    # Evaluate ranked list
    metrics = evaluate_ranked_list(ranked_list, relevance_judgments)

    # Write metrics to a file
    output_metrics_file = ranked_list_file.replace('.txt', '_metrics.txt')  # Create an output filename based on the input filename
    write_metrics_to_file(metrics, output_metrics_file)

    print(f"Metrics written to {output_metrics_file}.")

if __name__ == "__main__":
    main()

'''
def main():
    relevance_file = 'relevance_file.txt'
    ranked_list_A_file = 'Ranked_list_A.txt'
    ranked_list_B_file = 'Ranked_list_B.txt'
    ranked_list_C_file = 'Ranked_list_C.txt'

    # Read relevance judgments
    relevance_judgments = read_relevance_judgments(relevance_file)
    
    # # Print two entries from relevance_judgments
    # for i, (query_id, judgments) in enumerate(relevance_judgments.items()):
    #     if i < 2:  # Print only the first two entries
    #         print(f"Query ID: {query_id}, Judgments: {judgments}")
    #     else:
    #         break

    # # Read ranked lists
    ranked_list_A = read_ranked_list(ranked_list_A_file)
    
    # # Print two entries from ranked_list_A
    # for i, (query_id, ranked_docs) in enumerate(ranked_list_A.items()):
    #     if i < 2:  # Print only the first two entries
    #         print(f"Query ID: {query_id}, Ranked Documents: {ranked_docs}")
    #     else:
    #         break
        
    ranked_list_B = read_ranked_list(ranked_list_B_file)
    ranked_list_C = read_ranked_list(ranked_list_C_file)

    # Evaluate ranked lists
    metrics_A = evaluate_ranked_list(ranked_list_A, relevance_judgments)
    metrics_B = evaluate_ranked_list(ranked_list_B, relevance_judgments)
    metrics_C = evaluate_ranked_list(ranked_list_C, relevance_judgments)

    # Write metrics to files
    write_metrics_to_file(metrics_A, 'Metrics_A.txt')
    write_metrics_to_file(metrics_B, 'Metrics_B.txt')
    write_metrics_to_file(metrics_C, 'Metrics_C.txt')
    print("Metrics written to files.")

if __name__ == "__main__":
    main()
'''