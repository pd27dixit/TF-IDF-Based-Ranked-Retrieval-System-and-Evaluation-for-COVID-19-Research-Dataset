20CS30069

#LIBRARIES REQUIREMENT
Python 3.12.4  //version of python used here

python.exe -m pip install --upgrade pip
python -m pip install --upgrade pip
pip3 install nltk


python -m pip install --upgrade pip setuptools
pip install scikit-learn
pip install pandas

#DESIGN DETAILS
Vocabulary length: 17557

Preprocessing Pipeline:
Tokenization: Splits text into individual words.
Stop Words Removal: Filters out common words that don’t add significant meaning (e.g., "the", "is").
Punctuation Removal: Excludes punctuation marks.
Lemmatization: Reduces words to their base or root form (e.g., "running" to "run").


#STEPS TO RUN THE ASSIGNMENT TASKS
python Assignment1_indexer.py sampled_data.csv

python Assignment2_20CS30069_ranker.py sampled_data.csv model_queries_20CS30069.bin

python Assignment2_20CS30069_evaluator.py relevance_file.txt Assignment2_20CS30069_ranked_list_A.txt
python Assignment2_20CS30069_evaluator.py relevance_file.txt Assignment2_20CS30069_ranked_list_B.txt
python Assignment2_20CS30069_evaluator.py relevance_file.txt Assignment2_20CS30069_ranked_list_C.txt


#ADDITIONAL INFO
idf(t) = log_10_(N/df(t))   //as mentioned in lectures

#ranker.py
1) 
def vectorize()
This function converts documents and queries into vectors based on a predefined vocabulary, using a dictionary to map terms to their corresponding TF-IDF weights, facilitating efficient document representation for retrieval tasks.

2)
lnc.ltc: Applies logarithmic term frequency without document length normalization for documents, normalizing only queries for cosine similarity.

lnc.Ltc: Uses logarithmic term frequency with document length normalization for documents and applies logarithmic normalization for queries.

anc.apc: Implements augmented frequency, normalizing both queries and documents using the maximum term frequency for cosine similarity.

3)
The cosine similarity function uses the dot product and vector norms to quantify similarity while efficiently handling common terms, ensuring robustness against zero vectors to prevent division errors.


#evaluator.py
1)
The Average Precision (AP) uses cumulative precision at each relevant document up to 'k' to assess retrieval quality

2)
NDCG incorporates graded relevance and logarithmic discounting to prioritize higher-ranked relevant documents, offering a balanced evaluation of ranking effectiveness.

3) choice made: if a query ID not present in relevance_file.txt, it is considered irrelevant
4) choice made: judgement of 1 or 2, are both considered by me as under relevant.