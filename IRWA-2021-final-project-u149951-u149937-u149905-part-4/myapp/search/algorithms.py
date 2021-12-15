import web_app
from collections import defaultdict
import numpy as np
import math
import collections
from numpy import linalg as la

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from array import array

nltk.download('stopwords')

def search_in_corpus(query,corpus):

    # Applying preprocessing to query
    query_terms = preprocess(query)

    # 1. create create_tfidf_index
    indx, tf_score, doc_freq, idf_score = compute_TFIDF(corpus)

    # 2. apply ranking
    #result_docs, scores = rank_documents(query_terms, corpus, indx, idf_score, tf_score)
    
    # 3. Search
    docs = set()
    for term in query_terms:
        try:
            # store in term_docs the ids of the docs that contain "term"                        
            docs = indx[term]
            # docs = docs Union term_docs
        except:
            #term is not in index
            pass
    docs = list(docs)
    ranked_docs, scores = rank_documents(query_terms, docs, indx, idf_score, tf_score)
    
    return ranked_docs



def compute_TFIDF(tweets):

    doc_freq = defaultdict(int)
    tf_score = defaultdict(list)
    indx = defaultdict(list)
    idf_score = defaultdict(float)

    num_tweets = len(tweets)

    for id in tweets.keys():
        text = tweets[id].description
        tweets[id].words = preprocess(text)
        inv_indx_act = {}
        freqs_act = {}

        #compute frequencies
        for word in tweets[id].words:
           inv_indx_act[word] = id
           freqs_act[word] = float(tweets[id].words.count(word))

        #Normalize tf
        normalized = 0
        for word in inv_indx_act.keys():
            normalized = normalized + freqs_act[word] ** 2
        normalized = math.sqrt(normalized)

        #compute tf
        for word in inv_indx_act.keys():
            f = freqs_act[word]/normalized
            tf_score[word].append(f)
            doc_freq[word] = doc_freq[word] + 1

        for word, id in inv_indx_act.items():
            indx[word].append(id)
        for word in doc_freq:
            idf_score[word] = np.log(float(len(tweets)/doc_freq[word]))

    return indx, tf_score, doc_freq, idf_score

def rank_documents(terms, docs, index, idf, tf):

    # We are interested only on the element of the docVector corresponding to the query terms 
    # The remaining elements would became 0 when multiplied to the query_vector
    doc_vectors = defaultdict(lambda: [0] * len(terms)) # We call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query. 
    # Example: collections.Counter(["hello","hello","world"]) --> Counter({'hello': 2, 'world': 1})

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue

        ## Computing tf*idf(normalize TF as done with documents)
        query_vector[termIndex]= query_terms_count[term]/query_norm * idf[term]
        
        # Generating doc_vectors for matching docs
        for doc_index, (id) in enumerate(index[term]):          
            if id in docs:
                doc_vectors[id][termIndex] = tf[term][doc_index] * idf[term]

    # Calculating the score of each doc 
    # Computing the cosine similarity between queyVector and each docVector:

    doc_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items() ]
    doc_scores.sort(reverse=True)
    scores = [x[0] for x in doc_scores]
    result_docs = [x[1] for x in doc_scores]

    if len(result_docs) == 0:
        print("No results found, try again")
    return result_docs, scores


def preprocess(line):
    #stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    line = line.lower() ## Transform in lowercase
    line = line.split() ## Tokenize the text to get a list of terms
    line =  [l for l in line if l not in stopwords.words("english")] ##eliminate the stopwords (HINT: use List Comprehension)
    line = [stemmer.stem(l) for l in line] ## perform stemming (HINT: use List Comprehension)
    line = [l for l in line if "https://" not in l ] ##deletes the items that are url links 
    
    line = [l for l in line if "" != l ]

    return line