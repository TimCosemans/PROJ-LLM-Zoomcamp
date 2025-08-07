import json
from ollama import Client
import os
import sys
import numpy as np

from src.rag import RAG
sys.path.append('./')
import pandas as pd

from src.elasticsearch_utils import define_simple_mapping, save_docs 
from src.elasticsearch_utils import client

GROUND_TRUTH_MODEL = "llama3.2"
DATA_PATH = 'data/documents-llm.json'
GROUND_TRUTH_DATA_PATH = 'data/ground_truth.csv'

def relevance(query, answer, ollama_client, llm_model=GROUND_TRUTH_MODEL):
    prompt_template = """
    You are an API that evaluates a Retrieval-Augmented Generation (RAG) system.
    You analyze the relevance of the generated answer to the given question.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Your response is composed using strictly and exclusively the information provided in the given record. 
    
    Here is the data for evaluation:

    Question: {query}
    Generated Answer: {answer}   
    
    Do not provide any additional commentary. 
    Provide the output in parsable JSON without using code blocks on the following format:

    {{
      "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
      "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    prompt = prompt_template.format(query=query, answer=answer)
    response = ollama_client.generate(
            model=llm_model,
            prompt=prompt,
            options={
                'temperature': 0.7,
                'top_p': 0.9,
                'max_tokens': 500
            }
    )
    
    try:
        json_eval = json.loads(response['response'])
        return json_eval['Relevance'], json_eval['Explanation']
    except json.JSONDecodeError:
        return "UNKNOWN", "Failed to parse evaluation"
    
def save_results(es_client, results, index_name):

    mapping = {
        "mappings": {
            "properties": {
                "conversation_id": {"type": "keyword"},
                "encoder": {"type": "keyword"},
                "encoder_model": {"type": "keyword"},
                "n_context_docs": {"type": "integer"},
                "llm_model": {"type": "keyword"},
                "rag_id": {"type": "keyword"},
                "user_input": {"type": "keyword"},
                "answer": {"type": "keyword"},
                "answer_language": {"type": "keyword"},
                "answer_time": {"type": "integer"},
                "relevance": {"type": "keyword"},
                "explanation": {"type": "keyword"},
                "user_feedback": {"type": "integer"}
                }
            }   
        }
    
    save_docs(es_client, index_name, mapping, [results], delete_index=False)

def generate_ground_truth_data(ollama_client, ground_truth_data_path=GROUND_TRUTH_DATA_PATH, data_path=DATA_PATH):
    
    with open(data_path, 'rt') as f_in:
        docs_raw = json.load(f_in)
    docs = docs_raw[0]['documents']

    list_results = []
    for doc in docs: 
        questions = _generate_ground_truth(doc, ollama_client=ollama_client)

        for question in questions:
            results = {}
            results['id'] = doc['id']
            results['question'] = question
            list_results.append(results)

    df = pd.DataFrame(list_results)
    df.to_csv(ground_truth_data_path, index=False)
        
def _generate_ground_truth(record, ollama_client, llm_model=GROUND_TRUTH_MODEL):
    prompt_template =  """
        You emulate a computer program that automatically generates questions based on a record. 
        Your task is to generate 5 questions based on the provided record.
        These questions are composed using strictly and exclusively the information provided in the given record. 
        
        Your response should contain the answer to the questions, and the questions should be complete and not too short.
        If possible, use as few words as possible from the record. 

        The record: {record}

        Do not explain your answer or provide any additional commentary. 
        Provide the output in parsable JSON without using code blocks on the following format:

        {{
            "Questions": ["question1", "question2", ..., "question5"]
        }}
        
        """.strip()
    

    prompt = prompt_template.format(record=record)
    response = ollama_client.generate(
            model=llm_model,
            prompt=prompt,
            options={
                'temperature': 0.7,
                'top_p': 0.9,
                'max_tokens': 500
            }
    )
    
    try:
        json_eval = json.loads(response['response'])
        return json_eval['Questions']
    except json.JSONDecodeError:
        return "Failed to parse questions"
    
def offline_evaluation(rag: RAG, ollama_client, es_client):
      
    metrics = _metrics_retrieval(rag)
    metrics['rag_id'] = rag.id
    metrics.update(_metrics_rag(rag, ollama_client, es_client))
    
    mapping = {
        "mappings": {
            "properties": {
                key: {"type": "float"} for key in metrics.keys() if key != 'rag_id'
                }
            }   
        }
    mapping["mappings"]["properties"]["rag_id"] = {"type": "keyword"}

    save_docs(es_client, "rag-evaluation", mapping, [metrics], delete_index=False)

def _metrics_retrieval(rag, ground_truth_data_path=GROUND_TRUTH_DATA_PATH):
    """
    Computes metrics for the retrieval system.
    
    Args:
        ground_truth_index (str): The index name for ground truth data.
        documents_index (str): The index name for documents.
        es_client (Elasticsearch): The Elasticsearch client.
    
    Returns:
        dict: A dictionary containing the computed metrics.
    """
    
    docs = pd.read_csv(ground_truth_data_path)
    relevance_total = []

    def _relevance(doc, relevance_total):
        context = rag.lookup_context(doc['question']) # will always be the question field
        doc_id = doc['id']
        relevance = [d['id'] == doc_id for d in context]
        relevance_total.append(relevance)

    docs.apply(lambda x: _relevance(x, relevance_total), axis=1)

    return {
        'hit_rate': _hit_rate(relevance_total),
        'mrr': _mrr(relevance_total),
    }

  
def _metrics_rag(rag, ollama_client, es_client, ground_truth_data_path=GROUND_TRUTH_DATA_PATH):
    """
    Computes metrics for the RAG system.
    
    Args:
        ground_truth_index (str): The index name for ground truth data.
        documents_index (str): The index name for documents.
        es_client (Elasticsearch): The Elasticsearch client.
        rag (RAG): The RAG instance to evaluate.
    
    Returns:
        dict: A dictionary containing the computed metrics.
    """

    docs = pd.read_csv(ground_truth_data_path)

    relevance_scores = []
    cosine_similarities = []

    def _metrics(doc, relevance_scores, cosine_similarities):
        query = doc['question']
        answer = rag.predict(query)
        relevance_score, _ = relevance(query, answer, ollama_client)
        relevance_scores.append(relevance_score)

        answer_encoded = rag.encode(answer) # numeric representation of the answer
        ground_truth_answer = es_client.search(index=rag.index_name, 
                                               query={"match": {"id": doc['id']}})
        ground_truth_answer_encoded = ground_truth_answer['hits']['hits'][0]['_source'][f'{rag.field_to_encode}_encoded']
        cosine_similarity = np.array(answer_encoded) @ np.array(ground_truth_answer_encoded).T

        cosine_similarities.append(cosine_similarity)

    docs.iloc.apply(lambda x: _metrics(x, relevance_scores, cosine_similarities), axis=1)
        
    return {
        'count_non_relevant': relevance_scores.count("NON_RELEVANT"),
        'count_partly_relevant': relevance_scores.count("PARTLY_RELEVANT"),
        'count_relevant': relevance_scores.count("RELEVANT"),
        'average_cosine_similarity': sum(cosine_similarities) / len(cosine_similarities),
        'std_cosine_similarity': pd.Series(cosine_similarities).std(),
        'median_cosine_similarity': pd.Series(cosine_similarities).median(), 
        'max_cosine_similarity': max(cosine_similarities),
        'min_cosine_similarity': min(cosine_similarities),
        'q25_cosine_similarity': pd.Series(cosine_similarities).quantile(0.25),
        'q75_cosine_similarity': pd.Series(cosine_similarities).quantile(0.75)
    }


def _hit_rate(relevance_total):
    """
    Computes the hit rate from the relevance total. 
    This is the proportion of queries for which at least one relevant document was retrieved.

    """

    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def _mrr(relevance_total):
    """
    Computes the Mean Reciprocal Rank (MRR) from the relevance total.
    MRR is the average of the reciprocal ranks of the first relevant document for each query.
    """

    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)