import streamlit as st
import time
import uuid
import os 

from ollama import Client
import sys
sys.path.append('./src')  # Adjust the path to your src directory

from src.rag import RAG
from src.elasticsearch_utils import client, get_recent_docs
from src.evaluate import generate_ground_truth_data, offline_evaluation, relevance, save_results

DOCS_INDEX_NAME = "llm-doc"
RESULTS_INDEX_NAME = "app-results"

es_client = client(host="https://localhost:9200")
ollama = Client(host='http://localhost:11434')

# generate_ground_truth_data(ollama)

conversation_id = str(uuid.uuid4())
encoder = "ollama"
encoder_model = "llama3.2"
llm_model = "llama3.2"
language = "english"
n_context_docs = 10

# Dynamically construct the path to the data folder
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data/documents-llm.json')

rag = RAG(
    data_path=data_path,
    field_to_encode='question',
    index_name=DOCS_INDEX_NAME,
    encoder=encoder,
    encoder_model=encoder_model,
    n_context_docs=n_context_docs,
    llm_model=llm_model,
    es_client=es_client,
    ollama_client=ollama, 
    answer_language=language
) #.fit()

# user_input = "Can I still join the course?"

# start_time = time.time()
# answer = rag.predict(user_input)
# end_time = time.time()

# relevance_score, explanation = relevance(user_input, answer, ollama, llm_model)
# results = {
#         "rag_id": rag.id,
#         "conversation_id": conversation_id,
#         "encoder": encoder,
#         "encoder_model": encoder_model,
#         "n_context_docs": n_context_docs,
#         "llm_model": llm_model,
#         "user_input": user_input,
#         "answer": answer,
#         "answer_language": language,
#         "answer_time": int(end_time - start_time),  
#         "relevance": relevance_score,
#         "explanation": explanation
#         }

# save_results(es_client, results, index_name=RESULTS_INDEX_NAME)

offline_evaluation(rag, ollama, es_client)

