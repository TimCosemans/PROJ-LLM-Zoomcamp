import os 
from elasticsearch import Elasticsearch
from ollama import Client

ENCODER  = "huggingface"
ENCODER_MODEL = "intfloat/multilingual-e5-small" 
LLM_MODEL = "llama3.2"
INDEX_NAME = "llm-doc"
import sys
sys.path.append('./src')  # Adjust the path to your src directory

#https://discuss.elastic.co/t/issue-connecting-python-to-elasticsearch-in-docker-environment/361507/2
es_client = Elasticsearch(
    hosts=["https://localhost:9200"],
    basic_auth=('elastic', os.getenv("ELASTIC_PASSWORD")),
    verify_certs=False,
    max_retries=30,
    retry_on_timeout=True,
    request_timeout=30,
)

ollama = Client(host='http://localhost:11434')

from rag import RAG

# Dynamically construct the path to the data folder
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data/documents-llm.json')

rag = RAG(
    data_path=data_path,
    field_to_encode='question',
    index_name=INDEX_NAME,
    encoder=ENCODER,
    encoder_model=ENCODER_MODEL,
    llm_model=LLM_MODEL,
    es_client=es_client,
    ollama_client=ollama, 
    answer_language='dutch'
)

rag = rag.fit()

query = "Do I have to hand in the homework?"
answer = rag.predict(query)
