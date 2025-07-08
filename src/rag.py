import os 
from elasticsearch import Elasticsearch
import json
from sentence_transformers import SentenceTransformer
from ollama import Client

class RAG():
    
    def __init__(self, data_path, field_to_encode, index_name, encoder_model, llm_model, es_client, ollama_client=None):
        self.data_path = data_path
        self.field_to_encode = field_to_encode
        self.index_name = index_name
        self.encoder_model = encoder_model
        self.llm_model = llm_model
        self.es_client = es_client  
        self.ollama_client = ollama_client  

    def upload_documents(self):
        """
        Uploads documents to the Elasticsearch index.
        """

        with open(self.path, 'rt') as f_in:
            docs_raw = json.load(f_in)

        docs = docs_raw[0]['documents']

        mapping = self.define_mapping(docs)
        
        docs, mapping = self.encode_documents(docs, mapping, self.ollama_client, self.field_to_encode)

        self.es_client.indices.delete(index=self.index_name, ignore_unavailable=True)
        self.es_client.indices.create(index=self.index_name, body=mapping)
        for doc in docs: 
            self.es_client.index(index="llm-doc", document=doc)

        return docs


    def encode_documents(self, docs, mapping):
        """
        Encodes the documents using the specified encoder model.
        """

        if self.encoder == "ollama":
            # Using Ollama to embed the documents
            model = self.ollama_client
        elif self.encoder == "huggingface":
            # Using Hugging Face to embed the documents
            model = SentenceTransformer(self.encoder_model)

        for doc in docs:
                doc[f"{self.field_to_encode}_encoded"] = self._encode(doc[self.field_to_encode], model=model)
        
        
        mapping["mappings"]["properties"][f"{self.field_to_encode}_encoded"] = {"type": "dense_vector", "dims": len(docs[0][f'{self.field_to_encode}_encoded']), "index": True, "similarity": "cosine"},

        return (docs, mapping)

    def _encode(self, string, model=None):
        """
        Encodes a single document using the specified encoder model.
        """
        if self.encoder == "ollama":
            # Using Ollama to embed the documents
            result = model.embed(model=self.encoder_model, input=string)[0].tolist() #TO TEST
        elif self.encoder == "huggingface":
            # Using Hugging Face to embed the documents
            model = SentenceTransformer(self.encoder_model)
            # Transforming the title into an embedding using the model
            result = model.encode(string).tolist()
        else:
            raise ValueError(f"Unknown encoder: {self.encoder}")

        return result

    def define_mapping(self, docs): 
        """
        Defines the mapping for the Elasticsearch index.
        """
        mapping = {
        "mappings": {
            "properties": {
                key: {"type": "text"} for key in docs[0].keys() 
                }
            }   
        }
        return mapping
