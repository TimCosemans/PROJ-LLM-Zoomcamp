import json
from sentence_transformers import SentenceTransformer

class RAG():
    
    def __init__(self, data_path, field_to_encode, index_name, encoder, encoder_model, llm_model, es_client, ollama_client=None):
        self.data_path = data_path
        self.field_to_encode = field_to_encode
        self.index_name = index_name
        self.encoder = encoder
        self.encoder_model = encoder_model
        self.llm_model = llm_model
        self.es_client = es_client  
        self.ollama_client = ollama_client  

    def fit(self):
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
    
    def predict(self, query):
        """
        Searches the Elasticsearch index for the most relevant documents based on the query.

        We use Ollama to encode the query and then use Elasticsearch to retrieve relevant context.
        For more info, see:
        - https://collabnix.com/building-rag-applications-with-ollama-and-python-complete-2025-tutorial/
        - https://github.com/ollama/ollama-python?tab=readme-ov-file
        """

        query_encoded = self._encode(query, encoder='ollama', model=self.ollama_client, model_name=self.llm_model)
        context = self._lookup_context(query, query_encoded)
        answer = self._answer_query(query, context)

        return answer



    def _lookup_context(self, query, query_encoded):
        """
        Retrieves relevant context from the Elasticsearch index based on the query.

        Full-text search uses bm25 on the text and query directly. 
        Vector search can be based ond dense vectors (and knn to find similar vectors) or sparse vectors. 
        Among sparse vector models, ES provides ELSER. ELSER is most easy to use, but not available in the free version of ES.
        We therefore use dense vectors. 
        Hybrid search uses vector and full-text search together.

        For more info, see: 
        - https://www.elastic.co/docs/solutions/search/search-approaches
        - https://www.elastic.co/docs/reference/elasticsearch/clients/python/examples
        - https://github.com/elastic/elasticsearch-labs/tree/main/notebooks/search
        - https://www.elastic.co/docs/solutions/search/vector/knn
        """
        context = []

        knn_query = {
            "field": "question_encoded",
            "query_vector": query_encoded,
            "k": 5,
            "num_candidates": 10000, 
        }
        query = {
            "match": {
            "question": {
                "query": query,
            }
            }
        }
        res = self.es_client.search(index=self.index_name, 
                            query=query, knn=knn_query) #, source=["text", "section", "question", "course"])
        
        for hit in res['hits']['hits']:
            context.append(hit['_source'])

        return context

    def _answer_query(self, query, context):
        """Generate response using Ollama with retrieved context."""

        prompt_template = """
        You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
        Use only the facts from the CONTEXT when answering the QUESTION.

        QUESTION: {query}

        CONTEXT: 
        {context}
        """.strip()

        context = [[f'{key}: {value}' for key, value in context_item.items()] for context_item in context]
        context = "\n".join(context)
        
        prompt = prompt_template.format(question=query, context=context).strip()


        response = self.ollama_client.generate(
            model=self.llm_model,
            prompt=prompt,
            options={
                'temperature': 0.7,
                'top_p': 0.9,
                'max_tokens': 500
            }
        )

        return response['response']

    def _encode_documents(self, docs, mapping):
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
                doc[f"{self.field_to_encode}_encoded"] = self._encode(doc[self.field_to_encode], encoder=self.encoder, model=model, model_name=self.encoder_model)
        
        
        mapping["mappings"]["properties"][f"{self.field_to_encode}_encoded"] = {"type": "dense_vector", "dims": len(docs[0][f'{self.field_to_encode}_encoded']), "index": True, "similarity": "cosine"},

        return (docs, mapping)

    def _encode(self, string, encoder, model=None, model_name=None):
        """
        Encodes a single document using the specified encoder model.
        """
        if encoder == "ollama":
            # Using Ollama to embed the documents
            result = model.embed(model=model_name, input=string)[0].tolist() #TO TEST
        elif encoder == "huggingface":
            # Using Hugging Face to embed the documents
            model = SentenceTransformer(model_name)
            # Transforming the title into an embedding using the model
            result = model.encode(string).tolist()
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

        return result

    def _define_mapping(self, docs): 
        """
        Defines the mapping for the Elasticsearch index.
        """

        unique_keys = set([doc.keys() for doc in docs])
        mapping = {
        "mappings": {
            "properties": {
                key: {"type": "text"} for key in unique_keys()
                }
            }   
        }
        return mapping
    
