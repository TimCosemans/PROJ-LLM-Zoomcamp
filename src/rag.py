import json
from sentence_transformers import SentenceTransformer
import uuid 

from src.elasticsearch_utils import save_docs, define_simple_mapping

class RAG():
    
    def __init__(self, data_path, field_to_encode, index_name, encoder, encoder_model, n_context_docs, llm_model, es_client, ollama_client=None, answer_language='english'):
        
        self.data_path = data_path
        self.field_to_encode = field_to_encode
        self.index_name = index_name
        self.encoder = encoder
        self.encoder_model = encoder_model
        self.n_context_docs = n_context_docs
        self.llm_model = llm_model
        self.es_client = es_client  
        self.ollama_client = ollama_client  
        self.answer_language = answer_language
        self.id = str(uuid.uuid4())

    def fit(self):
        """
        Uploads documents to the Elasticsearch index. Option to also perform offline evaluation of the RAG system.
        """

        with open(self.data_path, 'rt') as f_in:
            docs_raw = json.load(f_in)

        docs = docs_raw[0]['documents']

        mapping = define_simple_mapping(docs)
        
        docs, mapping = self._encode_documents(docs, mapping)

        # Save the documents to Elasticsearch
        save_docs(self.es_client, self.index_name, mapping, docs)

        return self
    
    def predict(self, query):
        """
        Searches the Elasticsearch index for the most relevant documents based on the query.

        We use Ollama to encode the query and then use Elasticsearch to retrieve relevant context.
        For more info, see:
        - https://collabnix.com/building-rag-applications-with-ollama-and-python-complete-2025-tutorial/
        - https://github.com/ollama/ollama-python?tab=readme-ov-file
        """

        self._setup_ollama_model(self.llm_model)
        context = self.lookup_context(query)
        answer = self._answer_query(query, context)

        return answer



    def lookup_context(self, query):
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
        - https://github.com/elastic/elasticsearch-labs/blob/main/notebooks/search/02-hybrid-search.ipynb
        - https://www.elastic.co/guide/en/elasticsearch/reference/8.18/rrf.html#CO875-3
        """
        query_encoded = self.encode(query)

        context = []

        query_full = {
            "rrf": { 
                "retrievers": [
                    {
                        "standard": { 
                            "query": {
                                "match": {
                                    f"{self.field_to_encode}": {
                                        "query": query,
                                    }
                                }
                            }
                        }
                    },
                    {
                        "knn": {
                            "field": f"{self.field_to_encode}_encoded",
                            "query_vector": query_encoded,
                            "k": 50,
                            "num_candidates": 100, 
                        }
                    }
                ],
                "rank_window_size": 60,
                "rank_constant": 10
            }
        }
        
        res = self.es_client.search(index=self.index_name,
                                    retriever=query_full, 
                                    size=self.n_context_docs)

        for hit in res['hits']['hits']:
            tmp = hit['_source']
            tmp = {key: value for key, value in tmp.items() if not key.endswith('_encoded')}  # Remove '_encoded' keys
            context.append(tmp)

        return context

    def _answer_query(self, query, context):
        """Generate response using Ollama with retrieved context."""

        prompt_template = """
        You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
        Use only the facts from the CONTEXT when answering the QUESTION.

        Answer only in {language}.

        QUESTION: {query}

        CONTEXT: 
        {context}
        """.strip()

        context = [[f'{key}: {value}' for key, value in context_item.items()] for context_item in context]
        context = "\n\n".join(["\n".join(context_item) for context_item in context])
        
        prompt = prompt_template.format(query=query, context=context, language=self.answer_language).strip()

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
    
    def _setup_ollama_model(self, model_name):
        """Download and setup the Ollama model if not available."""
        try:
            self.ollama_client.show(model_name)
        except:
            print(f"Downloading {model_name}...")
            self.ollama_client.pull(model_name)

    def _encode_documents(self, docs, mapping):
        """
        Encodes the documents using the specified encoder model.
        """


        for doc in docs:
                doc[f"{self.field_to_encode}_encoded"] = self.encode(doc[self.field_to_encode])
        
        
        mapping["mappings"]["properties"][f"{self.field_to_encode}_encoded"] = {"type": "dense_vector", 
                                                                                "dims": len(docs[0][f'{self.field_to_encode}_encoded']), 
                                                                                "index": True, 
                                                                                "similarity": "cosine"}

        return (docs, mapping)

    def encode(self, string):
        """
        Encodes a single document using the specified encoder model.
        """
            
        if self.encoder == "ollama":
            # Using Ollama to embed the documents
            model = self.ollama_client
            self._setup_ollama_model(self.encoder_model)
            result = model.embed(model=self.encoder_model, input=string)['embeddings'][0] 
        elif self.encoder == "huggingface":
            # Using Hugging Face to embed the documents
            model = SentenceTransformer(self.encoder_model)
            # Transforming the title into an embedding using the model
            result = model.encode(string).tolist()
        else:
            raise ValueError(f"Unknown encoder: {self.encoder}")

        return result

    
    
