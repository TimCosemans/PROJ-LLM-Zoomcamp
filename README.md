# PROJ-LLM-Zoomcamp

The following docker containers are started:
- eslasticsearch: This is the container that runs the Elasticsearch service, which is used to store and search the data.
- kibana: This is the container that runs the Kibana service, which is used to visualize the data in Elasticsearch and perform the monitoring.
- ollama: This is the container that runs the ollama service, which is used to anwer the questions from the streamlit app.
- setup: This makes sure there is a connection between the kibana and elasticsearch container. In addition, it reads the data from the `data` folder and loads it into elasticsearch. Lastly, it adds a dense vector representation of the questions to the index using a huggingface model.
    - Elasticsearch supports both full-text (using bm25) as well as vector search (based on dense or sparse vectors). We want to allow for a multilingual encoding of the questions, so we use a multilingual model from Hugging Face to generate dense vectors. We will therefore not use the managed workflows in semantic search.
    - We  combine both full-text and vector search in a hybrid search. This allows us to use the strengths of both methods.

