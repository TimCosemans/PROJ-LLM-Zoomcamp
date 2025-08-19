# PROJ-LLM-Zoomcamp

The following docker containers are started:
- eslasticsearch: This is the container that runs the Elasticsearch service, which is used to store and search the data.
- kibana: This is the container that runs the Kibana service, which is used to visualize the data in Elasticsearch and perform the monitoring.
- ollama: This is the container that runs the ollama service, which is used to anwer the questions from the streamlit app.
- setup: This makes sure there is a connection between the kibana and elasticsearch container. In addition, it reads the data from the `data` folder and loads it into elasticsearch. Lastly, it adds a dense vector representation of the questions to the index using a huggingface model.
    - Elasticsearch supports both full-text (using bm25) as well as vector search (based on dense or sparse vectors). We want to allow for a multilingual encoding of the questions, so we use a multilingual model from Hugging Face to generate dense vectors. We will therefore not use the managed workflows in semantic search.
    - We  combine both full-text and vector search in a hybrid search. This allows us to use the strengths of both methods.

To make use of the reciprocal rank fusion (RRF) algorithm for reranking answers based on both vector and full-text search, go to Kibana > Stack Management > License Management and activate a 30-day trial license.

You can generate the ground truth data as well as perform the offline evaluation of any initiated rag using the functions in test.py. Beware, these will take a very long time to run on a local machine, so it is recommended to run them on a server with sufficient resources.

The app only performs fitting of the rag, prediction and online evaluation. The offline evaluation is done in the test.py file.

I implemented the following best practices:
- Hybrid search
- Document reranking using RRF

I did not implement the following best practices:
- Chunking (since the data is already in pretty small chunks)
- User query rewriting (running the LLM locally, this will take up even more resources)


Go to localhost:5601 and log in with username `elastic` and password `password` to access Kibana.


To run the test script, you need to install the required packages using the requirements.txt file. You can do this by running the following command in your terminal:

```bash
pip install -r requirements.txt