from elasticsearch import Elasticsearch

def client(host):
    """
    Create an Elasticsearch client.

    Args:
        host (str): The host URL of the Elasticsearch instance.
    
    Returns:
        Elasticsearch: An instance of the Elasticsearch client.
    """
    #https://discuss.elastic.co/t/issue-connecting-python-to-elasticsearch-in-docker-environment/361507/2
    es_client = Elasticsearch(
        hosts=[host],
        basic_auth=('elastic', 'password'), #os.getenv("ELASTIC_PASSWORD")), CHANGE 
        verify_certs=False,
        max_retries=30,
        retry_on_timeout=True,
        request_timeout=30,
    )

    return es_client


def save_docs(es_client, index_name, mapping, docs, delete_index=True):
    """
    Save documents to Elasticsearch index.

    Args:
        es_client (Elasticsearch): The Elasticsearch client.
        index_name (str): The name of the index to save documents to.
        mapping (dict): The mapping for the index.
        docs (list): A list of documents to save.
        delete_index (bool): Whether to delete the index before saving. Defaults to True.
    
    Returns:
        None
    """

    if delete_index:
        es_client.indices.delete(index=index_name, ignore_unavailable=True)
        es_client.indices.create(index=index_name, body=mapping)
    else:
        if not es_client.indices.exists(index=index_name):
            es_client.indices.create(index=index_name, body=mapping)
    
    for doc in docs: 
        es_client.index(index=index_name, document=doc)

    return None 

def define_simple_mapping(docs): 
        """
        Defines the mapping for the Elasticsearch index.
        """

        unique_keys = {k for doc in docs for k in doc.keys()}
        mapping = {
        "mappings": {
            "properties": {
                key: {"type": "keyword"} for key in unique_keys
                }
            }   
        }
        return mapping