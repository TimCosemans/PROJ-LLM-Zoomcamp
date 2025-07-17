import json

from src.elasticsearch import save_docs 

def relevance(query, answer, ollama_client, llm_model):
    prompt_template = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
    Your task is to analyze the relevance of the generated answer to the given question.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {query}
    Generated Answer: {answer}

    Please analyze the content and context of the generated answer in relation to the question
    and provide your evaluation in parsable JSON without using code blocks:

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
                "llm_model": {"type": "keyword"},
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
    
