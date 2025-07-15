import streamlit as st
import time
import uuid
import os 
from elasticsearch import Elasticsearch
from ollama import Client
import sys
sys.path.append('./src')  # Adjust the path to your src directory

from src.rag import RAG

INDEX_NAME = "llm-doc"

#https://discuss.elastic.co/t/issue-connecting-python-to-elasticsearch-in-docker-environment/361507/2
es_client = Elasticsearch(
    hosts=["https://localhost:9200"],
    basic_auth=('elastic', 'password'), #os.getenv("ELASTIC_PASSWORD")),
    verify_certs=False,
    max_retries=30,
    retry_on_timeout=True,
    request_timeout=30,
)

ollama = Client(host='http://localhost:11434')

def print_log(message):
    print(message, flush=True)


def main():
    print_log("Starting the LLM application")
    st.title("Multilingual Course Assistant Tester")

    # Session state initialization
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
        print_log(
            f"New conversation started with ID: {st.session_state.conversation_id}"
        )

    # Model selection
    encoder = st.selectbox("Select an encoder:", ["ollama", "huggingface"])
    if encoder == "ollama":
        print_log("Using Ollama as the encoder")
        encoder_model = st.selectbox(
            "Select a model:",
            ["llama3.2", "mistral-small3.2", "magistral", "phi4-mini-reasoning"],
        )
    else:   
        print_log("Using Hugging Face as the encoder")
        # For Hugging Face, we can use a fixed model or allow user selection
        encoder_model = st.selectbox(
            "Select a model:",
            ["intfloat/multilingual-e5-small", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
        )
    
    llm_model = st.selectbox(
        "Select a language model:",
        ["llama3.2", "mistral-small3.2", "magistral", "phi4-mini-reasoning"],
    )

    if st.button("Encode"):
        # Dynamically construct the path to the data folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'data/documents-llm.json')

        rag = RAG(
            data_path=data_path,
            field_to_encode='question',
            index_name=INDEX_NAME,
            encoder=encoder,
            encoder_model=encoder_model,
            llm_model=llm_model,
            es_client=es_client,
            ollama_client=ollama, 
            answer_language='dutch'
        )

        rag = rag.fit()

        # User input
        user_input = st.text_input("Enter your question:")

        if st.button("Ask"):
            print_log(f"User asked: '{user_input}'")
            with st.spinner("Processing..."):

                start_time = time.time()
                answer = rag.predict(query)
                end_time = time.time()
                print_log(f"Answer received in {end_time - start_time:.2f} seconds")
                st.success("Completed!")
                st.write(answer)

                # # Display monitoring information
                # st.write(f"Response time: {answer_data['response_time']:.2f} seconds")
                # st.write(f"Relevance: {answer_data['relevance']}")
                # st.write(f"Model used: {answer_data['model_used']}")
                # st.write(f"Total tokens: {answer_data['total_tokens']}")
                # if answer_data["openai_cost"] > 0:
                #     st.write(f"OpenAI cost: ${answer_data['openai_cost']:.4f}")

                # # Save conversation to database
                # print_log("Saving conversation to database")
                # save_conversation(
                #     st.session_state.conversation_id, user_input, answer_data, course
                # )
                # print_log("Conversation saved successfully")

                # # Feedback buttons
                # col1, col2 = st.columns(2)
                # with col1:
                #     if st.button("+1"):
                #         st.session_state.count += 1
                #         print_log(
                #             f"Positive feedback received. New count: {st.session_state.count}"
                #         )
                #         save_feedback(st.session_state.conversation_id, 1)
                #         print_log("Positive feedback saved to database")
                # with col2:
                #     if st.button("-1"):
                #         st.session_state.count -= 1
                #         print_log(
                #             f"Negative feedback received. New count: {st.session_state.count}"
                #         )
                #         save_feedback(st.session_state.conversation_id, -1)
                #         print_log("Negative feedback saved to database")


if __name__ == "__main__":
    print_log("Course Assistant application started")
    main()