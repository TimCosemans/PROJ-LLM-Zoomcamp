import streamlit as st
import time
import uuid
import os 

from ollama import Client
import sys
sys.path.append('./src')  # Adjust the path to your src directory

from src.rag import RAG
from src.elasticsearch_utils import client, get_recent_docs
from src.evaluate import relevance, save_results

DOCS_INDEX_NAME = "llm-doc"
RESULTS_INDEX_NAME = "app-results"

N_CONTEXT_DOCS = 10
FIELD_TO_ENCODE = 'question'

def main():
    st.title("Multilingual Course Assistant Tester")

    if 'conversation_id' not in st.session_state:
        # Initialize conversation ID if not already set
        st.session_state.conversation_id = str(uuid.uuid4())

    variables_to_initialize = [
        "encoded",
        "answered",
        "rag", 
        "feedback"
    ]

    # Session state initialization
    for var in variables_to_initialize:
        if var not in st.session_state:
            st.session_state[var] = None

    # Model selection
    st.session_state.encoder = st.selectbox("Select an encoder:", ["ollama", "huggingface"])
    if st.session_state.encoder == "ollama":
        st.session_state.encoder_model = st.selectbox(
            "Select a model:",
            ["llama3.2", "mistral-small3.2"],
        )
    else:   
        # For Hugging Face, we can use a fixed model or allow user selection
        st.session_state.encoder_model = st.selectbox(
            "Select a model:",
            ["intfloat/multilingual-e5-small", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
        )
    
    st.session_state.llm_model = st.selectbox(
        "Select a language model:",
        ["llama3.2", "mistral-small3.2"],
    )

    st.session_state.language = st.selectbox(
        "Select the answer language:",
        ["english", "dutch", "french"],
    )

    if st.button("Encode"):
        # Dynamically construct the path to the data folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        st.session_state.data_path = os.path.join(script_dir, 'data/documents-llm.json')

        st.session_state.es_client = client(host="host.docker.internal:9200")
        st.session_state.ollama = Client(host='host.docker.internal:11434')


        print('Encoding...')

        rag = RAG(
            data_path=st.session_state.data_path,
            field_to_encode=FIELD_TO_ENCODE,
            index_name=DOCS_INDEX_NAME,
            encoder=st.session_state.encoder,
            encoder_model=st.session_state.encoder_model,
            n_context_docs=N_CONTEXT_DOCS,
            llm_model=st.session_state.llm_model,
            es_client=st.session_state.es_client,
            ollama_client=st.session_state.ollama, 
            answer_language=st.session_state.language
        )

        st.session_state.rag = rag.fit()

        st.session_state.encoded = True

        st.success("Encoding completed successfully!")
        print("Encoding completed successfully!")

    if st.session_state.encoded:
        # User input
        st.session_state.user_input = st.text_input("Enter your question:")

        if st.button("Ask"):
            print('Processing user input...')

            start_time = time.time()
            answer = st.session_state.rag.predict(st.session_state.user_input)
            end_time = time.time()

            relevance_score, explanation = relevance(st.session_state.user_input, answer, st.session_state.ollama, st.session_state.llm_model)

            st.session_state.results = {
                "rag_id": st.session_state.rag.id,
                "conversation_id": st.session_state.conversation_id,
                "encoder": st.session_state.encoder,
                "encoder_model": st.session_state.encoder_model,
                "n_context_docs": N_CONTEXT_DOCS,
                "llm_model": st.session_state.llm_model,
                "user_input": st.session_state.user_input,
                "answer": answer,
                "answer_language": st.session_state.language,
                "answer_time": int(end_time - start_time),  
                "relevance": relevance_score,
                "explanation": explanation
            }

            st.session_state.answered = True

            st.success("Completed!")
            print("Processing completed!")
            
            st.write(answer)

        if st.session_state.answered:
            st.write('Please provide feedback on the answer to save')
            # Feedback buttons
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("+1"):
                    st.session_state.results["user_feedback"] = 1
                    save_results(st.session_state.es_client, st.session_state.results, index_name=RESULTS_INDEX_NAME)
                    st.success("Feedback saved: +1")
                    print("Feedback saved: +1")

                    st.session_state.feedback = True

            with col2:
                if st.button("-1"):
                    st.session_state.results["user_feedback"] = -1
                    save_results(st.session_state.es_client, st.session_state.results, index_name=RESULTS_INDEX_NAME)
                    st.success("Feedback saved: -1")
                    print("Feedback saved: -1")

                    st.session_state.feedback = True
            
            if st.session_state.feedback:
                # Display recent conversations
                st.subheader("Recent Conversations")
                recent_conversations = get_recent_docs(st.session_state.es_client, RESULTS_INDEX_NAME, st.session_state.conversation_id, size=5)
                for conv in recent_conversations:
                    st.write(f"Q: {conv['user_input']}")
                    st.write(f"A: {conv['answer']}")
                    st.divider()

if __name__ == "__main__":
    main()
