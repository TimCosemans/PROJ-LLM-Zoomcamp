import streamlit as st

# Define a sample function
def process_input(user_input):
    return f"Processed input: {user_input.upper()}"

# Streamlit app
st.title("Streamlit App Example")

# User input
user_input = st.text_input("Search query:")

# Display result
if user_input:
    result = process_input(user_input)
    st.write(result)