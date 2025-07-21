#saless
import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


genai.configure(api_key="API")  # Replace with your Gemini API key
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Sales Q&A Dashboard", page_icon="📊")
st.title("📊 Sales Q&A Dashboard (Gemini + RAG)")


uploaded_file = st.file_uploader("Upload your sales CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Sales Data")
    st.dataframe(df.head())

    # Convert rows to text
    sales_text = []
    for i, row in df.iterrows():
        row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        sales_text.append(row_text)

    # Create FAISS index
    embeddings = embedding_model.encode(sales_text)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

   
    def answer_sales_query(query, top_k=5):
        query_vector = embedding_model.encode([query])
        distances, indices = index.search(np.array(query_vector, dtype="float32"), top_k)
        retrieved_data = "\n".join([sales_text[i] for i in indices[0]])
        full_prompt = f"""
        You are a helpful sales data analyst. Use the following sales records to answer the question accurately.

        Sales Data:
        {retrieved_data}

        Question: {query}
        Answer:
        """
        response = gemini_model.generate_content(full_prompt)
        return response.text

    
    st.write("### Ask Questions about your Sales Data")
    user_query = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if user_query.strip():
            with st.spinner("Analyzing sales data..."):
                answer = answer_sales_query(user_query)
            st.success(answer)
        else:
            st.warning("Please enter a question.")
