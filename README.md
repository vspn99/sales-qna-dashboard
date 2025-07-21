# 📊 Sales Q&A Dashboard (Gemini + RAG)

An AI-powered sales analytics assistant that answers natural language questions from CSV sales data using **Google Gemini** and **RAG (Retrieval Augmented Generation)**.

## 🚀 Features
- Upload any sales CSV file
- Ask natural language questions like:
  - "Which product sold the most?"
  - "What is the total revenue from laptops?"
- Uses **Google Gemini** for analysis

## 🛠️ Tech Stack
- Streamlit (dashboard UI)
- Google Gemini (LLM)
- FAISS + Sentence Transformers (RAG for data retrieval)
- Pandas (CSV processing)

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run sales_dashboard.py
