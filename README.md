
# 📚 Multi-PDF RAG Chat Web App
**🚀 AI-Powered Document Q&A | OpenAI GPT-4o | FAISS | LangChain | Streamlit**

![Project Demo](Designer.jpeg)  

## 🔹 Overview
This is an **end-to-end Multi-PDF Retrieval-Augmented Generation (RAG) Chatbot** that allows users to **upload multiple PDFs** and ask **context-aware questions** using **OpenAI’s GPT-4o**. The system retrieves relevant document sections using **FAISS vector search** and generates detailed responses with **LangChain’s conversational AI**.  

### **Key Features**
- 📁 **Multi-PDF Support**: Upload multiple PDF files for document-based Q&A.
- 🧠 **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with GPT-4o for accurate answers.
- 🔍 **FAISS Vector Search**: Enables fast and efficient text retrieval.
- 💬 **Interactive Chat Interface**: Built with **Streamlit** for real-time Q&A.
- ⚡ **End-to-End Implementation**: Developed from scratch for intelligent document comprehension.

---

## 🛠️ Tech Stack & Usage

| Technology        | Usage                                              |
|------------------|--------------------------------------------------|
| **Python**      | Core programming language                        |
| **OpenAI GPT-4o** | LLM for generating answers                      |
| **FAISS**       | Vector database for efficient text retrieval      |
| **LangChain**   | Retrieval-Augmented Generation (RAG) pipeline    |
| **Streamlit**   | Web-based interactive chat interface              |
| **pdfplumber**  | Extracting text from PDFs                         |
| **dotenv**      | Secure API key management                         |

---

## 🚀 How It Works
### **1️⃣ PDF Upload & Processing**
- Extracts text from PDFs using `pdfplumber`.  
- Splits text into chunks for efficient retrieval (`RecursiveCharacterTextSplitter`).  
- Converts text into **FAISS embeddings** for similarity search.  

### **2️⃣ Question Processing & Retrieval**
- Users ask questions in the **Streamlit UI**.  
- FAISS performs **similarity search** to find relevant text chunks.  
- LangChain **retrieves context** and generates an AI-powered response using GPT-4o.  

### **3️⃣ Conversational AI & Streaming Responses**
- The app **remembers chat history** with `ConversationBufferMemory`.  
- Answers are **generated dynamically and streamed** in real time.  

---

## 📌 Setup & Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/multi-pdf-rag-chatbot.git
cd multi-pdf-rag-chatbot
