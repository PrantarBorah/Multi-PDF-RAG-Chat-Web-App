import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import pdfplumber

load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_docs): # reads multiple PDFs and combines text form  
    text=""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or "" 
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000) # Each chunk contains 50,000 characters and 1000 adjacent characters for overlap
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()  
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save locally

def get_conversational_chain():  # the conversational chain with ChatGPT
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say "answer is not available in the context."
    Do not provide incorrect or misleading information.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    model = ChatOpenAI(model="gpt-4o", temperature=0.3)  # Uses ChatGPT (GPT-4o)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = OpenAIEmbeddings()  # Load OpenAI embeddings
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)  # Load FAISS index
    docs = new_db.similarity_search(user_question)  # Retrieve relevant documents

    chain = get_conversational_chain()

    # response = chain(
    #     {"input_documents": docs, "question": user_question},
    #     return_only_outputs=True
    # )

    response = chain.run({"input_documents": docs, "question": user_question})

    # with st.empty(): # Streaming the response to make it interactive
    #     for word in response["output_text"].split():
    #         st.write(word + " ", end="")

    # st.write("Reply:", response["output_text"])  # Display response in Streamlit
    with st.empty():  # Streaming the response to make it interactive
        for word in response.split():
            st.write(word + " ", end="")

    st.write("Reply:", response)



def main():
    st.set_page_config("Multi-PDF Chatbot", page_icon="📚")
    st.header("Multi-PDF Chat Agent 🤖 (Powered by OpenAI's ChatGPT)")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded... ✍️")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.image("img/Designer.jpeg")
        st.write("---")
        
        st.title("📁 Upload PDF Files")
        pdf_docs = st.file_uploader("Upload PDF Files & Click Submit & Process", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)  # Extract text
                text_chunks = get_text_chunks(raw_text)  # Chunk text
                get_vector_store(text_chunks)  # Store in FAISS
                st.success("Processing Complete ✅")

        st.write("---")
        st.image("img/DALL·E 2025-02-27 01.14.13 - A Disney-style cartoon avatar of a stylish young man with a neatly trimmed beard, wavy black hair, and glasses. He is wearing a dark turtleneck sweate.jpg")
        st.write("AI App created by Prantar Borah, MS in Computational Data Science, Purdue")

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/PrantarBorah" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()