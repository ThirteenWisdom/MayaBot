import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 2. RAG SYSTEM (The Engine)
@st.cache_resource
def build_rag_system():
    # --- CONFIGURATION ---
    pdf_path = "ALMG2010.pdf" 
    
    # 1. DEFINE EMBEDDINGS HERE (This fixes the NameError)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    if not os.path.exists(pdf_path):
        return None, f"Error: Could not find '{pdf_path}' in the folder."

    # A. Load
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # B. Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # C. Create Vectorstore with the "Safety Net"
    try:
        # Now 'embeddings' is defined right above, so this will work
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    except Exception as e:
        if "429" in str(e):
            st.warning("Google is overwhelmed. Waiting 20 seconds to try again...")
            time.sleep(20)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        else:
            raise e

    # D. Retrieve
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    return retriever, "Ready"

# 3. INITIALIZE
with st.spinner("Reading the Dictionary (This takes 10-20 seconds once)..."):
    retriever, status = build_rag_system()

if not retriever:
    st.error(status)
    st.stop()
else:
    st.success("Ready")

# 4. CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input('Ex: What does "Ixim" mean?'):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5. THE STRICT PROMPT
    template = """
    You are a K'iche' language expert. 
    Answer based STRICTLY on the provided dictionary context.
    If the word has multiple meanings, list them all.
    
    STRICT RULES:
    1. Respond ONLY in English. 
    2. Do NOT use Spanish in your explanation.
    3. The only non-English words allowed are the K'iche' words from the dictionary.
    4. Provide the definition based STRICTLY on the provided dictionary context.
    5. Don't mention the dictionary
    6. The word/translation should be surrouneded by quotation marks "".

    Context: {context}
    
    Question: {question}
    Answer:
    """
    QA_PROMPT = PromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                response = qa_chain.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")