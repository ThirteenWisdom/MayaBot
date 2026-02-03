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

# Source name mapping
SOURCE_NAMES = {
    "ALMG2010.pdf": "Vocabulario K'iche', ALMG (2010)",
    "TedlockPopolVuhKindleWithPageNumber.pdf": "Popol Vuh, Tedlock"
}

# Document type mapping
DOC_TYPES = {
    "ALMG2010.pdf": "dictionary",
    "TedlockPopolVuhKindleWithPageNumber.pdf": "story"
}

# 2. RAG SYSTEM (The Engine)
@st.cache_resource(show_spinner=False)
def build_rag_system():
    # --- CONFIGURATION ---
    references_folder = "references"

    # 1. DEFINE EMBEDDINGS HERE (This fixes the NameError)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    if not os.path.exists(references_folder):
        return None, None, f"Error: Could not find '{references_folder}' folder."

    # A. Load all PDFs from references folder
    all_docs = []
    pdf_files = [f for f in os.listdir(references_folder) if f.endswith('.pdf')]

    if not pdf_files:
        return None, None, "Error: No PDF files found in references folder."

    for pdf_file in pdf_files:
        pdf_path = os.path.join(references_folder, pdf_file)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = pdf_file
            doc.metadata["doc_type"] = DOC_TYPES.get(pdf_file, "unknown")
        all_docs.extend(docs)

    docs = all_docs

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

    return vectorstore, embeddings, "Ready"

def classify_query(llm, query, language):
    """Classify query as 'dictionary' or 'story'"""
    classify_prompt = f"""Classify this question about K'iche' Maya into ONE category:
- "dictionary": Questions about word meanings, translations, vocabulary, definitions, how to say something
- "story": Questions about Popol Vuh, characters, events, mythology, narrative, plot

Question: {query}

Answer with ONLY one word: dictionary or story"""

    response = llm.invoke(classify_prompt)
    result = response.content.strip().lower()

    if "dictionary" in result:
        return "dictionary"
    elif "story" in result:
        return "story"
    else:
        return "dictionary"  # default to dictionary

# CUSTOM STYLING (must be before spinner)
st.markdown("""
<style>
/* Mayan blue spinner */
.stSpinner > div,
.stSpinner > div > div,
[data-testid="stSpinner"] > div,
[data-testid="stSpinner"] > div > div,
.stSpinner svg circle,
[data-testid="stSpinner"] svg circle {
    border-top-color: #4A8BAD !important;
    stroke: #4A8BAD !important;
}
[data-testid="stSpinner"] svg {
    color: #4A8BAD !important;
}
</style>
""", unsafe_allow_html=True)

# 3. INITIALIZE
with st.spinner("Loading..."):
    vectorstore, embeddings, status = build_rag_system()

if not vectorstore:
    st.error(status)
    st.stop()

# TITLE
st.markdown("""
<h1 style="text-align: center; font-family: 'Georgia', serif; font-weight: 300;
letter-spacing: 0.15em; color: #2c3e50; margin-bottom: 1em;">
Mayib'ot
</h1>
""", unsafe_allow_html=True)

# LANGUAGE SELECTOR
language = st.selectbox(
    "Select your language / Seleccione su idioma / 言語を選択してください",
    ["English", "Español", "日本語"],
    index=0
)

# 4. CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Localized placeholders
placeholders = {
    "English": 'Ex: What does "Ixim" mean? / Who are the Hero Twins?',
    "Español": 'Ej: ¿Qué significa "Ixim"? / ¿Quiénes son los Héroes Gemelos?',
    "日本語": '例：「Ixim」の意味は？ / 双子の英雄は誰？'
}
placeholder = placeholders.get(language, placeholders["English"])

if prompt := st.chat_input(placeholder):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5. THE STRICT PROMPT
    definition_labels = {"English": "Definition", "Español": "Definición", "日本語": "定義"}
    example_labels = {"English": "Example", "Español": "Ejemplo", "日本語": "例文"}
    definition_label = definition_labels.get(language, "Definition")
    example_label = example_labels.get(language, "Example")

    if language == "日本語":
        template = f"""
    あなたはキチェ語とポポル・ヴフの専門家です。
    提供されたコンテキストに基づいて厳密に回答してください。

    ルール:
    1. 日本語でのみ回答してください。
    2. キチェ語の単語のみ日本語以外で使用できます。
    3. 提供されたコンテキストに厳密に基づいて回答してください。
    4. 「辞書」や「文書」には言及しないでください。

    質問のタイプ:

    A) 語彙・単語の質問（例：「〇〇の意味は？」「〇〇はキチェ語で何？」）:
       - コンテキストから定義を提供してください。
       - キチェ語の単語は「」で囲んでください。
       - キチェ語を使った例文を作成してください。
       - フォーマット:
         **{definition_label}:** [定義]
         **{example_label}:** [キチェ語の文] — [日本語訳]

    B) 物語・ポポル・ヴフの質問（例：「双子の英雄は誰？」「〇〇で何が起きた？」）:
       - 物語、登場人物、出来事について回答してください。
       - 分かりやすく興味深い回答を心がけてください。

    コンテキスト: {{context}}

    質問: {{question}}
    回答:
    """
    else:
        template = f"""
    You are an expert on K'iche' Maya language and the Popol Vuh.
    Answer based STRICTLY on the provided context.

    STRICT RULES:
    1. Respond ONLY in {language}.
    2. The only non-{language} words allowed are K'iche' words.
    3. Answer based STRICTLY on the provided context.
    4. Don't mention "the dictionary" or "the document".

    QUESTION TYPES:

    A) For WORD/VOCABULARY questions (e.g., "What does X mean?", "How do you say X?"):
       - Provide the definition from the context.
       - K'iche' words should be in quotation marks "".
       - Create an example sentence using the K'iche' word.
       - FORMAT:
         **{definition_label}:** [definition]
         **{example_label}:** [K'iche' sentence] — [translation in {language}]

    B) For STORY/POPOL VUH questions (e.g., "Who are the Hero Twins?", "What happened to...?"):
       - Answer the question about the story, characters, or events.
       - Be informative and engaging.
       - You may use K'iche' names and terms from the text.

    Context: {{context}}

    Question: {{question}}
    Answer:
    """
    QA_PROMPT = PromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            # Classify the query to route to appropriate document
            query_type = classify_query(llm, prompt, language)

            # Create filtered retriever based on query type
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 15,
                    "filter": {"doc_type": query_type}
                }
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type_kwargs={"prompt": QA_PROMPT},
                return_source_documents=True
            )
            try:
                result = qa_chain({"query": prompt})
                response = result["result"]
                source_docs = result["source_documents"]

                # Group pages by source file
                from collections import defaultdict
                sources_pages = defaultdict(set)
                for doc in source_docs:
                    source_file = doc.metadata.get("source_file", "unknown")
                    page = doc.metadata.get("page", 0) + 1
                    sources_pages[source_file].add(page)

                st.markdown(response)

                # Translate source label
                source_labels = {"English": "Source", "Español": "Fuente", "日本語": "出典"}
                page_labels = {"English": "Page ", "Español": "Página ", "日本語": ""}
                source_label = source_labels.get(language, "Source")
                page_label = page_labels.get(language, "Page ")

                # Build citation string for each source
                citations = []
                for source_file, pages in sources_pages.items():
                    source_name = SOURCE_NAMES.get(source_file, source_file)
                    page_str = ", ".join(str(p) for p in sorted(pages))
                    citations.append(f"{source_name} ({page_label}{page_str})")

                citation_text = " | ".join(citations)
                st.caption(f"📖 {source_label}: {citation_text}")

                # Store response with page info for chat history
                full_response = f"{response}\n\n*{source_label}: {citation_text}*"
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"An error occurred: {e}")