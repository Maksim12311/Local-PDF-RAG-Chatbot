import os
import tempfile
import streamlit as st

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

MODEL_NAME = "llama3.1"
EMBEDDING_MODEL = "nomic-embed-text"


# ---------------------------------------------------------
# LOAD LLM
# ---------------------------------------------------------

@st.cache_resource
def get_llm():
    return ChatOllama(
        model=MODEL_NAME,
        temperature=0.2
    )


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

@st.cache_resource
def get_embedding():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )


# ---------------------------------------------------------
# BUILD VECTOR DATABASE FROM PDF
# ---------------------------------------------------------

def build_vectorstore_from_pdf(uploaded_file):

    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        pdf_path = temp_file.name

    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            raise ValueError(
                "No documents found in the uploaded PDF."
            )

        # Split PDF into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            add_start_index=True
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError(
                "No text chunks were created from the PDF."
            )

        # Create Chroma vector database
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embedding()
        )

        return vectorstore

    finally:
        # Remove temporary PDF
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


# ---------------------------------------------------------
# CREATE RAG CHAIN
# ---------------------------------------------------------

def make_rag_chain(vectorstore, k: int, answer_style: str):

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k
        }
    )

    def get_context(question: str):

        docs = retriever.invoke(question)

        if not docs:
            return "No relevant context found."

        contexts = []

        for i, document in enumerate(docs):

            page_number = document.metadata.get(
                "page",
                "Unknown"
            )

            context = (
                f"Context {i + 1} "
                f"(PDF page {page_number}):\n"
                f"{document.page_content}"
            )

            contexts.append(context)

        return "\n\n".join(contexts)

    prompt_template = ChatPromptTemplate.from_template(
        """
You are a helpful assistant that answers questions
based ONLY on the provided PDF context.

If the answer is not clearly available in the PDF,
say:

"I don't see this clearly in the document."

Do not invent information.

The user prefers {answer_style} answers.

Use the provided context to answer the question clearly.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = get_llm()

    output_parser = StrOutputParser()

    def ask_question(question: str):

        context = get_context(question)

        chain = (
            prompt_template
            | llm
            | output_parser
        )

        answer = chain.invoke(
            {
                "context": context,
                "question": question,
                "answer_style": answer_style
            }
        )

        return answer

    return ask_question, retriever


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

def init_session_state():

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None


# ---------------------------------------------------------
# MAIN STREAMLIT APPLICATION
# ---------------------------------------------------------

def main():

    st.set_page_config(
        page_title="PDF Q&A with LLaMA 3.1",
        page_icon="📄",
        layout="wide"
    )

    init_session_state()

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.title("📄 PDF Q&A with LLaMA 3.1")

    st.caption(
        "Upload a PDF and ask questions about its content."
    )

    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    with st.sidebar:

        st.header("Settings")

        uploaded_file = st.file_uploader(
            "Upload a PDF file",
            type=["pdf"]
        )

        k = st.slider(
            "Number of context chunks to retrieve (k)",
            min_value=1,
            max_value=10,
            value=3
        )

        answer_style = st.selectbox(
            "Preferred answer style",
            [
                "Concise",
                "Detailed",
                "Step-by-step"
            ]
        )

        st.divider()

        # Process PDF button
        if uploaded_file is not None:

            if st.button(
                "Process PDF",
                type="primary",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        "Reading PDF and creating vector database..."
                    ):

                        st.session_state.vectorstore = (
                            build_vectorstore_from_pdf(
                                uploaded_file
                            )
                        )

                        st.session_state.pdf_name = (
                            uploaded_file.name
                        )

                        st.session_state.chat_history = []

                    st.success(
                        "PDF processed successfully!"
                    )

                except Exception as error:

                    st.error(
                        f"Error processing PDF: {error}"
                    )

        # Clear chat
        if st.button(
            "Clear chat",
            use_container_width=True
        ):

            st.session_state.chat_history = []

            st.rerun()

    # -----------------------------------------------------
    # PDF STATUS
    # -----------------------------------------------------

    if st.session_state.vectorstore is None:

        st.info(
            "👈 Upload a PDF and click 'Process PDF' to begin."
        )

        return

    st.success(
        f"Current PDF: {st.session_state.pdf_name}"
    )

    # -----------------------------------------------------
    # DISPLAY CHAT HISTORY
    # -----------------------------------------------------

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # -----------------------------------------------------
    # USER QUESTION
    # -----------------------------------------------------

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        # Display user message
        with st.chat_message("user"):

            st.markdown(question)

        # Save user message
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        # -------------------------------------------------
        # CREATE RAG
        # -------------------------------------------------

        try:

            ask_question, retriever = make_rag_chain(
                vectorstore=st.session_state.vectorstore,
                k=k,
                answer_style=answer_style
            )

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching the PDF..."
                ):

                    answer = ask_question(
                        question
                    )

                st.markdown(answer)

                # -----------------------------------------
                # SHOW SOURCES
                # -----------------------------------------

                with st.expander(
                    "View retrieved PDF context"
                ):

                    source_documents = (
                        retriever.invoke(question)
                    )

                    for i, document in enumerate(
                        source_documents
                    ):

                        page = (
                            document.metadata.get(
                                "page",
                                "Unknown"
                            )
                        )

                        # PyPDFLoader starts page numbers at 0
                        if isinstance(page, int):
                            display_page = page + 1
                        else:
                            display_page = page

                        st.markdown(
                            f"### Source {i + 1} "
                            f"— Page {display_page}"
                        )

                        st.write(
                            document.page_content
                        )

                        st.divider()

            # Save assistant response
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as error:

            st.error(
                f"Error generating answer: {error}"
            )


# ---------------------------------------------------------
# RUN APP
# ---------------------------------------------------------

if __name__ == "__main__":
    main()