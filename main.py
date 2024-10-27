import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

import tempfile
import os

# Streamlit app
st.title("Chat with Your PDF using Ollama")

# Sidebar for customization
st.sidebar.header("Customization")
chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=1000, value=500, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=100, value=0, step=10)
ollama_model = st.sidebar.selectbox("Ollama Model", ["llama3.2:1b", "llama3.2:7b", "llama3.2:13b"])

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_file is not None:
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load and process the PDF
        loader = PyPDFLoader(tmp_file_path)
        data = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = text_splitter.split_documents(data)

        # Create and store the vectorstore
        st.session_state.vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

        # Remove temporary file
        os.unlink(tmp_file_path)

        st.success("File processed successfully!")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")

# Query input
query = st.text_input("Ask a question about your document:")
submit_button = st.button("Submit")

if submit_button and query and st.session_state.vectorstore:
    # Prompt
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    class StreamHandler(StreamingStdOutCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text = initial_text

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text += token
            self.container.markdown(self.text)

    stream_handler = StreamHandler(st.empty())
    llm = Ollama(model=ollama_model, callback_manager=CallbackManager([stream_handler]))
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=st.session_state.vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    try:
        with st.spinner("Generating response..."):
            result = qa_chain({"query": query})
    except Exception as e:
        st.error(f"An error occurred while generating the response: {str(e)}")

elif submit_button and not st.session_state.vectorstore:
    st.warning("Please upload a PDF file first.")
