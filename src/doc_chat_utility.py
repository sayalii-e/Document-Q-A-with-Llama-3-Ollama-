import os

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader


working_dir = os.path.dirname(os.path.abspath(__file__))

llm = Ollama(
    model="gemma:2b",
    temperature = 0
)

embeddings = HuggingFaceEmbeddings()

def get_answer(file_name, query):
    file_path = f"{working_dir}/{file_name}"

    # loading the document
    # loader = UnstructuredFileLoader(file_path)
    # documents = loader.load()
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    # create text chunks
    text_splitter = CharacterTextSplitter(separator="/n",
                                         chunk_size=1000,
                                         chunk_overlap = 200
                                         )
    
    text_chunks = text_splitter.split_documents(pages)

    # vector embeddings from text chunks
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever()
    )
    

    response = qa_chain.invoke({"query": query})

    return response["result"]