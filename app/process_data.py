# Get data path
from pathlib import Path
import streamlit as st

# Read data from pdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Splitting docks into small chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
# from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_data_folder_path() -> Path:
    data_folder = Path(__file__).parent / "data"
    return data_folder


# Load PDF data from 'data' file:
def load_data_from_pdf(file_path: Path) ->list:
    """
    Load all PDF files from the specified directory and return a list of Document objects.
    Each Document contains the content of a page and its metadata (source file name, page number, file path).
    Args:
        file_path (Path): Path to the directory containing PDF files.
    Returns:
        list: A list of Document objects with page content and metadata.
    """
    documents = []

    for pdf_path in file_path.glob("*.pdf"):
        # print(f"Wczytuję plik: {pdf_path}")

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        for i, page in enumerate(pages):
            doc = Document(
                page_content=page.page_content,
                metadata={
                    "source": pdf_path.name,
                    "page_number": i + 1,
                    "file_path": str(pdf_path),
                }
            )
            documents.append(doc)
        
    return documents
    

def split_docks_into_chunks(documents: list,chunk_size: int=1000, chunk_overlap: int=200) ->list:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter
    Args:
        documents (list): A list of Document objects to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.
    Returns:
        list: A list of Document objects representing the split chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)
    return all_splits


# def create_embedding(sentences):

#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#     embeddings = model.encode(sentences)
#     return embeddings

def create_embeddings_with_metadata(sentences, embedding_model):
    """
    Create embeddings for a list of sentences and return them with their metadata.
    Args:
        sentences (list): A list of Document objects containing sentences to be embedded.
        embedding_model: An instance of a sentence transformer model for generating embeddings.
    Returns:
        list: A list of dictionaries, each containing the embedding, text, and metadata.
    """

    texts = [sentence.page_content for sentence in sentences]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    embeddings_with_metadata = []
    for emb, sentence in zip(embeddings, sentences):
        embeddings_with_metadata.append({
            "embedding": emb,
            "text": sentence.page_content,
            "metadata": sentence.metadata
        })
    return embeddings_with_metadata


# LOADING DATA
data_folder_path = get_data_folder_path()
all_data_from_pdfs = load_data_from_pdf(file_path=data_folder_path)
# print(all_data_from_pdfs[0])
# print('==='*80)
# SPLITTING DATA
all_splits = split_docks_into_chunks(documents=all_data_from_pdfs)
# print(all_splits[0])
# CREATING EMBEDDINGS
# print('==='*80)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=st.secrets["OPENAI_API_KEY"],)
embeddings = create_embeddings_with_metadata(sentences=all_splits,embedding_model=EMBEDDING_MODEL)
# print(len(embeddings[0]['embedding']))
# print(embeddings[0])





