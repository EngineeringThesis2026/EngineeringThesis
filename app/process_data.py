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

# Initialize embedding model with error handling
try:
    _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except Exception as e:
    raise RuntimeError(f"Cannot initialize embedding model: {e}") from e

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

    # Check if directory exists
    try:
        if not file_path.exists():
            print(f"WARNING: Data directory does not exist: {file_path}")
            return []

        if not file_path.is_dir():
            print(f"WARNING: Path is not a directory: {file_path}")
            return []
    except Exception as e:
        print(f"WARNING: Error checking data directory: {e}")
        return []

    # Check if any PDF files exist (#1 from task1.md)
    try:
        pdf_files = list(file_path.glob("*.pdf"))
        if not pdf_files:
            print(f"WARNING: No PDF files found in {file_path} directory")
            return []

        print(f"INFO: Found {len(pdf_files)} PDF file(s) to process")
    except Exception as e:
        print(f"WARNING: Error searching for PDF files: {e}")
        return []

    # Load each PDF file
    for pdf_path in pdf_files:
        try:
            print(f"INFO: Loading file: {pdf_path.name}")
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
        except Exception as e:
            print(f"WARNING: Failed to load PDF {pdf_path.name}: {e}")
            continue

    print(f"INFO: Successfully loaded {len(documents)} document page(s)")
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

    # Check if documents list is empty (#2 from task1.md)
    if not documents or len(documents) == 0:
        print("WARNING: No documents to split (documents count = 0)")
        return []

    print(f"INFO: Splitting {len(documents)} document(s) into chunks")

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(documents)

        print(f"INFO: Created {len(all_splits)} chunk(s) from documents")
        return all_splits
    except Exception as e:
        print(f"WARNING: Error splitting documents into chunks: {e}")
        return []


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

    if not sentences or len(sentences) == 0:
        print("WARNING: No sentences to create embeddings for")
        return []

    try:
        print(f"INFO: Creating embeddings for {len(sentences)} sentence(s)")
        texts = [sentence.page_content for sentence in sentences]
        embeddings = embedding_model.encode(texts, show_progress_bar=True)

        embeddings_with_metadata = []
        for emb, sentence in zip(embeddings, sentences):
            embeddings_with_metadata.append({
                "embedding": emb,
                "text": sentence.page_content,
                "metadata": sentence.metadata
            })

        print(f"INFO: Successfully created {len(embeddings_with_metadata)} embedding(s)")
        return embeddings_with_metadata
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to create embeddings: {e}")
        raise RuntimeError(f"Cannot create embeddings: {e}") from e


# LOADING DATA
# data_folder_path = get_data_folder_path()
# all_data_from_pdfs = load_data_from_pdf(file_path=data_folder_path)

# # SPLITTING DATA
# all_splits = split_docks_into_chunks(documents=all_data_from_pdfs)

# # CREATING EMBEDDINGS
# # embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=st.secrets["OPENAI_API_KEY"],)
# embeddings = create_embeddings_with_metadata(sentences=all_splits, embedding_model=_embedding_model)






