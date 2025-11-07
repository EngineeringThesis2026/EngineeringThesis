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

from app.exceptions import EmbeddingModelError, PDFProcessingError, DataImportError
from app.logger import logger

# Global embedding model instance (lazy loaded)
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Get or create the embedding model instance (singleton pattern).

    Returns:
        SentenceTransformer: The embedding model instance.

    Raises:
        EmbeddingModelError: If model fails to load.
    """
    global _embedding_model

    if _embedding_model is None:
        try:
            logger.info("Loading SentenceTransformer model: all-MiniLM-L6-v2")
            _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Successfully loaded embedding model")
        except OSError as e:
            logger.error(f"Failed to download/load embedding model: {e}")
            raise EmbeddingModelError(f"Failed to load embedding model. Check internet connection: {e}")
        except ValueError as e:
            logger.error(f"Invalid model name or configuration: {e}")
            raise EmbeddingModelError(f"Invalid embedding model configuration: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading embedding model: {type(e).__name__}: {e}")
            raise EmbeddingModelError(f"Failed to initialize embedding model: {e}")

    return _embedding_model

def get_data_folder_path() -> Path:
    """
    Get the path to the data folder containing PDF files.

    Returns:
        Path: Path to the data folder.

    Raises:
        DataImportError: If data folder doesn't exist or is not accessible.
    """
    try:
        data_folder = Path(__file__).parent / "data"

        # Check if folder exists
        if not data_folder.exists():
            logger.error(f"Data folder not found at: {data_folder}")
            raise DataImportError(f"Data folder not found at: {data_folder}")

        # Check if it's a directory
        if not data_folder.is_dir():
            logger.error(f"Data path exists but is not a directory: {data_folder}")
            raise DataImportError(f"Data path is not a directory: {data_folder}")

        logger.info(f"Data folder found at: {data_folder}")
        return data_folder

    except PermissionError as e:
        logger.error(f"Permission denied accessing data folder: {e}")
        raise DataImportError(f"No permission to access data folder: {e}")
    except Exception as e:
        if isinstance(e, DataImportError):
            raise
        logger.error(f"Unexpected error accessing data folder: {type(e).__name__}: {e}")
        raise DataImportError(f"Failed to access data folder: {e}")


# Load PDF data from 'data' file:
def load_data_from_pdf(file_path: Path) -> list:
    """
    Load all PDF files from the specified directory and return a list of Document objects.
    Each Document contains the content of a page and its metadata (source file name, page number, file path).
    Corrupted or unreadable PDFs are skipped with a warning.

    Args:
        file_path (Path): Path to the directory containing PDF files.

    Returns:
        list: A list of Document objects with page content and metadata.

    Raises:
        DataImportError: If no valid PDF files are found or loaded.
    """
    documents = []
    failed_files = []
    pdf_files = list(file_path.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"No PDF files found in directory: {file_path}")
        raise DataImportError(f"No PDF files found in directory: {file_path}")

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")

    for pdf_path in pdf_files:
        try:
            logger.info(f"Loading PDF: {pdf_path.name}")
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            if not pages:
                logger.warning(f"PDF has no pages or could not extract content: {pdf_path.name}")
                failed_files.append(pdf_path.name)
                continue

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

            logger.info(f"Successfully loaded {len(pages)} page(s) from: {pdf_path.name}")

        except PermissionError as e:
            logger.warning(f"Permission denied reading PDF {pdf_path.name}: {e}")
            failed_files.append(pdf_path.name)
            continue

        except Exception as e:
            # Catch all PDF-related errors (corrupted, password-protected, etc.)
            logger.warning(f"Failed to load PDF {pdf_path.name}: {type(e).__name__}: {e}")
            failed_files.append(pdf_path.name)
            continue

    # Check if any documents were successfully loaded
    if not documents:
        error_msg = f"No documents could be loaded from any PDF files. Failed files: {', '.join(failed_files)}"
        logger.error(error_msg)
        raise DataImportError(error_msg)

    if failed_files:
        logger.warning(f"Skipped {len(failed_files)} problematic PDF(s): {', '.join(failed_files)}")

    logger.info(f"Successfully loaded {len(documents)} document(s) from {len(pdf_files) - len(failed_files)} PDF file(s)")
    return documents
    

def split_docks_into_chunks(documents: list, chunk_size: int=1000, chunk_overlap: int=200) -> list:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        documents (list): A list of Document objects to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.

    Returns:
        list: A list of Document objects representing the split chunks.

    Raises:
        DataImportError: If document splitting fails.
    """
    try:
        if not documents:
            logger.error("Cannot split empty document list")
            raise DataImportError("No documents provided for splitting")

        logger.info(f"Splitting {len(documents)} document(s) into chunks (size: {chunk_size}, overlap: {chunk_overlap})")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(documents)

        if not all_splits:
            logger.error("Document splitting produced no chunks")
            raise DataImportError("Document splitting failed - no chunks created")

        logger.info(f"Successfully split documents into {len(all_splits)} chunk(s)")
        return all_splits

    except ValueError as e:
        logger.error(f"Invalid parameters for text splitter: {e}")
        raise DataImportError(f"Invalid chunk size or overlap parameters: {e}")

    except AttributeError as e:
        logger.error(f"Invalid document structure for splitting: {e}")
        raise DataImportError(f"Documents have invalid structure: {e}")

    except Exception as e:
        if isinstance(e, DataImportError):
            raise
        logger.error(f"Unexpected error splitting documents: {type(e).__name__}: {e}")
        raise DataImportError(f"Failed to split documents into chunks: {e}")


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

    Raises:
        EmbeddingModelError: If embedding generation fails.
    """
    try:
        if not sentences:
            logger.error("Cannot create embeddings for empty sentence list")
            raise EmbeddingModelError("No sentences provided for embedding")

        logger.info(f"Creating embeddings for {len(sentences)} chunk(s)")

        # Extract texts from Document objects
        texts = [sentence.page_content for sentence in sentences]

        # Generate embeddings
        embeddings = embedding_model.encode(texts, show_progress_bar=True)

        if embeddings is None or len(embeddings) == 0:
            logger.error("Embedding generation produced no results")
            raise EmbeddingModelError("Embedding generation failed - no embeddings created")

        # Combine embeddings with metadata
        embeddings_with_metadata = []
        for emb, sentence in zip(embeddings, sentences):
            embeddings_with_metadata.append({
                "embedding": emb,
                "text": sentence.page_content,
                "metadata": sentence.metadata
            })

        logger.info(f"Successfully created {len(embeddings_with_metadata)} embedding(s)")
        return embeddings_with_metadata

    except MemoryError as e:
        logger.error(f"Out of memory while creating embeddings: {e}")
        raise EmbeddingModelError(f"Out of memory - try processing fewer documents or use smaller chunks: {e}")

    except RuntimeError as e:
        logger.error(f"Runtime error during embedding generation: {e}")
        raise EmbeddingModelError(f"Embedding model runtime error: {e}")

    except (AttributeError, KeyError) as e:
        logger.error(f"Invalid document structure for embedding: {e}")
        raise EmbeddingModelError(f"Documents have invalid structure for embedding: {e}")

    except Exception as e:
        if isinstance(e, EmbeddingModelError):
            raise
        logger.error(f"Unexpected error creating embeddings: {type(e).__name__}: {e}")
        raise EmbeddingModelError(f"Failed to create embeddings: {e}")


# LOADING DATA
# data_folder_path = get_data_folder_path()
# all_data_from_pdfs = load_data_from_pdf(file_path=data_folder_path)

# # SPLITTING DATA
# all_splits = split_docks_into_chunks(documents=all_data_from_pdfs)

# # CREATING EMBEDDINGS
# # embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=st.secrets["OPENAI_API_KEY"],)
# embeddings = create_embeddings_with_metadata(sentences=all_splits, embedding_model=_embedding_model)






