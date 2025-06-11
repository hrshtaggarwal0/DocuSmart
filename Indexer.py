import os
import time
import logging
import pdfplumber
import pickle
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PDF_FOLDER = os.path.abspath("../PDFs") # Monitor this folder
INDEX_SAVE_PATH = "indexed_data.pkl" # Path to save the indexed data
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Header chunking config ---
HEADER_REGEX = r"^\d+(\.\d+)+\s+.+"
MIN_CHUNK_LENGTH = 100      # Minimum characters for a chunk to be indexed

# --- Global variables for document store and retriever ---
document_store = None
retriever = None
indexed_files = {} # To keep track of indexed files and their modification times

def load_or_initialize_index():
    """Loads existing index or initializes a new one."""
    global document_store, retriever, indexed_files
    if os.path.exists(INDEX_SAVE_PATH):
        try:
            with open(INDEX_SAVE_PATH, 'rb') as f:
                saved_data = pickle.load(f)
                document_store = saved_data['document_store']
                retriever = saved_data['retriever']
                indexed_files = saved_data.get('indexed_files', {})
            logging.info(f"Loaded existing index from {INDEX_SAVE_PATH}")
            logging.info(f"Currently indexed files: {list(indexed_files.keys())}")
            # Re-initialize retriever with loaded document store
            retriever = EmbeddingRetriever(
                document_store=document_store,
                embedding_model=EMBEDDING_MODEL
            )
            logging.info("Retriever re-initialized with loaded document store.")
        except Exception as e:
            logging.error(f"Failed to load index: {e}. Initializing new index.")
            document_store = InMemoryDocumentStore(use_bm25=True, embedding_dim=384)
            retriever = EmbeddingRetriever(
                document_store=document_store,
                embedding_model=EMBEDDING_MODEL
            )
            indexed_files = {}
    else:
        logging.info("No existing index found. Initializing new index.")
        document_store = InMemoryDocumentStore(use_bm25=True, embedding_dim=384)
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=EMBEDDING_MODEL
        )
        indexed_files = {}

def save_index():
    """Saves the current document store and retriever to disk."""
    global document_store, retriever, indexed_files
    if document_store and retriever:
        try:
            with open(INDEX_SAVE_PATH, 'wb') as f:
                pickle.dump({
                    'document_store': document_store,
                    'retriever': retriever,
                    'indexed_files': indexed_files
                }, f)
            logging.info(f"Index saved to {INDEX_SAVE_PATH}")
        except Exception as e:
            logging.error(f"Failed to save index: {e}")
    else:
        logging.warning("No document store or retriever to save.")

def chunk_by_headers(text, header_regex=HEADER_REGEX, min_length=MIN_CHUNK_LENGTH):
    """
    Splits text into chunks based on headers matching header_regex.
    Returns a list of Document objects.
    """
    lines = text.splitlines()
    chunks = []
    current_chunk = []
    current_header = None

    for line in lines:
        if re.match(header_regex, line.strip()):
            # Save previous chunk
            if current_chunk:
                chunk_text = "\n".join(current_chunk).strip()
                if len(chunk_text) > min_length:
                    chunks.append((chunk_text, current_header))
            # Start new chunk
            current_header = line.strip()
            current_chunk = [current_header]
        else:
            current_chunk.append(line)
    # Add last chunk
    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if len(chunk_text) > min_length:
            chunks.append((chunk_text, current_header))
    return chunks

def index_pdf(file_path):
    """Indexes a single PDF file using header-based chunking."""
    global document_store, retriever, indexed_files
    filename = os.path.basename(file_path)
    logging.info(f"Indexing new/modified PDF: {filename}")
    docs_to_add = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    header_chunks = chunk_by_headers(text)
                    for chunk_text, header in header_chunks:
                        docs_to_add.append(Document(
                            content=chunk_text,
                            meta={"file": filename, "page": i + 1, "header": header}
                        ))

        if docs_to_add:
            # Remove old documents from this file before adding new ones
            if filename in indexed_files:
                logging.info(f"Removing old chunks for {filename}")
                document_store.delete_documents(filters={"file": [filename]})

            document_store.write_documents(docs_to_add)
            document_store.update_embeddings(retriever)
            indexed_files[filename] = os.path.getmtime(file_path) # Update modification time
            logging.info(f"Successfully indexed {len(docs_to_add)} header-based chunks from {filename}.")
            save_index()
        else:
            logging.warning(f"No text extracted from {filename}. Skipping.")
    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")

def remove_pdf_from_index(file_path):
    """Removes a PDF and its chunks from the index."""
    global document_store, indexed_files
    filename = os.path.basename(file_path)
    if filename in indexed_files:
        logging.info(f"Removing deleted PDF from index: {filename}")
        document_store.delete_documents(filters={"file": [filename]})
        del indexed_files[filename]
        save_index()
        logging.info(f"Successfully removed {filename} from index.")

def initial_indexing_of_folder():
    """Performs initial indexing of all PDFs in the folder."""
    global document_store, indexed_files

    if not os.path.isdir(PDF_FOLDER):
        logging.error(f"PDF folder does not exist: {PDF_FOLDER}")
        return

    existing_pdfs = {f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")}
    indexed_pdf_names = set(indexed_files.keys())

    # Index new or modified PDFs
    for filename in existing_pdfs:
        file_path = os.path.join(PDF_FOLDER, filename)
        current_mtime = os.path.getmtime(file_path)
        if filename not in indexed_pdf_names or indexed_files[filename] < current_mtime:
            index_pdf(file_path)

    # Remove deleted PDFs from index
    for filename in list(indexed_files.keys()): # Iterate over a copy as we might modify indexed_files
        if filename not in existing_pdfs:
            remove_pdf_from_index(os.path.join(PDF_FOLDER, filename))
    
    logging.info("Initial folder scan and indexing complete.")

class PDFEventHandler(FileSystemEventHandler):
    """Handles file system events for PDF folder."""
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            logging.info(f"Detected new file: {event.src_path}")
            index_pdf(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            logging.info(f"Detected modified file: {event.src_path}")
            index_pdf(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            logging.info(f"Detected deleted file: {event.src_path}")
            remove_pdf_from_index(event.src_path)

if __name__ == "__main__":
    load_or_initialize_index()
    initial_indexing_of_folder() # Perform an initial scan on startup

    event_handler = PDFEventHandler()
    observer = Observer()
    observer.schedule(event_handler, PDF_FOLDER, recursive=False) # Only watch the specified folder, not subfolders
    observer.start()
    logging.info(f"Started monitoring PDF folder: {PDF_FOLDER}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    logging.info("PDF indexing script stopped.")