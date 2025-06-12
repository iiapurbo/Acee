import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import warnings

# Suppress huggingface_hub FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# Configuration
DEFAULT_DIRECTORY_PATH = "./markdown_files"  # Directory containing .md files
DEFAULT_PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/LaBSE"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize embeddings
# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/LaBSE", 
    model_kwargs={"device": "cpu"}  
    )

def load_and_process_single_md(file_path):
    """
    Load and process a single Markdown file into document chunks.
    
    Args:
        file_path (str): Path to the Markdown file
        
    Returns:
        list: List of document chunks with metadata
    """
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        # Split documents into chunks with metadata
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        
        # Add metadata to each chunk
        for i, split in enumerate(splits):
            split.metadata = {
                "source": file_path,
                "chunk_index": i,
                "filename": os.path.basename(file_path)
            }
        return splits
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return []

def load_and_process_directory(directory_path=DEFAULT_DIRECTORY_PATH):
    """
    Load and process all Markdown files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing Markdown files
        
    Returns:
        list: Combined list of document chunks from all files
    """
    if not os.path.exists(directory_path):
        raise Exception(f"Directory {directory_path} not found. Please create the directory and add Markdown files.")
    
    all_splits = []
    md_files = [f for f in os.listdir(directory_path) if f.endswith('.md')]
    
    if not md_files:
        raise Exception(f"No Markdown files found in {directory_path}. Please add some .md files.")
    
    print(f"Found {len(md_files)} Markdown files in {directory_path}")
    
    for md_file in md_files:
        file_path = os.path.join(directory_path, md_file)
        print(f"Processing {md_file}...")
        file_splits = load_and_process_single_md(file_path)
        all_splits.extend(file_splits)
        print(f"  - Added {len(file_splits)} chunks from {md_file}")
    
    print(f"Total chunks processed: {len(all_splits)}")
    return all_splits

def initialize_vectorstore(splits, persist_directory=DEFAULT_PERSIST_DIRECTORY):
    """
    Initialize and return a Chroma vector store with the given document splits.
    
    Args:
        splits (list): List of document chunks
        persist_directory (str): Directory to persist the vector store
        
    Returns:
        Chroma: Initialized vector store
    """
    try:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        vectorstore.persist()  # Explicitly persist the database
        return vectorstore
    except Exception as e:
        raise Exception(f"Error creating vector store: {str(e)}")

def get_vectorstore(directory_path=DEFAULT_DIRECTORY_PATH, persist_directory=DEFAULT_PERSIST_DIRECTORY):
    """
    Process all Markdown files in a directory and return a vector store.
    If the vector store already exists in the persist directory, it will be loaded
    unless force_recreate is True.
    
    Args:
        directory_path (str): Path to the directory containing Markdown files
        persist_directory (str): Directory to persist the vector store
        force_recreate (bool): Whether to force recreate the vector store even if it exists
        
    Returns:
        Chroma: Vector store instance
    """
    if not os.path.exists(directory_path):
        raise Exception(f"Directory {directory_path} not found. Please create the directory and add Markdown files.")
    
    if os.path.exists(persist_directory):
        print(f"Loading existing vector store from {persist_directory}")
        return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    else:
            print(f"No vector stored in {persist_directory}")
        #else:
       #     print(f"Creating new vector store in {persist_directory}")
        
        # Create new vector store
       # splits = load_and_process_directory(directory_path)
       # return initialize_vectorstore(splits, persist_directory)

def get_retriever(vectorstore, k=3):
    """
    Get a retriever from the vector store with specified number of results.
    
    Args:
        vectorstore (Chroma): The vector store instance
        k (int): Number of results to retrieve
        
    Returns:
        Retriever: A retriever instance
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})

# Example usage
if __name__ == "__main__":
    splits = load_and_process_directory()
    
    # Create the vector store using those splits
    vectorstore = initialize_vectorstore(splits)
    
    # Create a retriever
    retriever = get_retriever(vectorstore)
    
    print("Vector store and retriever created successfully!")
    print(f"Vector store contains {vectorstore._collection.count()} documents")