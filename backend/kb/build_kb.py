from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader # <--- NEW IMPORT
import os

# --- 1. Dependency Check (Crucial for HuggingFaceEmbeddings) ---
try:
    import transformers
except ImportError:
    print("🚨 ERROR: The 'transformers' library is required for HuggingFaceEmbeddings.")
    print("Please run: pip install transformers")
    exit()

# --- 2. Resolve paths safely ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
print(f"Reading documents from: {DOCS_DIR}")

# --- 3. Load documents with robust error handling ---
docs_raw_content = []  # Stores raw string content from txt/md files
loaders = []           # Stores LangChain loaders for PDF files

if not os.path.isdir(DOCS_DIR):
    print(f"🚨 ERROR: Document directory not found at {DOCS_DIR}. Please create a 'docs' folder.")
    exit()

for filename in os.listdir(DOCS_DIR):
    path = os.path.join(DOCS_DIR, filename)
    
    if os.path.isdir(path):
        continue 
        
    ext = filename.lower().split('.')[-1]

    if ext in ["txt", "md"]:
        # Handle TXT/MD files using your previous logic (manual reading)
        try:
            with open(path, 'r', encoding='cp1252') as f:
                content = f.read()
                if content.strip():
                    docs_raw_content.append(content)
                else:
                    print(f"⚠️ Warning: Skipping empty file {filename}")
        except Exception as e:
            print(f"🚨 ERROR: Could not read text file {filename} due to: {e}")

    elif ext == "pdf":
        # Handle PDF files using the specialized LangChain loader
        print(f"📄 Adding PDF loader for: {filename}")
        loaders.append(PyPDFLoader(path))
    
    else:
        print(f"⚠️ Warning: Skipping unsupported file type: {filename}")


# Execute all loaders to get PDF documents
pdf_documents = []
for loader in loaders:
    pdf_documents.extend(loader.load())

print(f"Loaded {len(docs_raw_content)} raw text strings and {len(pdf_documents)} Documents from PDFs.")

# --- 4. Validation before proceeding ---
if not docs_raw_content and not pdf_documents:
    print("🛑 Process halted: No readable documents were loaded. Check your 'docs' folder.")
    exit()

# --- 5. Split text ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split the raw string content from text files into LangChain Documents
text_chunks = splitter.create_documents(docs_raw_content)

# Combine the chunks from text files with the documents from PDFs (which will be split later by Chroma if needed)
# NOTE: The PyPDFLoader often returns chunks (one document per page), but we combine here for simplicity.
documents = text_chunks + pdf_documents

# Split any documents that haven't been split yet (if needed, though this is often cleaner)
final_documents = splitter.split_documents(documents)

print(f"Split and combined into {len(final_documents)} final chunks.")


# --- 6. Embeddings ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- 7. Vector store ---
print(f"Starting Chroma DB creation in: {DB_DIR}")

db = Chroma.from_documents(
    final_documents,
    embeddings,
    persist_directory=DB_DIR
)

db.persist()
print(f"✅ Knowledge base successfully built and persisted to '{DB_DIR}'")