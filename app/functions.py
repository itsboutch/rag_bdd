# Import Langchain modules
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.pydantic import BaseModel, Field
from langchain_core.documents import Document
from PIL import Image
import pytesseract
import pdfplumber
from pdf2image import convert_from_path

import os
import tempfile
import uuid
import pandas as pd
import re

def clean_filename(filename):
    """
    Remove "(number)" pattern from a filename 
    (because this could cause error when used as collection name when creating Chroma database).

    Parameters:
        filename (str): The filename to clean

    Returns:
        str: The cleaned filename
    """
    # Regular expression to find "(number)" pattern
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    return new_filename

def is_scanned_pdf(pdf_path: str) -> bool:
    """
    Détecte si un PDF est probablement scanné (pas de texte brut détectable).
    """
    try:
        with open(pdf_path, "rb") as f:
            from PyPDF2 import PdfReader
            reader = PdfReader(f)
            text = ""
            for page in reader.pages[:2]:  # on teste les 2 premières pages
                text += page.extract_text() or ""
            return len(text.strip()) < 100  # peu ou pas de texte = scanné
    except Exception:
        return True

# def get_pdf_text(uploaded_file): 
#     """
#     Load a PDF document from an uploaded file and return it as a list of documents

#     Parameters:
#         uploaded_file (file-like object): The uploaded PDF file to load

#     Returns:
#         list: A list of documents created from the uploaded PDF file
#     """
#     try:
#         # Read file content
#         input_file = uploaded_file.read()

#         # Create a temporary file (PyPDFLoader requires a file path to read the PDF,
#         # it can't work directly with file-like objects or byte streams that we get from Streamlit's uploaded_file)
#         temp_file = tempfile.NamedTemporaryFile(delete=False)
#         temp_file.write(input_file)
#         temp_file.close()

#         # load PDF document
#         loader = PyPDFLoader(temp_file.name)
#         documents = loader.load()

#         return documents
    
#     finally:
#         # Ensure the temporary file is deleted when we're done with it
#         os.unlink(temp_file.name)


def extract_text_with_pypdf(pdf_path: str) -> str:
    """
    Extraction texte classique (PDF numérique).
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return "\n".join([d.page_content for d in docs])

def extract_tables_with_pdfplumber(pdf_path: str) -> str:
    """
    Extraction de texte à partir de tableaux (utile pour les capacités, listes, etc.).
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        text += " | ".join([cell or "" for cell in row]) + "\n"
    except Exception:
        pass
    return text

def extract_text_with_ocr(pdf_path: str) -> str:
    """
    OCR sur PDF scanné.
    """
    try:
        # pages = convert_from_path(pdf_path, dpi=300, poppler_path=r"C:\poppler-24.07.0\Library\bin")
        pages = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        print(f"Erreur conversion OCR : {e}")
        return ""

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    all_text = []
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang="fra+eng")
        all_text.append(text)
    return "\n".join(all_text)

# ---------- CHARGEMENT PDF ----------

def get_pdf_text(uploaded_file):
    """
    Extraction mixte :
    - tente l’extraction textuelle standard ;
    - complète avec OCR si texte trop pauvre ;
    - ajoute les tableaux si présents.
    """
    input_file = uploaded_file.read()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(input_file)
    temp_file.close()

    pdf_path = temp_file.name
    try:
        # 1️⃣ Vérifie si le PDF semble scanné
        scanned = is_scanned_pdf(pdf_path)

        if not scanned:
            print("→ PDF numérique détecté (texte brut exploitable).")
            text = extract_text_with_pypdf(pdf_path)
        else:
            print("→ PDF scanné détecté, OCR en cours...")
            text = extract_text_with_ocr(pdf_path)

        # 2️⃣ Vérifie la richesse du texte
        if len(text.strip()) < 500:
            print("⚠️ Texte partiel, tentative d’enrichissement par OCR mixte...")
            text_ocr = extract_text_with_ocr(pdf_path)
            if len(text_ocr) > len(text):
                text = text_ocr

        # 3️⃣ Ajoute les données tabulaires (capacités, services, etc.)
        tables_text = extract_tables_with_pdfplumber(pdf_path)
        if tables_text:
            text += "\n" + tables_text

        # 4️⃣ Retourne sous forme de documents LangChain
        return [Document(page_content=text)]

    finally:
        os.unlink(pdf_path)

# def split_document(documents, chunk_size, chunk_overlap):    
#     """
#     Function to split generic text into smaller chunks.
#     chunk_size: The desired maximum size of each chunk (default: 400)
#     chunk_overlap: The number of characters to overlap between consecutive chunks (default: 20).

#     Returns:
#         list: A list of smaller text chunks created from the generic text
#     """
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
#                                           chunk_overlap=chunk_overlap,
#                                           length_function=len,
#                                           separators=["\n\n", "\n", " "])
    
#     return text_splitter.split_documents(documents)

def split_document(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=800,
        length_function=len,
        separators=["\n\n", "\n", " "],
    )

    # `documents` is a list[Document] → OK
    return splitter.split_documents(documents)


def get_embedding_function(api_key):
    """
    Return an OpenAIEmbeddings object, which is used to create vector embeddings from text.
    The embeddings model used is "text-embedding-3-small" and the OpenAI API key is provided
    as an argument to the function.

    Parameters:
        api_key (str): The OpenAI API key to use when calling the OpenAI Embeddings API.

    Returns:
        OpenAIEmbeddings: An OpenAIEmbeddings object, which can be used to create vector embeddings from text.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=api_key
    )
    return embeddings


def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):

    """
    Create a vector store from a list of text chunks.

    :param chunks: A list of generic text chunks
    :param embedding_function: A function that takes a string and returns a vector
    :param file_name: The name of the file to associate with the vector store
    :param vector_store_path: The directory to store the vector store

    :return: A Chroma vector store object
    """

    # Create a list of unique ids for each document based on the content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    # Ensure that only unique docs with unique ids are kept
    unique_ids = set()
    unique_chunks = []
    
    unique_chunks = [] 
    for chunk, id in zip(chunks, ids):     
        if id not in unique_ids:       
            unique_ids.add(id)
            unique_chunks.append(chunk)        

    # Create a new Chroma database from the documents
    vectorstore = Chroma.from_documents(documents=unique_chunks, 
                                        collection_name=clean_filename(file_name),
                                        embedding=embedding_function, 
                                        ids=list(unique_ids), 
                                        persist_directory = vector_store_path)

    # The database should save automatically after we create it
    # but we can also force it to save using the persist() method
    vectorstore.persist()
    
    return vectorstore


# def create_vectorstore_from_texts(documents, api_key, file_name):
    
#     # Step 2 split the documents  
#     """
#     Create a vector store from a list of texts.

#     :param documents: A list of generic text documents
#     :param api_key: The OpenAI API key used to create the vector store
#     :param file_name: The name of the file to associate with the vector store

#     :return: A Chroma vector store object
#     """
#     docs = split_document(documents)
    
#     # Step 3 define embedding function
#     embedding_function = get_embedding_function(api_key)

#     # Step 4 create a vector store  
#     vectorstore = create_vectorstore(docs, embedding_function, file_name)
    
#     return vectorstore

def create_vectorstore_from_texts(documents, api_key, file_name):
    docs = split_document(documents)

    embedding_function = get_embedding_function(api_key)

    vector_store = Chroma.from_documents(
        docs,
        embedding_function,
        collection_name=file_name,
    )

    return vector_store


def load_vectorstore(file_name, api_key, vectorstore_path="db"):

    """
    Load a previously saved Chroma vector store from disk.

    :param file_name: The name of the file to load (without the path)
    :param api_key: The OpenAI API key used to create the vector store
    :param vectorstore_path: The path to the directory where the vector store was saved (default: "db")
    
    :return: A Chroma vector store object
    """
    embedding_function = get_embedding_function(api_key)
    return Chroma(persist_directory=vectorstore_path, 
                  embedding_function=embedding_function, 
                  collection_name=clean_filename(file_name))

# Prompt template
# PROMPT_TEMPLATE = """
# You are an assistant for extracting structured hospital information.
# From the provided context, extract the following fields if available. If you don't know the answer, say that you
# don't know. DON'T MAKE UP ANYTHING.

# {context}

# ---

# Answer the question based on the above context: {question}
# """

PROMPT_TEMPLATE = """
You are an assistant for extracting structured information from medical/hospital documents.

Your task is to extract:
1. Hospital-level information
2. Department-level information
3. Staff-level information

Follow these interpretation rules to identify departments and staff:

1. A "department" is any section header that:
   - starts with words like "Direction", "Pôle", "Département", "Coordination", "Service",
   - OR is visually formatted as a title followed by staff members.
2. Each department should be extracted even if no capacity is listed. Use:
   "capacity": null
3. A "staff member" is any line that contains:
   - a name (Mr/Mme/Dr + name)
   - a role (Directeur, Directrice, adjoint, etc.)
   - optional email
   - optional phone
4. Staff members listed under a department belong to that department until the next department header appears.
5. Ignore “Voir la fiche,” URL links, and site navigation items.
6. Ignore repeated email duplicates and treat multi-line entries as a single staff entry.

Extract the following fields if available:

## Hospital Information (hospital)
- hospital_name
- hospital_address
- hospital_phone
- hospital_email
- total capacity (sum if several categories listed)
- hospital_category
- hospital_director
- hospital_city

## Department Information (departments)
For each department:
- service_name (department title)
- capacity (numeric if applicable, else null)

## Staff Information (staff)
For each person:
- staff_name
- staff_surname
- staff_sex (infer from title: M = male, Mme/Mlle = female)
- staff_email
- staff_phone
- staff_role
- deparment_name

---

## Output Format

Return THREE separate tables in a Pandas-compatible structure:

### 1. Hospital table (1 row)
Columns:
- hospital_name
- hospital_address
- hospital_phone
- hospital_email
- hospital_total_capacity
- hospital_category
- hospital_director
- hospital_city

### 2. Departments table (N rows)
Columns:
- service_name
- capacity

### 3. Staff table (N rows)
Columns:
- staff_name
- staff_surname
- staff_sex
- staff_email
- staff_phone
- staff_role
- department_name ( the department the staff belongs to)

Return the final answer as a **single JSON object containing 3 keys**:
- "hospital" → a dict representing a 1-row DataFrame
- "departments" → a list of dicts (rows)
- "staff" → a list of dicts (rows)

This JSON must be strictly compatible with:
pandas.DataFrame.from_records()

If any field is missing, return null.
If a list is empty, return an empty list.

---

Context:
{context}

---
Question: Extract detailed structured hospital, department, and staff information.

"""


# class AnswerWithSources(BaseModel):
#     """An answer to the question, with sources and reasoning."""
#     answer: str = Field(description="Answer to question")
#     sources: str = Field(description="Full direct text chunk from the context used to answer the question")
#     reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")
    

# class ExtractedInfoWithSources(BaseModel):
#     """Extracted information about the research article"""
#     hospital_name: AnswerWithSources
#     hospital_address: AnswerWithSources
#     hospital_phone: AnswerWithSources
#     hospital_email: AnswerWithSources
#     hospital_total_capacity: AnswerWithSources
#     hospital_category: AnswerWithSources
#     hopsital_director: AnswerWithSources
#     service_names: AnswerWithSources
#     service_capacity: AnswerWithSources
#     # hopsital_year: AnswerWithSources
#     # paper_authors: AnswerWithSources


class HospitalInfo(BaseModel):
    hospital_name: str | None = None
    hospital_address: str | None = None
    hospital_phone: str | None = None
    hospital_email: str | None = None
    hospital_total_capacity: str | None = None
    hospital_category: str | None = None
    hospital_director: str | None = None
    hospital_city: str | None = None

class DepartmentInfo(BaseModel):
    service_name: str | None = None
    capacity: str| None = None

    

class StaffInfo(BaseModel):
    staff_name: str | None = None
    staff_surname: str | None = None
    staff_sex: str | None = None
    staff_email: str | None = None
    staff_phone: str | None = None
    staff_role: str | None = None
    depatment_name: str | None = None


class HospitalData(BaseModel):
    hospital: HospitalInfo | None = None
    departments: list[DepartmentInfo] | None = None
    staff: list[StaffInfo] | None = None

def format_docs(docs):
    """
    Format a list of Document objects into a single string.

    :param docs: A list of Document objects

    :return: A string containing the text of all the documents joined by two newlines
    """
    return "\n\n".join(doc.page_content for doc in docs)

# retriever | format_docs passes the question through the retriever, generating Document objects, and then to format_docs to generate strings;
# RunnablePassthrough() passes through the input question unchanged.
# def query_document(vectorstore, query, api_key):

#     """
#     Query a vector store with a question and return a structured response.

#     :param vectorstore: A Chroma vector store object
#     :param query: The question to ask the vector store
#     :param api_key: The OpenAI API key to use when calling the OpenAI Embeddings API

#     :return: A pandas DataFrame with three rows: 'answer', 'source', and 'reasoning'
#     """
#     llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

#     retriever=vectorstore.as_retriever(search_type="similarity")

#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

#     rag_chain = (
#             {"context": retriever | format_docs, "question": RunnablePassthrough()}
#             | prompt_template
#             | llm.with_structured_output(ExtractedInfoWithSources, strict=True)
#         )

#     structured_response = rag_chain.invoke(query)
#     df = pd.DataFrame([structured_response.dict()])

#     # Transforming into a table with two rows: 'answer' and 'source'
#     answer_row = []
#     source_row = []
#     reasoning_row = []

#     for col in df.columns:
#         answer_row.append(df[col][0]['answer'])
#         source_row.append(df[col][0]['sources'])
#         reasoning_row.append(df[col][0]['reasoning'])

#     # Create new dataframe with two rows: 'answer' and 'source'
#     structured_response_df = pd.DataFrame([answer_row, source_row, reasoning_row], columns=df.columns, index=['answer', 'source', 'reasoning'])
  
#     return structured_response_df.T

def query_document(vectorstore, query, api_key):
    """Exécute une requête RAG sur le vectorstore."""
    llm = ChatOpenAI(model="gpt-5", api_key=api_key)
    retriever = vectorstore.as_retriever(search_type="similarity")
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm.with_structured_output(HospitalData, strict=False)
    )

    response = rag_chain.invoke(query)
    df = pd.DataFrame([response.dict()])
    return df