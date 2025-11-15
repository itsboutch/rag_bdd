
# from app.functions import create_vectorstore_from_texts, get_pdf_text
# import streamlit as st  
# from app.functions import *
# import base64

# # Initialize the API key in session state if it doesn't exist
# if 'api_key' not in st.session_state:
#     st.session_state.api_key = ''

# def display_pdf(uploaded_file):

#     """
#     Display a PDF file that has been uploaded to Streamlit.

#     The PDF will be displayed in an iframe, with the width and height set to 700x1000 pixels.

#     Parameters
#     ----------
#     uploaded_file : UploadedFile
#         The uploaded PDF file to display.

#     Returns
#     -------
#     None
#     """
#     # Read file as bytes:
#     bytes_data = uploaded_file.getvalue()
    
#     # Convert to Base64
#     base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    
#     # Embed PDF in HTML
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    
#     # Display file
#     st.markdown(pdf_display, unsafe_allow_html=True)


# def load_streamlit_page():

#     """
#     Load the streamlit page with two columns. The left column contains a text input box for the user to input their OpenAI API key, and a file uploader for the user to upload a PDF document. The right column contains a header and text that greet the user and explain the purpose of the tool.

#     Returns:
#         col1: The left column Streamlit object.
#         col2: The right column Streamlit object.
#         uploaded_file: The uploaded PDF file.
#     """
#     st.set_page_config(layout="wide", page_title="LLM Tool")

#     # Design page layout with 2 columns: File uploader on the left, and other interactions on the right.
#     col1, col2 = st.columns([0.5, 0.5], gap="large")

#     with col1:
#         st.header("Input your OpenAI API key")
#         st.text_input('OpenAI API key', type='password', key='api_key',
#                     label_visibility="collapsed", disabled=False)
#         st.header("Upload file")
#         uploaded_file = st.file_uploader("Please upload your PDF document:", type= "pdf")

#     return col1, col2, uploaded_file


# # Make a streamlit page
# col1, col2, uploaded_file = load_streamlit_page()

# # Process the input
# if uploaded_file is not None:
#     with col2:
#         display_pdf(uploaded_file)
        
#     # Load in the documents
#     documents = get_pdf_text(uploaded_file)
#     st.session_state.vector_store = create_vectorstore_from_texts(documents, 
#                                                                   api_key=st.session_state.api_key,
#                                                                   file_name=uploaded_file.name)
#     st.write("Input Processed")

# # Generate answer
# with col1:
#     if st.button("Generate table"):
#         with st.spinner("Generating answer"):
#             # Load vectorstore:

#             answer = query_document(vectorstore = st.session_state.vector_store, 
#                                     query = "Give me the title, summary, publication date, and authors of the research paper.",
#                                     api_key = st.session_state.api_key)
                            
#             placeholder = st.empty()
#             placeholder = st.write(answer)

import pandas as pd
import uuid
import streamlit as st
import base64
from app.functions import (
    get_pdf_text,
    create_vectorstore_from_texts,
    query_document
)

st.set_page_config(layout="wide", page_title="Hospital Information Extractor")


def display_pdf(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# ---- LAYOUT ----
col1, col2 = st.columns([0.45, 0.55], gap="large")

with col1:
    st.header("🔑 OpenAI API Key")
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    uploaded_file = st.file_uploader("📄 Upload a hospital PDF", type="pdf")

with col2:
    if uploaded_file:
        display_pdf(uploaded_file)

# ---- PROCESS ----
if uploaded_file and api_key:
    with st.spinner("Processing PDF..."):
        documents = get_pdf_text(uploaded_file)
        vector_store = create_vectorstore_from_texts(
            documents, api_key=api_key, file_name=uploaded_file.name
        )
        st.session_state.vector_store = vector_store
        st.success("✅ PDF processed successfully!")

# ---- EXTRACTION ----
if "vector_store" in st.session_state and st.button("📊 Extract Hospital Information"):

    result = query_document(
        vectorstore=st.session_state.vector_store,
        query="Extract hospital information",
        api_key=api_key,
    )

    data = result.iloc[0]

    # --- Build base DataFrames ---
    hospital_df = pd.DataFrame([data["hospital"]])
    departments_df = pd.DataFrame(data["departments"]) if data["departments"] else pd.DataFrame()
    staff_df = pd.DataFrame(data["staff"]) if data["staff"] else pd.DataFrame()

    # --- STEP 1: add hospital_id ---
    hospital_id = str(uuid.uuid4())
    hospital_df["hospital_id"] = hospital_id

    if not departments_df.empty:
        departments_df["hospital_id"] = hospital_id

        # --- STEP 2: add department_id for each department ---
        departments_df["department_id"] = [
            str(uuid.uuid4()) for _ in range(len(departments_df))
        ]

    # --- STEP 3: link staff to departments (if department_name exists) ---
    if not staff_df.empty and "department_name" in staff_df.columns:

        # build mapping: department_name -> department_id
        dept_map = {
            row["service_name"]: row["department_id"]
            for _, row in departments_df.iterrows()
        }

        staff_df["department_id"] = staff_df["department_name"].map(dept_map)

    # --- DISPLAY ---
    st.subheader("🏥 Hospital Information")
    st.dataframe(hospital_df)

    st.subheader("📌 Departments")
    st.dataframe(departments_df)

    st.subheader("👥 Staff")
    st.dataframe(staff_df)

    # --- DOWNLOAD BUTTONS ---
    st.download_button(
        "Download Hospital CSV",
        hospital_df.to_csv(index=False).encode("utf-8"),
        "hospital.csv",
        "text/csv",
    )

    st.download_button(
        "Download Departments CSV",
        departments_df.to_csv(index=False).encode("utf-8"),
        "departments.csv",
        "text/csv",
    )

    st.download_button(
        "Download Staff CSV",
        staff_df.to_csv(index=False).encode("utf-8"),
        "staff.csv",
        "text/csv",
    )


