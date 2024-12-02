import json
import os
from pathlib import Path
import PIL.Image
import pandas as pd
import streamlit as st
from datasets import load_from_disk
from openai import OpenAI
import PIL
import base64
from io import BytesIO
from pydantic import BaseModel
from ai_search_demo.qdrant_inexing import IngestClient, SearchClient, pdfs_to_hf_dataset

STORAGE_DIR = "storage"
COLLECTION_INFO_FILENAME = "collection_info.json"
HF_DATASET_DIRNAME = "hf_dataset"
README_FILENAME = "README.md"
SEARCH_TOP_K = 5
COLPALI_TOKEN = "super-secret-token"
VLLM_URL = "https://truskovskiyk--qwen2-vllm-serve.modal.run/v1/"
search_client = SearchClient()
ingest_client = IngestClient()

# Set Streamlit to use wide mode by default
st.set_page_config(layout="wide")



def call_vllm(image_data: PIL.Image.Image):
    model = "Qwen2-VL-7B-Instruct"
    prompt = """
    If the user query is a question, try your best to answer it based on the provided images. 
    If the user query can not be interpreted as a question, or if the answer to the query can not be inferred from the images,
    answer with the exact phrase "I am sorry, I can't find enough relevant information on these pages to answer your question.".
    """

    # Convert PIL image to base64 string after resizing
    buffered = BytesIO()
    # Resize the image to reduce its size
    max_size = (512, 512)
    image_data.thumbnail(max_size)
    image_data.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    client = OpenAI(base_url=VLLM_URL, api_key=COLPALI_TOKEN)
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64_str}"},
                    },
                ],
            }
        ],
    )
    return chat_completion.choices[0].message.content

def ai_search():
    st.header("AI Search")

    # Input form for user query and collection name
    with st.form("search_form"):
        user_query = st.text_input("Enter your search query")
        if os.path.exists(STORAGE_DIR):
            collections = os.listdir(STORAGE_DIR)
            collection_name = st.selectbox("Select a collection", collections)
        else:
            st.error("No collections found.")
            collection_name = None
        search_button = st.form_submit_button("Search")

    if search_button and user_query and collection_name:
        collection_info_path = os.path.join(STORAGE_DIR, collection_name, COLLECTION_INFO_FILENAME)
        
        if os.path.exists(collection_info_path):
            with open(collection_info_path, "r") as json_file:
                collection_info = json.load(json_file)
                search_results = search_client.search_images_by_text(user_query, collection_name=collection_name, top_k=SEARCH_TOP_K)
                if search_results:
                    dataset_path = os.path.join(STORAGE_DIR, collection_name, HF_DATASET_DIRNAME)
                    dataset = load_from_disk(dataset_path)
                    
                    # Collect all search results
                    search_results_data = []
                    for result in search_results.points:
                        payload = result.payload
                        score = result.score
                        image_data = dataset[payload['index']]['image']
                        pdf_name = dataset[payload['index']]['pdf_name']
                        pdf_page = dataset[payload['index']]['pdf_page']
                        search_results_data.append((image_data, score, pdf_name, pdf_page))
                    
                    # Create columns for displaying results
                    col1, col2 = st.columns(2)
                    
                    # Display images in the first column
                    with col1:
                        st.markdown("<h3 style='color:green;'>Relevant Images</h3>", unsafe_allow_html=True)
                        for image_data, score, pdf_name, pdf_page in search_results_data:
                            st.image(image_data, caption=f"Score: {score}, PDF Name: {pdf_name}, Page: {pdf_page}")
                    
                    # Display VLLM output in the second column
                    with col2:
                        st.markdown("<h3 style='color:green;'>Interpretation with LLM</h3>", unsafe_allow_html=True)
                        for image_data, _, _, _ in search_results_data:
                            with st.spinner("Processing with VLLM..."):
                                vllm_output = call_vllm(image_data)
                                
                                st.write(vllm_output)
                                st.write(f"Score: {score}, PDF Name: {pdf_name}, Page: {pdf_page}")
                else:
                    st.write("No results found.")

def create_new_collection():
    st.header("Create PDFs collection")

    # Create a form for uploading PDFs and entering the collection name
    with st.form("upload_form"):
        uploaded_files = st.file_uploader("Choose multiple PDF files", type="pdf", accept_multiple_files=True)
        collection_name = st.text_input("Enter the name of the collection")
        submit_button = st.form_submit_button("Create")

    if submit_button and uploaded_files and collection_name:
        # Create a directory for the collection
        collection_dir = f"{STORAGE_DIR}/{collection_name}"
        os.makedirs(collection_dir, exist_ok=True)

        # Save PDFs to the collection directory
        file_names = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(collection_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_names.append(uploaded_file.name)

        # Create a JSON file with collection details
        collection_info = {
            "name": collection_name,
            "status": "processing",
            "number_of_PDFs": len(uploaded_files),
            "files": file_names
        }
        with open(os.path.join(collection_dir, COLLECTION_INFO_FILENAME), "w") as json_file:
            json.dump(collection_info, json_file)

        st.success(f"Uploaded {len(uploaded_files)} PDFs to collection '{collection_name}'")

        # Function to process and ingest PDFs
        def process_and_ingest():
            try:
                # Transform PDFs to HF dataset
                dataset = pdfs_to_hf_dataset(collection_dir)
                dataset.save_to_disk(os.path.join(collection_dir, HF_DATASET_DIRNAME))

                # Ingest collection with IngestClient
                ingest_client = IngestClient()
                ingest_client.ingest(collection_name, dataset)

                # Update JSON status to 'done'
                collection_info['status'] = 'done'
            except Exception as e:
                # Update JSON status to 'error' in case of any exception
                collection_info['status'] = 'error'
                st.error(f"An error occurred: {e}")
            finally:
                with open(os.path.join(collection_dir, COLLECTION_INFO_FILENAME), "w") as json_file:
                    json.dump(collection_info, json_file)

        # Run the processing and ingestion in the current function with a spinner
        with st.spinner('Processing and ingesting PDFs...'):
            process_and_ingest()

def display_all_collections():
    st.header("Previously Uploaded Collections")

    if os.path.exists(STORAGE_DIR):
        collections = os.listdir(STORAGE_DIR)
        collection_data = []

        for collection in collections:
            collection_info_path = os.path.join(STORAGE_DIR, collection, COLLECTION_INFO_FILENAME)
            if os.path.exists(collection_info_path):
                with open(collection_info_path, "r") as json_file:
                    collection_info = json.load(json_file)
                    collection_data.append(collection_info)

        if collection_data:
            df = pd.DataFrame(collection_data)
            st.table(df)
        else:
            st.write("No collection data found.")
    else:
        st.write("No collections found.")

def about():
    with open(README_FILENAME, "r") as readme_file:
        readme_content = readme_file.read()
    st.markdown(readme_content)

tab1, tab2, tab3, tab4 = st.tabs(["AI Search", "Upload", "Collections", "About"])

with tab1:
    ai_search()
with tab2:
    create_new_collection()
with tab3:
    display_all_collections()
with tab4:
    about()