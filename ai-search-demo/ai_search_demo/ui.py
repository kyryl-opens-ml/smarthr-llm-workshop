import streamlit as st
import os
import json
import pandas as pd
import threading
from ai_search_demo.qdrant_inexing import IngestClient, pdfs_to_hf_dataset
from ai_search_demo.qdrant_inexing import SearchClient
from datasets import load_from_disk

STORAGE_DIR = "storage"
COLLECTION_INFO_FILENAME = "collection_info.json"
HF_DATASET_DIRNAME = "hf_dataset"
README_FILENAME = "README.md"
SEARCH_TOP_K = 5

search_client = SearchClient()
ingest_client = IngestClient()

    
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
                st.write(f"Searching in collection: {collection_info['name']}")
                # Here you would implement the actual search logic
                # For now, we just display a placeholder message
                st.write(f"Results for query '{user_query}' in collection '{collection_name}':")
                search_results = search_client.search_images_by_text(user_query, collection_name=collection_name, top_k=SEARCH_TOP_K)
                if search_results:
                    dataset_path = os.path.join(STORAGE_DIR, collection_name, HF_DATASET_DIRNAME)
                    dataset = load_from_disk(dataset_path)
                    for result in search_results.points:
                        payload = result.payload
                        score = result.score
                        image_data = dataset[payload['index']]['image']
                        pdf_name = dataset[payload['index']]['pdf_name']
                        pdf_page = dataset[payload['index']]['pdf_page']
                        # Display the extracted information in the UI
                        st.image(image_data, caption=f"Score: {score}, PDF Name: {pdf_name}, Page: {pdf_page}")
                else:
                    st.write("No results found.")

def create_new_collection():
    st.header("Upload PDFs")

    # Create a form for uploading PDFs and entering the collection name
    with st.form("upload_form"):
        uploaded_files = st.file_uploader("Choose multiple PDF files", type="pdf", accept_multiple_files=True)
        collection_name = st.text_input("Enter the name of the collection")
        submit_button = st.form_submit_button("Upload")

    if submit_button and uploaded_files and collection_name:
        # Create a directory for the collection
        collection_dir = f"{STORAGE_DIR}/{collection_name}"
        os.makedirs(collection_dir, exist_ok=True)

        # Save PDFs to the collection directory
        for uploaded_file in uploaded_files:
            with open(os.path.join(collection_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Create a JSON file with collection details
        collection_info = {
            "name": collection_name,
            "status": "processing",
            "number_of_PDFs": len(uploaded_files)
        }
        with open(os.path.join(collection_dir, COLLECTION_INFO_FILENAME), "w") as json_file:
            json.dump(collection_info, json_file)

        st.success(f"Uploaded {len(uploaded_files)} PDFs to collection '{collection_name}'")

        # Function to process and ingest PDFs
        def process_and_ingest():
            # Transform PDFs to HF dataset
            dataset = pdfs_to_hf_dataset(collection_dir)
            dataset.save_to_disk(os.path.join(collection_dir, HF_DATASET_DIRNAME))

            # Ingest collection with IngestClient
            ingest_client = IngestClient()
            ingest_client.ingest(collection_name, dataset)

            # Update JSON status to 'done'
            collection_info['status'] = 'done'
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