from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
from pdf2image import convert_from_path
from pypdf import PdfReader
import io






import requests

class ColPaliClient:
    def __init__(self, base_url: str = "https://truskovskiyk--colpali-embedding-serve.modal.run", token: str = "super-secret-token"):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}

    def query_text(self, query_text: str):
        response = requests.post(
            f"{self.base_url}/query",
            headers=self.headers,
            params={"query_text": query_text}
        )
        response.raise_for_status()
        return response.json()

    def process_image(self, image_path: str):
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            response = requests.post(
                f"{self.base_url}/process_image",
                files=files,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()

    def process_pil_image(self, pil_image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        files = {"image": buffered.getvalue()}
        response = requests.post(
            f"{self.base_url}/process_image",
            files=files,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
        

class IngestClient:
    def __init__(self, qdrant_uri: str = "https://qdrant.up.railway.app"):
        self.qdrant_client = QdrantClient(qdrant_uri, port=443, https=True)
        self.colpali_client = ColPaliClient()

    def ingest(self, collection_name, dataset):
        vector_size = 128 

        self.qdrant_client.create_collection(
            collection_name=collection_name,
            on_disk_payload=True,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=100
            ),
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
            ),
        )

        # Use tqdm to create a progress bar
        with tqdm(total=len(dataset), desc="Indexing Progress") as pbar:
            for i in range(len(dataset)):
                # The images are already PIL Image objects, so we can use them directly
                image = dataset[i]["image"]

                # Process and encode image using ColPaliClient
                response = self.colpali_client.process_pil_image(image)
                image_embedding = response['embedding']

                # Prepare point for Qdrant
                point = models.PointStruct(
                    id=i,  # we just use the index as the ID
                    vector=image_embedding,  # This is now a list of vectors
                    payload={
                        "index": dataset[i]['index'],
                        "pdf_name": dataset[i]['pdf_name'],
                        "pdf_page": dataset[i]['pdf_page'],
                    },  # can also add other metadata/data
                )

                # Upload point to Qdrant
                try:
                    self.qdrant_client.upsert(
                        collection_name=collection_name,
                        points=[point],
                        wait=False,
                    )                    
                # clown level error handling here 🤡
                except Exception as e:
                    print(f"Error during upsert: {e}")
                    continue

                # Update the progress bar
                pbar.update(1)

        print("Indexing complete!")           

class SearchClient:
    def __init__(self, qdrant_uri: str = "https://qdrant.up.railway.app/"):
        self.qdrant_client = QdrantClient(qdrant_uri, port=443, https=True)
        self.colpali_client = ColPaliClient()

    def search_images_by_text(self, query_text, collection_name: str, top_k=5):
        # Use ColPaliClient to query text and get the embedding
        query_embedding = self.colpali_client.query_text(query_text)

        # Extract the embedding from the response
        multivector_query = query_embedding['embedding']

        # Search in Qdrant
        search_result = self.qdrant_client.query_points(
            collection_name=collection_name, query=multivector_query, limit=top_k
        )

        return search_result

def get_pdf_images(pdf_path):
    reader = PdfReader(pdf_path)
    page_texts = []
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text = page.extract_text()
        page_texts.append(text)
    # Convert to PIL images
    images = convert_from_path(pdf_path)
    assert len(images) == len(page_texts)
    return images, page_texts

def pdfs_to_hf_dataset(path_to_folder):
    from pathlib import Path
    from datasets import Dataset
    from PIL import Image
    from tqdm import tqdm

    data = []
    global_index = 0

    folder_path = Path(path_to_folder)
    pdf_files = list(folder_path.glob("*.pdf"))
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        images, page_texts = get_pdf_images(str(pdf_file))

        for page_number, (image, text) in enumerate(zip(images, page_texts)):
            data.append({
                "image": image,
                "index": global_index,
                "pdf_name": pdf_file.name,
                "pdf_page": page_number + 1,
                "page_text": text
            })
            global_index += 1

    dataset = Dataset.from_list(data)
    return dataset

if __name__ == "__main__":
    dataset = pdfs_to_hf_dataset('./pdf_collections/test')
    dataset = pdfs_to_hf_dataset('./pdf_collections/smart-hr')

    ingest_client = IngestClient()
    ingest_client.ingest("smart-hr", dataset)

    search_client = SearchClient("smart-hr")
    search_client.search_images_by_text("a cat")