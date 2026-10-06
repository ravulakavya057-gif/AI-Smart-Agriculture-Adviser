from pathlib import Path
import re

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "data" / "agriculture_pdfs"
VECTOR_DB_DIR = BASE_DIR / "vector_db"


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

COLLECTION_NAME = "agriculture_knowledge"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 300


# --------------------------------------------------
# AGRICULTURE TOPICS
# --------------------------------------------------

TOPIC_KEYWORDS = {
    "pest": [
        "pest",
        "insect",
        "insects",
        "aphid",
        "thrips",
        "stem borer",
        "leafhopper",
        "bollworm",
        "caterpillar",
        "weevil",
        "midge",
        "mite",
        "borer",
        "nematode",
        "white fly",
        "whitefly",
        "bug",
        "beetle",
        "fly",
    ],

    "disease": [
        "disease",
        "diseases",
        "fungal",
        "bacterial",
        "virus",
        "viral",
        "blight",
        "blast",
        "smut",
        "wilt",
        "rot",
        "leaf spot",
        "rust",
        "discoloration",
    ],

    "cultivation": [
        "cultivation",
        "cultivate",
        "sowing",
        "transplanting",
        "planting",
        "seed",
        "seedling",
        "spacing",
        "harvesting",
        "harvest",
        "yield",
    ],

    "soil_nutrition": [
        "soil",
        "fertilizer",
        "fertiliser",
        "nutrient",
        "nitrogen",
        "phosphorus",
        "potassium",
        "manure",
        "micronutrient",
    ],

    "irrigation": [
        "irrigation",
        "water requirement",
        "water management",
        "drip",
        "sprinkler",
        "irrigate",
    ],

    "weed_management": [
        "weed",
        "weeds",
        "weeding",
        "herbicide",
        "weed management",
    ],

    "harvesting_storage": [
        "harvesting",
        "harvest",
        "storage",
        "stored grain",
        "warehouse",
        "post harvest",
        "post-harvest",
        "grain storage",
    ],

    "variety": [
        "variety",
        "varieties",
        "cultivar",
        "hybrid",
        "parentage",
        "duration",
        "grain type",
    ],
}


# --------------------------------------------------
# CROP KEYWORDS
# --------------------------------------------------

CROP_KEYWORDS = [
    "rice",
    "paddy",
    "wheat",
    "maize",
    "corn",
    "cotton",
    "tomato",
    "groundnut",
    "sorghum",
    "millet",
    "ragi",
    "sugarcane",
    "chilli",
    "chillies",
    "brinjal",
    "okra",
    "banana",
    "mango",
    "vegetable",
    "pulses",
    "soybean",
    "sunflower",
]


# --------------------------------------------------
# CLEAN TEXT
# --------------------------------------------------

def clean_text(text):
    if not text:
        return ""

    # Replace multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# DETECT CROP
# --------------------------------------------------

def detect_crop(text):

    text_lower = text.lower()

    detected = []

    for crop in CROP_KEYWORDS:

        if crop in text_lower:
            detected.append(crop)

    if not detected:
        return "general"

    # Remove duplicates while preserving order
    detected = list(dict.fromkeys(detected))

    return ", ".join(detected[:5])


# --------------------------------------------------
# DETECT TOPIC
# --------------------------------------------------

def detect_topic(text):

    text_lower = text.lower()

    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():

        score = 0

        for keyword in keywords:

            if keyword in text_lower:
                score += 1

        scores[topic] = score

    if not scores:
        return "general"

    best_topic = max(
        scores,
        key=lambda topic: scores[topic]
    )

    if scores[best_topic] == 0:
        return "general"

    return best_topic


# --------------------------------------------------
# CREATE CHUNKS
# --------------------------------------------------

def create_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# --------------------------------------------------
# READ PDF FILES
# --------------------------------------------------

def load_pdfs():

    documents = []
    metadatas = []
    ids = []

    pdf_files = sorted(
        PDF_DIR.glob("*.pdf")
    )

    if not pdf_files:

        print("No PDF files found.")
        print(f"Check this folder: {PDF_DIR}")

        return documents, metadatas, ids

    print(
        f"Found {len(pdf_files)} PDF files."
    )

    for pdf_path in pdf_files:

        print(
            f"\nReading: {pdf_path.name}"
        )

        try:

            reader = PdfReader(
                str(pdf_path)
            )

            print(
                f"Pages: {len(reader.pages)}"
            )

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text()

                if not text:
                    continue

                text = clean_text(text)

                chunks = create_chunks(text)

                for chunk_number, chunk in enumerate(
                    chunks
                ):

                    crop = detect_crop(chunk)

                    topic = detect_topic(chunk)

                    documents.append(chunk)

                    metadatas.append(
                        {
                            "source": pdf_path.name,
                            "page": page_number,
                            "chunk": chunk_number,
                            "crop": crop,
                            "topic": topic,
                        }
                    )

                    ids.append(
                        f"{pdf_path.stem}_"
                        f"page_{page_number}_"
                        f"chunk_{chunk_number}"
                    )

        except Exception as error:

            print(
                f"Error reading "
                f"{pdf_path.name}: {error}"
            )

    return documents, metadatas, ids


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print(
        "AI SMART AGRICULTURE - "
        "KNOWLEDGE INDEXING"
    )
    print("=" * 60)

    print("\nPDF folder:")
    print(PDF_DIR)

    print("\nVector database:")
    print(VECTOR_DB_DIR)

    # --------------------------------------------------
    # LOAD PDF CONTENT
    # --------------------------------------------------

    documents, metadatas, ids = load_pdfs()

    if not documents:

        print(
            "\nNo text was extracted "
            "from the PDFs."
        )

        return

    print("\n" + "=" * 60)

    print(
        f"Total chunks created: "
        f"{len(documents)}"
    )

    print("=" * 60)

    # --------------------------------------------------
    # SHOW TOPIC INFORMATION
    # --------------------------------------------------

    topic_counts = {}

    for metadata in metadatas:

        topic = metadata["topic"]

        topic_counts[topic] = (
            topic_counts.get(topic, 0) + 1
        )

    print("\nDetected topics:")

    for topic, count in sorted(
        topic_counts.items()
    ):

        print(
            f"  {topic}: {count}"
        )

    # --------------------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    print(
        f"Model: {EMBEDDING_MODEL}"
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded."
    )

    # --------------------------------------------------
    # CREATE CHROMADB
    # --------------------------------------------------

    print(
        "\nCreating ChromaDB..."
    )

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    # Delete previous collection
    try:

        client.delete_collection(
            COLLECTION_NAME
        )

        print(
            "Old collection deleted."
        )

    except Exception:

        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description":
                "Agriculture knowledge base "
                "for AI Smart Agriculture"
        }
    )

    # --------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------

    print(
        "\nCreating embeddings..."
    )

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True
    ).tolist()

    # --------------------------------------------------
    # STORE IN CHROMADB
    # --------------------------------------------------

    print(
        "\nStoring knowledge in ChromaDB..."
    )

    batch_size = 100

    for start in range(
        0,
        len(documents),
        batch_size
    ):

        end = start + batch_size

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end]
        )

        print(
            f"Stored "
            f"{min(end, len(documents))}"
            f"/{len(documents)} chunks"
        )

    # --------------------------------------------------
    # FINAL INFORMATION
    # --------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "INDEXING COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)

    print(
        f"PDF files: "
        f"{len(list(PDF_DIR.glob('*.pdf')))}"
    )

    print(
        f"Knowledge chunks: "
        f"{collection.count()}"
    )

    print(
        f"Database location: "
        f"{VECTOR_DB_DIR}"
    )

    print(
        "\nYour agriculture PDFs are now "
        "searchable by the RAG."
    )


if __name__ == "__main__":
    main()