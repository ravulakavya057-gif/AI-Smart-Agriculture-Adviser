from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "vector_db"

COLLECTION_NAME = "agriculture_knowledge"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

from functools import lru_cache


@lru_cache(maxsize=1)
def get_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )

# --------------------------------------------------
# QUERY KEYWORDS
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
    "soybean",
    "sunflower",
]


TOPIC_KEYWORDS = {
    "pest": [
        "pest",
        "pests",
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
        "whitefly",
        "white fly",
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
        "rust",
        "leaf spot",
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
    ],
}


# --------------------------------------------------
# TEXT HELPERS
# --------------------------------------------------

def find_keywords(text, keyword_list):

    text = text.lower()

    found = []

    for keyword in keyword_list:

        if keyword in text:
            found.append(keyword)

    return found


def detect_crop(query):

    found = find_keywords(
        query,
        CROP_KEYWORDS
    )

    if not found:
        return None

    # Treat paddy and rice as the same crop
    if "paddy" in found and "rice" not in found:
        return "rice"

    return found[0]


def detect_topic(query):

    query = query.lower()

    topic_scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():

        score = 0

        for keyword in keywords:

            if keyword in query:
                score += 1

        topic_scores[topic] = score

    best_topic = max(
        topic_scores,
        key=lambda topic: topic_scores[topic]
    )

    if topic_scores[best_topic] == 0:
        return None

    return best_topic


# --------------------------------------------------
# RAG CLASS
# --------------------------------------------------

class AgricultureRAG:

    def __init__(self):

        print(
            "Loading agriculture embedding model..."
        )

        self.embedding_model = get_embedding_model()

        self.client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        print(
            f"RAG collection loaded: "
            f"{self.collection.count()} chunks"
        )

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    def search(
        self,
        query,
        n_results=8):
    

        if not query or not query.strip():
            return []

        query = query.strip()

        # ------------------------------------------
        # Understand query
        # ------------------------------------------

        query_crop = detect_crop(query)
        query_topic = detect_topic(query)

        print(f"\nRAG query: {query}")
        print(f"Detected crop: {query_crop}")
        print(f"Detected topic: {query_topic}")

        # ------------------------------------------
        # Create query embedding
        # ------------------------------------------

        query_embedding = self.embedding_model.encode(
            [query]
        ).tolist()

        # ------------------------------------------
        # Retrieve candidates
        # ------------------------------------------

        candidate_count = max(
            n_results * 8,
            40
        )

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=candidate_count,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        documents_result = results.get("documents")
        metadatas_result = results.get("metadatas")
        distances_result = results.get("distances")

        if not documents_result:
            return []

        documents = documents_result[0]

        metadatas = (
            metadatas_result[0]
            if metadatas_result
            else [{} for _ in documents]
        )

        distances = (
            distances_result[0]
            if distances_result
            else [None for _ in documents]
        )

        # ------------------------------------------
        # RERANK RESULTS
        # ------------------------------------------

        reranked = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            metadata = metadata or {}

            crop = str(
                metadata.get("crop", "")
            ).lower()

            topic = str(
                metadata.get("topic", "")
            ).lower()

            crop_words = [
                word.strip()
                for word in crop.split(",")
                if word.strip()
            ]

            # --------------------------------------
            # Detect whether this chunk belongs
            # to the requested crop
            # --------------------------------------

            crop_match = False

            if query_crop:

                if query_crop in crop_words:
                    crop_match = True

                elif query_crop in crop:
                    crop_match = True

            # --------------------------------------
            # IMPORTANT:
            # Reject unrelated crop-specific chunks
            # --------------------------------------

            if query_crop and crop:

                # Metadata contains crop information.
                # If requested crop is not present,
                # don't use this chunk.
                if not crop_match:

                    # Allow only truly general chunks
                    # without crop-specific information.
                    if crop != "general":

                        continue

            # --------------------------------------
            # Base semantic score
            # --------------------------------------

            score = 0.0

            if distance is not None:

                score += max(
                    0,
                    1.5 - float(distance)
                )

            # --------------------------------------
            # Strong crop matching
            # --------------------------------------

            if query_crop:

                if query_crop in crop_words:

                    score += 5.0

                elif query_crop in crop:

                    score += 3.5

                elif crop == "general":

                    score += 0.5

            # --------------------------------------
            # Topic matching
            # --------------------------------------

            if query_topic:

                if topic == query_topic:

                    score += 3.0

                elif topic == "general":

                    score += 0.5

                else:

                    score -= 1.0

            # --------------------------------------
            # Keyword evidence
            # --------------------------------------

            document_lower = document.lower()

            if query_crop:

                if query_crop in document_lower:

                    score += 1.0

            if query_topic:

                topic_words = TOPIC_KEYWORDS.get(
                    query_topic,
                    []
                )

                matches = 0

                for keyword in topic_words:

                    if keyword in document_lower:

                        matches += 1

                score += min(
                    matches * 0.1,
                    0.8
                )

            # --------------------------------------
            # Store result
            # --------------------------------------

            reranked.append(
                {
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                    "score": score
                }
            )

        # ------------------------------------------
        # Sort by relevance
        # ------------------------------------------

        reranked.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        # ------------------------------------------
        # Return best results
        # ------------------------------------------

        return reranked[:n_results]

    # --------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------

    def build_context(self, results):

        if not results:
            return ""

        context_parts = []

        for number, result in enumerate(
            results,
            start=1
        ):

            document = result["document"]

            metadata = result["metadata"]

            source = metadata.get(
                "source",
                "Unknown source"
            )

            page = metadata.get(
                "page",
                "Unknown"
            )

            topic = metadata.get(
                "topic",
                "Unknown"
            )

            crop = metadata.get(
                "crop",
                "Unknown"
            )

            context_parts.append(
                f"""
[AGRICULTURE SOURCE {number}]

Source: {source}
Page: {page}
Crop: {crop}
Topic: {topic}

{document}
"""
            )

        return "\n\n".join(
            context_parts
        )

    # --------------------------------------------------
    # SOURCE INFORMATION
    # --------------------------------------------------

    def get_sources(self, results):

        sources = []

        for result in results:

            metadata = result["metadata"]

            source = metadata.get(
                "source",
                "Unknown source"
            )

            page = metadata.get(
                "page",
                "Unknown"
            )

            source_info = {
                "source": source,
                "page": page
            }

            if source_info not in sources:

                sources.append(
                    source_info
                )

        return sources

    # --------------------------------------------------
    # DATABASE INFORMATION
    # --------------------------------------------------

    def count(self):

        return self.collection.count()