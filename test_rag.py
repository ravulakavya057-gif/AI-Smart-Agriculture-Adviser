from backend.rag import AgricultureRAG

rag = AgricultureRAG()

results = rag.search(
    "What are the major pests of rice?",
    8
)

print("\n--- RETRIEVED CONTENT ---")

for i, result in enumerate(results, start=1):

    print("\n" + "=" * 60)
    print(f"RESULT {i}")
    print("Distance:", result["distance"])
    print("Metadata:", result["metadata"])
    print("-" * 60)

    print(result["document"][:1000])