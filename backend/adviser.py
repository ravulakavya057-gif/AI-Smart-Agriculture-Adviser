from backend.rag import AgricultureRAG
from backend.ollama_service import generate_answer


class AgricultureAdviser:

    def __init__(self):
        self.rag = AgricultureRAG()

    def answer(
        self,
        question,
        image_description=""
    ):

        if not question and not image_description:
            return {
                "answer": "Please provide an agriculture question or image.",
                "sources": []
            }

        # ------------------------------------------
        # Build search query
        # ------------------------------------------

        search_query = question or ""

        if image_description:
            search_query = (
                f"{search_query} {image_description}"
            ).strip()

        # ------------------------------------------
        # Retrieve agriculture knowledge
        # ------------------------------------------

        results = self.rag.search(
            search_query,
            n_results=3
        )

        # ------------------------------------------
        # Keep only useful results
        # ------------------------------------------

        relevant_results = []

        for result in results:

            score = result.get(
                "score",
                0
            )

            if score > 1.0:
                relevant_results.append(result)

        # If filtering removed everything,
        # use the best available results.
        if not relevant_results:
            relevant_results = results[:3]

        # ------------------------------------------
        # Build context
        # ------------------------------------------

        context = self.rag.build_context(
            relevant_results
        )

        # ------------------------------------------
        # Generate answer
        # ------------------------------------------

        answer = generate_answer(
            question=question,
            context=context,
            image_description=image_description
        )

        # ------------------------------------------
        # Real source information
        # ------------------------------------------

        sources = self.rag.get_sources(
            relevant_results
        )

        return {
            "answer": answer,
            "sources": sources
        }