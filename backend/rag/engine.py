"""
RAG Engine: ChromaDB-based retrieval for the Finance StratAIgist system.

Uses a ChromaDB collection with SentenceTransformer embeddings to retrieve
relevant document chunks for a given query.
"""

import chromadb
from chromadb.utils import embedding_functions
import os


# Ruta persistente para la base de datos ChromaDB
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "wikicat_es"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RAGEngine:
    """
    Motor de RAG que gestiona la conexión a ChromaDB y la recuperación de contexto.
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = COLLECTION_NAME,
                 embedding_model: str = EMBEDDING_MODEL):
        """
        Inicializa el motor RAG con ChromaDB y un modelo de embeddings.

        Args:
            persist_dir: Directorio donde ChromaDB persiste los datos.
            collection_name: Nombre de la colección en ChromaDB.
            embedding_model: Nombre del modelo de embeddings de SentenceTransformers.
        """
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[RAG] Colección '{collection_name}' cargada con {self.collection.count()} documentos.")

    def retrieve_context(self, query: str, top_k: int = 5, similarity_threshold: float = 0.75) -> list[dict]:
        """
        Recupera los documentos más relevantes para una consulta,
        filtrando aquellos cuya distancia coseno supere el umbral.

        Args:
            query: La consulta del usuario.
            top_k: Número máximo de documentos a recuperar.
            similarity_threshold: Distancia coseno máxima permitida.

        Returns:
            Lista de diccionarios con 'text', 'label' y 'distance' por cada resultado.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        context_list = []
        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i]
            if distance < similarity_threshold:
                context_list.append({
                    "text": results["documents"][0][i],
                    "label": results["metadatas"][0][i].get("label", "desconocido"),
                    "distance": distance
                })

        return context_list

    def format_rag_prompt(self, query: str, context_list: list[dict]) -> str:
        """
        Formatea un prompt con el contexto recuperado para enviarlo al LLM.

        Args:
            query: La consulta original del usuario.
            context_list: Lista de contextos recuperados.

        Returns:
            Prompt formateado con el contexto inyectado.
        """
        if not context_list:
            return f"Pregunta: {query}\nRespuesta:"

        context_text = "\n\n".join(
            [f"[Documento {i+1} — {ctx['label']}]\n{ctx['text']}" for i, ctx in enumerate(context_list)]
        )

        prompt = f"""Se ha recuperado el siguiente contexto de la base de conocimiento que podría ser relevante.
        Usalo siempre que sea util para responder la pregunta. Si no es relevante, responde con tu propio conocimiento.

        --- CONTEXTO ---
        {context_text}
        --- FIN CONTEXTO ---

        Pregunta: {query}
        Respuesta:"""

        return prompt
