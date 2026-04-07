"""
Dataset Loader: Carga el dataset WikiCAT_esv2 de HuggingFace en ChromaDB.

El dataset PlanTL-GOB-ES/WikiCAT_esv2 contiene artículos de Wikipedia en español
clasificados en 12 categorías. Cada artículo se almacena como un chunk en ChromaDB
con su texto y etiqueta de categoría como metadato.

Uso:
    python -m src.rag.load_dataset
"""

from datasets import load_dataset
import chromadb
from chromadb.utils import embedding_functions
import os
from tqdm import tqdm

# Configuración
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "wikicat_es"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_NAME = "PlanTL-GOB-ES/WikiCAT_esv2"
DATASET_CONFIG = "wikiCAT_es"

# Mapeo de etiquetas numéricas a nombres de categoría
LABEL_NAMES = [
    "Religión", "Entretenimiento", "Música", "Ciencia_y_Tecnología",
    "Política", "Economía", "Matemáticas", "Humanidades",
    "Deporte", "Derecho", "Historia", "Filosofía"
]


def load_wikicat_to_chroma(
    dataset_name: str = DATASET_NAME,
    dataset_config: str = DATASET_CONFIG,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
    batch_size: int = 100,
    split: str = "train"
):
    """
    Carga el dataset WikiCAT_esv2 desde HuggingFace y lo inserta en una colección de ChromaDB.

    Args:
        dataset_name: Nombre del dataset en HuggingFace.
        dataset_config: Configuración/subconjunto del dataset.
        persist_dir: Directorio de persistencia de ChromaDB.
        collection_name: Nombre de la colección en ChromaDB.
        embedding_model: Modelo de embeddings a utilizar.
        batch_size: Tamaño de lote para insertar documentos.
        split: Split del dataset a cargar ('train' o 'test').
    """
    print(f"[RAG] Cargando dataset '{dataset_name}' (config: {dataset_config}, split: {split})...")
    dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
    print(f"[RAG] Dataset cargado: {len(dataset)} artículos.")

    # Inicializar ChromaDB
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    client = chromadb.PersistentClient(path=persist_dir)

    # Borrar colección existente si la hay, para empezar limpio
    try:
        client.delete_collection(name=collection_name)
        print(f"[RAG] Colección '{collection_name}' existente eliminada.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Insertar documentos por lotes
    print(f"[RAG] Insertando documentos en lotes de {batch_size}...")
    ids_batch = []
    documents_batch = []
    metadatas_batch = []

    for idx, item in enumerate(tqdm(dataset, desc="Procesando artículos")):
        text = item["text"]
        label_id = item["label"]

        # Solo cargar artículos de la categoría Economía (label=5)
        if isinstance(label_id, int) and label_id != 5:
            continue
        label_name = "Economía"

        # Saltar textos vacíos
        if not text or len(text.strip()) == 0:
            continue

        ids_batch.append(f"wikicat_{split}_{idx}")
        documents_batch.append(text)
        metadatas_batch.append({"label": label_name, "source": "WikiCAT_esv2", "split": split})

        # Insertar cuando el lote está lleno
        if len(ids_batch) >= batch_size:
            collection.add(
                ids=ids_batch,
                documents=documents_batch,
                metadatas=metadatas_batch
            )
            ids_batch = []
            documents_batch = []
            metadatas_batch = []

    # Insertar el último lote parcial
    if ids_batch:
        collection.add(
            ids=ids_batch,
            documents=documents_batch,
            metadatas=metadatas_batch
        )

    print(f"[RAG] ¡Carga completada! Total de documentos en la colección: {collection.count()}")


if __name__ == "__main__":
    load_wikicat_to_chroma()
