import chromadb
from ..VectorDBInterface import VectorDBInterface
import logging
from typing import List
from models.db_schemes import RetrievedDocument

class ChromaDBProvider(VectorDBInterface):

    def __init__(self, db_path: str, distance_method: str = "cosine"):
        self.client = None
        self.db_path = db_path
        # ChromaDB HNSW space metadata setting: 'cosine', 'l2', or 'ip'
        self.distance_method = "cosine" if distance_method == "cosine" else "l2"
        self.logger = logging.getLogger(__name__)

    def connect(self):
        self.client = chromadb.PersistentClient(path=self.db_path)

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        try:
            self.client.get_collection(name=collection_name)
            return True
        except ValueError:
            return False

    def list_all_collections(self) -> List:
        return self.client.list_collections()

    def get_collection_info(self, collection_name: str) -> dict:
        collection = self.client.get_collection(name=collection_name)
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata
        }

    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            self.client.delete_collection(name=collection_name)

    def create_collection(self, collection_name: str, 
                                embedding_size: int,
                                do_reset: bool = False):
        if do_reset:
            self.delete_collection(collection_name=collection_name)

        # ChromaDB handles cosine vs l2 inside the metadata HNSW space config
        self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.distance_method}
        )
        return True

    def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, 
                         record_id: str = None):
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Cannot insert new record to non-existent collection: {collection_name}")
            return False

        try:
            collection = self.client.get_collection(name=collection_name)
            collection.add(
                ids=[str(record_id)],
                embeddings=[vector],
                metadatas=[metadata] if metadata else [{}],
                documents=[text]
            )
        except Exception as e:
            self.logger.error(f"Error while inserting record: {e}")
            return False
        return True

    def insert_many(self, collection_name: str, texts: list, 
                          vectors: list, metadata: list = None, 
                          record_ids: list = None, batch_size: int = 50):
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Cannot insert new record to non-existent collection: {collection_name}")
            return False

        if metadata is None:
            metadata = [{}] * len(texts)
        else:
            # Normalize metadata to contain only primitive types
            metadata = [dict(m) if m else {} for m in metadata]

        if record_ids is None:
            record_ids = [str(x) for x in range(len(texts))]
        else:
            record_ids = [str(x) for x in record_ids]

        try:
            collection = self.client.get_collection(name=collection_name)
            for i in range(0, len(texts), batch_size):
                batch_end = i + batch_size
                collection.add(
                    ids=record_ids[i:batch_end],
                    embeddings=vectors[i:batch_end],
                    metadatas=metadata[i:batch_end],
                    documents=texts[i:batch_end]
                )
        except Exception as e:
            self.logger.error(f"Error while inserting batch: {e}")
            return False

        return True

    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5) -> List[RetrievedDocument]:
        if not self.is_collection_existed(collection_name):
            return []

        collection = self.client.get_collection(name=collection_name)
        results = collection.query(
            query_embeddings=[vector],
            n_results=limit
        )

        if not results or not results['documents'] or len(results['documents'][0]) == 0:
            return []

        retrieved_docs = []
        for idx in range(len(results['documents'][0])):
            score = 1.0 - results['distances'][0][idx] if results['distances'] else 0.0
            retrieved_docs.append(
                RetrievedDocument(**{
                    "score": float(score),
                    "text": results['documents'][0][idx]
                })
            )

        return retrieved_docs
