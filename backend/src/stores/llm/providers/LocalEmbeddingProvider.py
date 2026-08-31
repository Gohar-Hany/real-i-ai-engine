from sentence_transformers import SentenceTransformer
import torch
from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums

class LocalEmbeddingProvider(LLMInterface):
    def __init__(self, default_embedding_model: str = "Alibaba-NLP/gte-multilingual-base",
                 default_embedding_size: int = 768):
        self.model = None
        self.model_id = default_embedding_model
        self.embedding_size = default_embedding_size
        self.enums = OpenAIEnums

    def set_generation_model(self, model_id: str):
        pass

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.model_id = model_id
        self.embedding_size = embedding_size
        print(f"Loading local embedding model: {self.model_id}...")
        self.model = SentenceTransformer(self.model_id, trust_remote_code=True)
        
        # Apply GTE multilingual base patch if needed
        if "gte-multilingual-base" in self.model_id:
            print("Applying position_ids patch for GTE model...")
            base_transformer = self.model[0].auto_model
            if hasattr(base_transformer, "embeddings") and hasattr(base_transformer.embeddings, "position_ids"):
                max_positions = base_transformer.config.max_position_embeddings
                base_transformer.embeddings.position_ids = torch.arange(max_positions).expand((1, -1))

    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                      temperature: float = None):
        return ""

    def embed_text(self, text: str, document_type: str = None):
        if self.model is None:
            self.set_embedding_model(self.model_id, self.embedding_size)
        
        processed_text = text
        if "gte-multilingual-base" in self.model_id:
            if document_type == "query":
                processed_text = f"Instruct: retrieve semantically similar documents.\nQuery: {text}"
        
        embedding = self.model.encode([processed_text])[0]
        return embedding.tolist()

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": prompt}
