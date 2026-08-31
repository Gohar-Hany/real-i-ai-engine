from enum import Enum

class VectorDBEnums(Enum):
    QDRANT = "QDRANT"
    CHROMADB = "CHROMADB"


class DistanceMethodEnums(Enum):
    COSINE = "cosine"
    DOT = "dot"
