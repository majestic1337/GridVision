from dataclasses import dataclass
import os

from config.constants import (
    DEFAULT_COLLECTION,
    DEFAULT_DENSE_VECTOR,
    DEFAULT_SPARSE_VECTOR,
    ENV_COLLECTION,
    ENV_DENSE_VECTOR,
    ENV_SPARSE_VECTOR,
)


@dataclass(frozen=True)
class RuntimeConfig:
    collection: str
    dense_vector: str
    sparse_vector: str


def get_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        collection=os.getenv(ENV_COLLECTION, DEFAULT_COLLECTION),
        dense_vector=os.getenv(ENV_DENSE_VECTOR, DEFAULT_DENSE_VECTOR),
        sparse_vector=os.getenv(ENV_SPARSE_VECTOR, DEFAULT_SPARSE_VECTOR),
    )
