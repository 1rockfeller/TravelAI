import os
import logging
from typing import List

from dotenv import load_dotenv
from azure.cosmos import CosmosClient, PartitionKey
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding

load_dotenv()

logger = logging.getLogger(__name__)
_client = None
_container = None


def get_container():
    """Create or retrieve the configured vector container on first use."""
    global _client, _container
    if _container is None:
        dimensions = int(os.getenv("AZURE_OPENAI_EMBED_DIMENSIONS", "1536"))
        _client = CosmosClient(
            os.environ["COSMOS_ENDPOINT"],
            os.environ["COSMOS_KEY"],
        )
        database = _client.create_database_if_not_exists(
            id=os.environ["COSMOS_DB"]
        )
        embedding_policy = {
            "vectorEmbeddings": [{
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": dimensions,
            }]
        }
        indexing_policy = {
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [
                {"path": '/"_etag"/?'},
                {"path": "/embedding/*"},
            ],
            "vectorIndexes": [{
                "path": "/embedding",
                "type": "quantizedFlat",
            }],
        }
        _container = database.create_container_if_not_exists(
            id=os.environ["COSMOS_CONTAINER"],
            partition_key=PartitionKey(path=os.environ["COSMOS_PARTITION_KEY"]),
            indexing_policy=indexing_policy,
            vector_embedding_policy=embedding_policy,
        )
        logger.info(
            "Using Cosmos DB container %s/%s for vector ingestion",
            os.environ["COSMOS_DB"],
            os.environ["COSMOS_CONTAINER"],
        )
    return _container


def create_embedding_kernel():
    kernel = Kernel()
    kernel.add_service(
        AzureTextEmbedding(
            deployment_name=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"],
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"]
        )
    )
    return kernel


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate a batch of embeddings using Semantic Kernel.
    """
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every embedding input must be a non-empty string")

    kernel = create_embedding_kernel()
    embedding_service = kernel.get_service(type=AzureTextEmbedding)
    vectors = await embedding_service.generate_embeddings(texts)
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    return [[float(value) for value in vector] for vector in vectors]


async def upsert_snippet(doc_id, text, pk="cards"):
    """
    Generate a vector and upsert the snippet into Cosmos DB.
    """
    if not doc_id or not text.strip() or not pk:
        raise ValueError("doc_id, text, and pk must not be empty")
    try:
        embedding = (await embed_texts([text]))[0]
        document = {
            "id": doc_id,
            "pk": pk,
            "text": text,
            "embedding": embedding,
        }
        get_container().upsert_item(body=document)
        logger.info("Upserted vector snippet %s", doc_id)
        return document
    except Exception:
        logger.exception("Failed to upsert vector snippet %s", doc_id)
        raise

if __name__ == "__main__":
    import asyncio

    async def ingest_examples():
        snippets = [
            ("bankgold-1", "BankGold: 4x points on dining worldwide; no FX fees."),
            ("lounge-1", "Lounge access in CDG Terminal 2 with BankGold Premium."),
            ("bankgold-2", "BankGold Premium: Priority boarding, lounge access, concierge service."),
            ("dining-1", "4x points on dining worldwide with BankGold card."),
        ]
        for doc_id, text in snippets:
            await upsert_snippet(doc_id, text)

    asyncio.run(ingest_examples())
