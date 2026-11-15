import logging
import os

from azure.cosmos import CosmosClient
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding

load_dotenv()

logger = logging.getLogger(__name__)
_client = None
_container = None


def get_cosmos_client():
    """Return the process-wide Cosmos client and configured snippets container."""
    global _client, _container
    if _client is None:
        _client = CosmosClient(
            os.environ["COSMOS_ENDPOINT"],
            os.environ["COSMOS_KEY"],
        )
        _container = _client.get_database_client(
            os.environ["COSMOS_DB"]
        ).get_container_client(os.environ["COSMOS_CONTAINER"])
        logger.info(
            "Connected to Cosmos DB vector container %s/%s",
            os.environ["COSMOS_DB"],
            os.environ["COSMOS_CONTAINER"],
        )
    return _client, _container


def create_embedding_kernel():
    kernel = Kernel()
    kernel.add_service(
        AzureTextEmbedding(
            deployment_name=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"],
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    )
    return kernel


async def embed_texts(texts):
    """Generate embeddings for a batch of strings using AzureTextEmbedding."""
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


async def retrieve(query: str, k: int = 5, partition_key: str = None):
    """Retrieve the nearest vector snippets using Cosmos DB VectorDistance."""
    if not query.strip():
        raise ValueError("Retrieval query must not be empty")
    if not isinstance(k, int) or not 1 <= k <= 20:
        raise ValueError("k must be an integer between 1 and 20")

    embeddings = await embed_texts([query])
    if not embeddings or not embeddings[0]:
        raise ValueError("Embedding service returned an empty query vector")
    query_vector = embeddings[0]
    logger.info("Generated query embedding with %d dimensions", len(query_vector))

    sql = f"""
    SELECT TOP {k} c.id, c.text, c.pk,
           VectorDistance(c.embedding, @queryVector) AS similarity_score
    FROM c
    WHERE IS_DEFINED(c.embedding) AND IS_ARRAY(c.embedding)
    """
    parameters = [{"name": "@queryVector", "value": query_vector}]
    if partition_key is not None:
        sql += " AND c.pk = @pk"
        parameters.append({"name": "@pk", "value": partition_key})
    sql += "\nORDER BY VectorDistance(c.embedding, @queryVector)"

    _, container = get_cosmos_client()
    query_options = {
        "query": sql,
        "parameters": parameters,
        "enable_cross_partition_query": partition_key is None,
    }
    if partition_key is not None:
        query_options["partition_key"] = partition_key

    results = list(container.query_items(**query_options))
    logger.info(
        "Cosmos VectorDistance retrieved %d result(s): %s",
        len(results),
        [round(float(result["similarity_score"]), 4) for result in results],
    )
    return results


def main():
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Demonstrate Cosmos vector retrieval")
    parser.add_argument("query", nargs="?", default="BankGold dining benefits")
    parser.add_argument("--partition-key", default="cards")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    print(f"Query: {args.query}", flush=True)
    results = asyncio.run(retrieve(args.query, k=args.limit, partition_key=args.partition_key))
    if not results:
        raise RuntimeError("VectorDistance returned no documents")
    for result in results:
        print(f"Cosine similarity: {result['similarity_score']:.4f}")
        print(f"Document: {result['text']}")


if __name__ == "__main__":
    main()
