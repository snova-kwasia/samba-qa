from typing import List
from urllib.parse import urlparse

from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from backend.constants import DATA_POINT_FQN_METADATA_KEY, DATA_POINT_HASH_METADATA_KEY
from backend.logger import logger
from backend.modules.vector_db.base import BaseVectorDB
from backend.types import DataPointVector, QdrantClientConfig, VectorDBConfig

from fastembed.sparse.bm25 import Bm25
from fastembed.late_interaction import LateInteractionTextEmbedding
from tqdm import tqdm



MAX_SCROLL_LIMIT = int(1e6)
BATCH_SIZE = 1000

class QdrantVectorDB(BaseVectorDB):
    def __init__(self, config: VectorDBConfig):
        logger.debug(f"Connecting to qdrant using config: {config.dict()}")
        if config.local is True:
            self.qdrant_client = QdrantClient(
                path="./qdrant_db",
            )
        else:
            url = config.url
            api_key = config.api_key
            if not api_key:
                api_key = None
            qdrant_kwargs = QdrantClientConfig.parse_obj(config.config or {})
            if url.startswith("http://") or url.startswith("https://"):
                if qdrant_kwargs.port is None:
                    parsed_port = urlparse(url).port
                    if parsed_port:
                        qdrant_kwargs.port = parsed_port
                    else:
                        qdrant_kwargs.port = 443 if url.startswith("https://") else 6333
            self.qdrant_client = QdrantClient(
                url=url, api_key=api_key, **qdrant_kwargs.dict()
            )

    def create_collection(self, collection_name: str, embeddings: Embeddings):
        logger.debug(f"[Qdrant] Creating new collection {collection_name}")

        # Calculate embedding size for dense embeddings
        dense_vector_size = len(embeddings.embed_documents(["Initial document"])[0])

        # Get size for late interaction embeddings
        late_interaction_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
        late_interaction_embeddings = list(late_interaction_model.passage_embed(["Initial document"]))
        late_interaction_size = len(late_interaction_embeddings[0][0])

        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense_embedding": models.VectorParams(
                    size=dense_vector_size,
                    distance=models.Distance.COSINE,
                    on_disk=True
                ),
                "late_interaction": models.VectorParams(
                    size=late_interaction_size,
                    on_disk=True,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    )
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                    index=models.SparseIndexParams(on_disk=True)
                )
            }
        )

        self.qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name=f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.debug(f"[Qdrant] Created new collection {collection_name}")

    def _get_records_to_be_upserted(
        self, collection_name: str, data_point_fqns: List[str], incremental: bool
    ):
        if not incremental:
            return []
        logger.debug(
            f"[Qdrant] Incremental Ingestion: Fetching documents for {len(data_point_fqns)} data point fqns for collection {collection_name}"
        )
        stop = False
        offset = None
        record_ids_to_be_upserted = []
        while stop is not True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key=f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
                            match=models.MatchAny(
                                any=data_point_fqns,
                            ),
                        ),
                    ]
                ),
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for record in records:
                record_ids_to_be_upserted.append(record.id)
                if len(record_ids_to_be_upserted) > MAX_SCROLL_LIMIT:
                    stop = True
                    break
            if next_offset is None:
                stop = True
            else:
                offset = next_offset

        logger.debug(
            f"[Qdrant] Incremental Ingestion: collection={collection_name} Addition={len(data_point_fqns)}, Updates={len(record_ids_to_be_upserted)}"
        )
        return record_ids_to_be_upserted

    def upsert_documents(
        self,
        collection_name: str,
        documents,
        embeddings: Embeddings,
        incremental: bool = True,
    ):
        if len(documents) == 0:
            logger.warning("No documents to index")
            return
        
        logger.debug(f"[Qdrant] Adding {len(documents)} documents to collection {collection_name}")
        
        data_point_fqns = []
        for document in documents:
            if document.metadata.get(DATA_POINT_FQN_METADATA_KEY):
                data_point_fqns.append(
                    document.metadata.get(DATA_POINT_FQN_METADATA_KEY)
                )
        record_ids_to_be_upserted: List[str] = self._get_records_to_be_upserted(
            collection_name=collection_name,
            data_point_fqns=data_point_fqns,
            incremental=incremental,
        )

        bm25_embedding_model = Bm25("Qdrant/bm25")
        late_interaction_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

        batch_size = 8

        total_batches = (len(documents) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(documents), batch_size), total=total_batches, desc="Processing batches"):
            batch = documents[i:i+batch_size]
            texts = [doc.page_content for doc in batch]

            logger.debug(f"[Qdrant] Generating dense embeddings for batch {i//batch_size + 1}/{total_batches}")
            dense_embeddings = embeddings.embed_documents(texts)
            logger.debug(f"[Qdrant] Dense embeddings generated for batch {i//batch_size + 1}/{total_batches}")

            logger.debug(f"[Qdrant] Generating sparse embeddings for batch {i//batch_size + 1}/{total_batches}")
            sparse_embeddings = [
        {
            "values": emb.values.tolist(),
            "indices": emb.indices.tolist()
        }
        for emb in bm25_embedding_model.passage_embed(texts)
    ]
            logger.debug(f"[Qdrant] Sparse embeddings generated for batch {i//batch_size + 1}/{total_batches}")

            logger.debug(f"[Qdrant] Generating late interaction embeddings for batch {i//batch_size + 1}/{total_batches}")
            late_interaction_embeddings = list(late_interaction_model.passage_embed(texts))
            logger.debug(f"[Qdrant] Late interaction embeddings generated for batch {i//batch_size + 1}/{total_batches}")

            points = []
            for j, doc in enumerate(batch):
                point = models.PointStruct(
                    id=int(doc.metadata.get("_id", i * batch_size + j)),
                    vector={
                        "dense_embedding": dense_embeddings[j],
                        "bm25": sparse_embeddings[j],
                        "late_interaction": late_interaction_embeddings[j].tolist(),
                    },
                    payload={
                        "metadata": doc.metadata,
                        "content": doc.page_content,
                    }
                )
                points.append(point)

            logger.debug(f"[Qdrant] Upserting batch {i//batch_size + 1}/{total_batches}")
            try:
                logger.debug(f"Sample point structure: {points[0].dict()}")
                self.qdrant_client.upload_points(
                collection_name=collection_name,
                points=points,
                batch_size=batch_size,
            )
                logger.debug(f"[Qdrant] Batch {i//batch_size + 1}/{total_batches} upserted successfully")
            except Exception as e:
                logger.error(f"Error upserting batch {i//batch_size + 1}/{total_batches}: {str(e)}")
                logger.error(f"First point in batch: {points[0]}")
                raise e

        logger.debug(
            f"[Qdrant] Added {len(documents)} documents to collection {collection_name}"
        )

        if len(record_ids_to_be_upserted):
            logger.debug(
                f"[Qdrant] Deleting {len(record_ids_to_be_upserted)} outdated documents from collection {collection_name}"
            )
            for i in range(0, len(record_ids_to_be_upserted), batch_size):
                record_ids_to_be_processed = record_ids_to_be_upserted[
                    i : i + batch_size
                ]
                self.qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(
                        points=record_ids_to_be_processed,
                    ),
                )
            logger.debug(
                f"[Qdrant] Deleted {len(record_ids_to_be_upserted)} outdated documents from collection {collection_name}"
            )

    def get_collections(self) -> List[str]:
        logger.debug(f"[Qdrant] Fetching collections")
        collections = self.qdrant_client.get_collections().collections
        logger.debug(f"[Qdrant] Fetched {len(collections)} collections")
        return [collection.name for collection in collections]

    def delete_collection(self, collection_name: str):
        logger.debug(f"[Qdrant] Deleting {collection_name} collection")
        self.qdrant_client.delete_collection(collection_name=collection_name)
        logger.debug(f"[Qdrant] Deleted {collection_name} collection")

    def get_vector_store(self, collection_name: str, embeddings: Embeddings):
        logger.debug(f"[Qdrant] Getting vector store for collection {collection_name}")
        return Qdrant(
            client=self.qdrant_client,
            embeddings=embeddings,
            collection_name=collection_name,
        )

    def get_vector_client(self):
        logger.debug(f"[Qdrant] Getting Qdrant client")
        return self.qdrant_client

    def list_data_point_vectors(
        self, collection_name: str, data_source_fqn: str, batch_size: int = BATCH_SIZE
    ) -> List[DataPointVector]:
        logger.debug(
            f"[Qdrant] Listing all data point vectors for collection {collection_name}"
        )
        stop = False
        offset = None
        data_point_vectors: List[DataPointVector] = []
        while stop is not True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                with_payload=[
                    f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
                    f"metadata.{DATA_POINT_HASH_METADATA_KEY}",
                ],
                scroll_filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key=f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
                            match=models.MatchText(
                                text=data_source_fqn,
                            ),
                        ),
                    ]
                ),
                with_vectors=False,
                offset=offset,
            )
            for record in records:
                metadata: dict = record.payload.get("metadata")
                if (
                    metadata
                    and metadata.get(DATA_POINT_FQN_METADATA_KEY)
                    and metadata.get(DATA_POINT_HASH_METADATA_KEY)
                ):
                    data_point_vectors.append(
                        DataPointVector(
                            data_point_vector_id=record.id,
                            data_point_fqn=metadata.get(DATA_POINT_FQN_METADATA_KEY),
                            data_point_hash=metadata.get(DATA_POINT_HASH_METADATA_KEY),
                        )
                    )
                if len(data_point_vectors) > MAX_SCROLL_LIMIT:
                    stop = True
                    break
            if next_offset is None:
                stop = True
            else:
                offset = next_offset
        logger.debug(
            f"[Qdrant] Listing {len(data_point_vectors)} data point vectors for collection {collection_name}"
        )
        return data_point_vectors

    def delete_data_point_vectors(
        self,
        collection_name: str,
        data_point_vectors: List[DataPointVector],
        batch_size: int = BATCH_SIZE,
    ):
        logger.debug(f"[Qdrant] Deleting {len(data_point_vectors)} data point vectors")
        vectors_to_be_deleted_count = len(data_point_vectors)
        deleted_vectors_count = 0
        for i in range(0, vectors_to_be_deleted_count, batch_size):
            data_point_vectors_to_be_processed = data_point_vectors[i : i + batch_size]
            self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[
                        document_vector_point.data_point_vector_id
                        for document_vector_point in data_point_vectors_to_be_processed
                    ],
                ),
            )
            deleted_vectors_count = deleted_vectors_count + len(
                data_point_vectors_to_be_processed
            )
            logger.debug(
                f"[Qdrant] Deleted [{deleted_vectors_count}/{vectors_to_be_deleted_count}] data point vectors"
            )
        logger.debug(
            f"[Qdrant] Deleted {vectors_to_be_deleted_count} data point vectors"
        )

    def list_documents_in_collection(
        self, collection_name: str, base_document_id: str = None
    ) -> List[str]:
        logger.debug(
            f"[Qdrant] Listing all documents with base document id {base_document_id} for collection {collection_name}"
        )
        stop = False
        offset = None
        document_ids_set = set()
        while stop is not True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    should=(
                        [
                            models.FieldCondition(
                                key=f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
                                match=models.MatchText(
                                    text=base_document_id,
                                ),
                            ),
                        ]
                        if base_document_id
                        else None
                    )
                ),
                limit=BATCH_SIZE,
                with_payload=[f"metadata.{DATA_POINT_FQN_METADATA_KEY}"],
                with_vectors=False,
                offset=offset,
            )
            for record in records:
                if record.payload.get("metadata") and record.payload.get(
                    "metadata"
                ).get(DATA_POINT_FQN_METADATA_KEY):
                    document_ids_set.add(
                        record.payload.get("metadata").get(DATA_POINT_FQN_METADATA_KEY)
                    )
                if len(document_ids_set) > MAX_SCROLL_LIMIT:
                    stop = True
                    break
            if next_offset is None:
                stop = True
            else:
                offset = next_offset
        logger.debug(
            f"[Qdrant] Found {len(document_ids_set)} documents with base document id {base_document_id} for collection {collection_name}"
        )
        return list(document_ids_set)

    def delete_documents(self, collection_name: str, document_ids: List[str]):
        logger.debug(
            f"[Qdrant] Deleting {len(document_ids)} documents from collection {collection_name}"
        )
        try:
            self.qdrant_client.get_collection(collection_name=collection_name)
        except Exception as exp:
            logger.debug(exp)
            return

        for i in range(0, len(document_ids), BATCH_SIZE):
            document_ids_to_be_processed = document_ids[i : i + BATCH_SIZE]
            self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key=f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
                                match=models.MatchAny(any=document_ids_to_be_processed),
                            ),
                        ],
                    )
                ),
            )
        logger.debug(
            f"[Qdrant] Deleted {len(document_ids)} documents from collection {collection_name}"
        )

    def list_document_vector_points(
        self, collection_name: str
    ) -> List[DataPointVector]:
        logger.debug(
            f"[Qdrant] Listing all document vector points for collection {collection_name}"
        )
        stop = False
        offset = None
        document_vector_points: List[DataPointVector] = []
        while stop is not True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=BATCH_SIZE,
                with_payload=[
                    f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
                    f"metadata.{DATA_POINT_HASH_METADATA_KEY}",
                ],
                with_vectors=False,
                offset=offset,
            )
            for record in records:
                metadata: dict = record.payload.get("metadata")
                if (
                    metadata
                    and metadata.get(DATA_POINT_FQN_METADATA_KEY)
                    and metadata.get(DATA_POINT_HASH_METADATA_KEY)
                ):
                    document_vector_points.append(
                        DataPointVector(
                            point_id=record.id,
                            document_id=metadata.get(DATA_POINT_FQN_METADATA_KEY),
                            document_hash=metadata.get(DATA_POINT_HASH_METADATA_KEY),
                        )
                    )
                if len(document_vector_points) > MAX_SCROLL_LIMIT:
                    stop = True
                    break
            if next_offset is None:
                stop = True
            else:
                offset = next_offset
        logger.debug(
            f"[Qdrant] Listing {len(document_vector_points)} document vector points for collection {collection_name}"
        )
        return document_vector_points