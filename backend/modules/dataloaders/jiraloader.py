import os
from typing import Dict, Iterator, List
from unstructured.ingest.connector.jira import JiraAccessConfig, SimpleJiraConfig
from unstructured.ingest.interfaces import (
    PartitionConfig,
    ProcessorConfig,
    ReadConfig,
    ChunkingConfig,
)
from unstructured.ingest.runner import JiraRunner

from backend.logger import logger
from backend.settings import settings
from backend.modules.dataloaders.loader import BaseDataLoader
from backend.types import DataIngestionMode, DataPoint, DataSource, LoadedDataPoint

class JiraLoader(BaseDataLoader):
    """
    Load data from a Jira instance using unstructured's JiraConnector
    """

    def load_filtered_data(
        self,
        data_source: DataSource,
        dest_dir: str,
        previous_snapshot: Dict[str, str],
        batch_size: int,
        data_ingestion_mode: DataIngestionMode,
    ) -> Iterator[List[LoadedDataPoint]]:
        """
        Loads data from a Jira instance specified by the given source URI.
        """
        jira_url = data_source.uri

        runner = JiraRunner(
            processor_config=ProcessorConfig(
                verbose=True,
                output_dir=dest_dir,
                num_processes=2,
            ),
            read_config=ReadConfig(),
            partition_config=PartitionConfig(
                metadata_exclude=[
                    "filename",
                    "file_directory",
                    "metadata.data_source.date_processed",
                ],
            ),
            chunking_config=ChunkingConfig(
                chunk_elements=True,
                chunking_strategy="by_title",
                max_characters=1500,  # Set maximum characters per chunk
                overlap=300,  # Set overlap between chunks
                combine_text_under_n_chars=1500,  # Combine small sections
            ),
            connector_config=SimpleJiraConfig(
                projects=['CP'],
                access_config=JiraAccessConfig(
                    api_token=settings.JIRA_API_TOKEN,
                ),
                url=jira_url,
                user_email=settings.JIRA_EMAIL,
            ),
        )

        # Run the Jira runner
        logger.info("Starting Jira data ingestion with chunking")
        runner.run()
        logger.info("Jira data ingestion and chunking completed")

        loaded_data_points: List[LoadedDataPoint] = []
        for root, _, files in os.walk(dest_dir):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, dest_dir)
                file_size = os.path.getsize(full_path)

                data_point = DataPoint(
                    data_source_fqn=data_source.fqn,
                    data_point_uri=relative_path,
                    data_point_hash=f"{file_size}",
                )

                # If the data ingestion mode is incremental, check if the data point already exists.
                if (
                    data_ingestion_mode == DataIngestionMode.INCREMENTAL
                    and previous_snapshot.get(data_point.data_point_fqn)
                    and previous_snapshot.get(data_point.data_point_fqn)
                    == data_point.data_point_hash
                ):
                    continue

                loaded_data_points.append(
                    LoadedDataPoint(
                        data_point_hash=data_point.data_point_hash,
                        data_point_uri=data_point.data_point_uri,
                        data_source_fqn=data_point.data_source_fqn,
                        local_filepath=full_path,
                        file_extension=os.path.splitext(file)[1],
                    )
                )

                if len(loaded_data_points) >= batch_size:
                    yield loaded_data_points
                    loaded_data_points.clear()

        if loaded_data_points:
            yield loaded_data_points