import os
from typing import Dict, Iterator, List
from urllib.parse import urlparse
from datetime import datetime, timedelta
from unstructured.ingest.connector.slack import SimpleSlackConfig, SlackAccessConfig
from unstructured.ingest.interfaces import (
    PartitionConfig,
    ProcessorConfig,
    ReadConfig,
    ChunkingConfig,
)
from unstructured.ingest.runner import SlackRunner

from backend.logger import logger
from backend.settings import settings
from backend.modules.dataloaders.loader import BaseDataLoader
from backend.types import DataIngestionMode, DataPoint, DataSource, LoadedDataPoint

# Default number of days to look back if start_date is not specified
DEFAULT_DAYS_LOOKBACK = 7

def format_date(date: datetime) -> str:
    """Format date to YYYY-MM-DDTHH:MM:SS"""
    return date.strftime("%Y-%m-%dT%H:%M:%S")

class SlackLoader(BaseDataLoader):
    """
    Load data from a Slack workspace using unstructured's SlackConnector
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
        Loads data from a Slack workspace specified by the given source URI.
        """
        slack_url = data_source.uri
        
        # Parse the URL to extract the workspace name or ID if needed
        parsed_url = urlparse(slack_url)
        workspace = parsed_url.netloc.split('.')[0]  # This assumes the URL format is 'workspace.slack.com'

        # Determine start and end dates
        now = datetime.now()
        
        if settings.SLACK_END_DATE:
            end_date = settings.SLACK_END_DATE
        else:
            end_date = format_date(now)

        if settings.SLACK_START_DATE:
            start_date = settings.SLACK_START_DATE
        else:
            # If start_date is not specified, use the default lookback period
            start_date = format_date(now - timedelta(days=DEFAULT_DAYS_LOOKBACK))

        runner = SlackRunner(
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
            connector_config=SimpleSlackConfig(
                access_config=SlackAccessConfig(
                    token=settings.SLACK_TOKEN,
                ),
                channels=settings.SLACK_CHANNELS,
                start_date=start_date,
                end_date=end_date,
            ),
        )

        # Run the Slack runner
        logger.info(f"Starting Slack data ingestion with chunking from {start_date} to {end_date}")
        runner.run()
        logger.info("Slack data ingestion and chunking completed")

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