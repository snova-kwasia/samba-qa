import os
from typing import List, Optional
from langchain.docstore.document import Document
from backend.modules.parsers.parser import BaseParser
from unstructured.partition.auto import partition, partition_pdf
from unstructured.chunking.basic import chunk_elements
from backend.modules.parsers.utils import additional_processing
from unstructured.staging.base import convert_to_dict
from backend.logger import logger  # Import the logger


class UniversalParser(BaseParser):
    """
    UniversalParser is a parser class for processing various file types using unstructured.
    """

    supported_file_extensions = [
        ".c", ".cc", ".cpp", ".csv", ".cs", ".cxx", ".doc", ".docx", ".eml", ".epub",
        ".go", ".html", ".java", ".js", ".msg", ".odt", ".pdf", ".php", ".ppt",
        ".pptx", ".py", ".rb", ".rtf", ".swift", ".ts", ".tsv", ".txt", ".xml",
        ".yaml", ".yml", ".xlsx"
    ]

    def __init__(self, max_chunk_size: int = 1000, *args, **kwargs):
        """
        Initializes the UniversalParser object.
        """
        self.max_chunk_size = max_chunk_size
        logger.info(f"Initialized UniversalParser with max_chunk_size: {max_chunk_size}")

    async def get_chunks(
        self, filepath: str, metadata: Optional[dict] = None, *args, **kwargs
    ) -> List[Document]:
        """
        Asynchronously processes a file and returns LangChain documents.
        """
        logger.info(f"Starting to process file: {filepath}")
        try:
            # Step 1: Check the file extension and use appropriate partition function
            if filepath.lower().endswith('.pdf'):
                try:
                    logger.info("Using partition_pdf with 'fast' strategy")
                    elements = partition_pdf(filepath, strategy="fast")
                except Exception as e:
                    logger.warning(f"Error with 'fast' strategy: {e}. Falling back to hi_res.")
                    # Fallback to hi_res or another appropriate strategy if needed
                    elements = partition_pdf(filepath, strategy="hi_res")
            else:
                logger.info("Using general partition function")
                elements = partition(filename=filepath)

            # Step 2: Use chunker to get chunks and then convert to dict for addn processing
            logger.info("Chunking elements")
            chunks = chunk_elements(
                elements,
                max_characters=self.max_chunk_size,
                new_after_n_chars=kwargs.get("new_after_n_chars", self.max_chunk_size),
                overlap=kwargs.get("overlap", 0),
            )
            chunks = convert_to_dict(chunks)
            logger.info(f"Created {len(chunks)} chunks")

            # Step 3: Process the chunks using additional_processing
            logger.info("Performing additional processing on chunks")
            _, _, langchain_docs = await additional_processing(
                chunks,
                extend_metadata=kwargs.get("extend_metadata", False),
                additional_metadata=kwargs.get("additional_metadata"),
                replace_table_text=kwargs.get("replace_table_text", False),
                table_text_key=kwargs.get("table_text_key", ""),
                return_langchain_docs=True,
                convert_metadata_keys_to_string=kwargs.get(
                    "convert_metadata_keys_to_string", True
                ),
            )

            logger.info(f"Successfully processed {len(langchain_docs)} documents")
            return langchain_docs
        except Exception as e:
            logger.error(f"Error processing file at {filepath}: {str(e)}")
            return []