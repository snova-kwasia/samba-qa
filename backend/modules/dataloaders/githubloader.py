import os
import re
from typing import Dict, Iterator, List

from git import Repo

from backend.logger import logger
from backend.modules.dataloaders.loader import BaseDataLoader
from backend.types import DataIngestionMode, DataPoint, DataSource, LoadedDataPoint
from urllib.parse import urlparse, urljoin


class GithubLoader(BaseDataLoader):
    """
    Load data from a github repository
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
        Loads data from a Git repository specified by the given source URI. [supports public repository for now]
        """
        if not self.is_valid_github_repo_url(data_source.uri):
            raise Exception("Invalid Github repo URL")

        # Clone the specified GitHub repository to the destination directory.
        logger.info("Cloning repo: %s", data_source.uri)
        Repo.clone_from(data_source.uri, dest_dir)
        logger.info("Git repo cloned successfully")

        # Process the cloned repository files
        process_repo_files(dest_dir, data_source.uri)
        logger.info("Repository files processed and updated")

        loaded_data_points: List[LoadedDataPoint] = []
        for root, d_names, f_names in os.walk(dest_dir):
            for f in f_names:
                if f.startswith("."):
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, dest_dir)
                file_ext = os.path.splitext(f)[1]
                data_point = DataPoint(
                    data_source_fqn=data_source.fqn,
                    data_point_uri=rel_path,
                    data_point_hash=f"{os.path.getsize(full_path)}",
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
                        file_extension=file_ext,
                    )
                )
                if len(loaded_data_points) >= batch_size:
                    yield loaded_data_points
                    loaded_data_points.clear()
        yield loaded_data_points

    def is_valid_github_repo_url(self, url):
        """
        Checks if the provided URL is a valid GitHub repository URL.

        Args:
            url (str): The URL to be checked.

        Returns:
            bool: True if the URL is a valid GitHub repository URL, False otherwise.
        """
        pattern = r"^(?:https?://)?github\.com/[\w-]+/[\w.-]+/?$"
        return re.match(pattern, url) is not None


def generate_github_url(repo_url: str, file_path: str, is_directory: bool = False, anchor: str = None) -> str:
    """
    Generate a GitHub-style URL for a file, directory, or section within a repository.

    Args:
        repo_url (str): The base URL of the GitHub repository.
        file_path (str): The relative path of the file or directory within the repository.
        is_directory (bool): Whether the path is a directory (default: False).
        anchor (str): The anchor (section) within a file, if applicable.

    Returns:
        str: The GitHub-style URL for the file, directory, or section.
    """
    # Parse the repository URL
    parsed_url = urlparse(repo_url)
    
    # Remove .git suffix if present
    path = parsed_url.path.rstrip('/')
    if path.endswith('.git'):
        path = path[:-4]
    
    # Extract the owner and repository name from the path
    path_parts = path.strip('/').split('/')
    if len(path_parts) < 2:
        raise ValueError("Invalid repository URL")
    owner, repo = path_parts[:2]
    
    # Handle special cases for README and root directory
    if file_path in ['', '.', './'] or file_path.lower() == 'readme.md':
        if anchor:
            return f"https://github.com/{owner}/{repo}?tab=readme-ov-file#{anchor}"
        else:
            return f"https://github.com/{owner}/{repo}"

    # Determine the appropriate base URL for files or directories
    if is_directory:
        base_url = f"https://github.com/{owner}/{repo}/tree/main/"
    else:
        base_url = f"https://github.com/{owner}/{repo}/blob/main/"
    
    # Join the base URL with the file path
    full_url = urljoin(base_url, file_path.lstrip('/'))
    
    # Add anchor if present
    if anchor:
        full_url += f"#{anchor}"
    
    return full_url

def process_repo_files(repo_dir: str, repo_url: str) -> None:
    """
    Process files in the repository, adding source URLs and updating relative links.
    Skip non-text files and handle errors gracefully.
    """
    for root, dirs, files in os.walk(repo_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_dir)
            
            if not is_text_file(file_path):
                logger.info(f"Skipping non-text file: {rel_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Generate the GitHub URL for this file
                github_url = generate_github_url(repo_url, rel_path)
                new_content = f"Source: {github_url}\n\n{content}"
                
                # Update relative links
                def replace_link(match):
                    link = match.group(1)
                    if link.startswith(('http://', 'https://')):
                        return match.group(0)  # Already an absolute link
                    else:
                        # Split the link into path and anchor
                        link_parts = link.split('#', 1)
                        link_path = link_parts[0]
                        anchor = link_parts[1] if len(link_parts) > 1 else None
                        
                        # Construct the full path for the linked file
                        full_path = os.path.normpath(os.path.join(os.path.dirname(rel_path), link_path))
                        is_dir = os.path.isdir(os.path.join(repo_dir, full_path))
                        
                        return f'({generate_github_url(repo_url, full_path, is_directory=is_dir, anchor=anchor)})'
                
                new_content = re.sub(r'\(((?!#)[^)]+)\)', replace_link, new_content)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                logger.info(f"Processed file: {rel_path}")
            
            except UnicodeDecodeError:
                logger.warning(f"Unable to read file as text: {rel_path}")
            except IOError as e:
                logger.error(f"IO error processing file {rel_path}: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error processing file {rel_path}: {str(e)}")

# Helper function to check if a file is likely to be a text file
def is_text_file(file_path: str) -> bool:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)
        return True
    except UnicodeDecodeError:
        return False