"""Hugging Face dataset downloader."""

from typing import Optional, Dict, Any, List, Union
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
import numpy as np

from ..config import DatasetConfig


class HuggingFaceDownloader:
    """Download datasets from Hugging Face Hub."""
    
    # Predefined dataset mappings
    DATASET_MAPPINGS = {
        "UCR": "timeseriesAI/UCR",
        "M4": "timeseriesAI/M4",
    }
    
    def __init__(self, config: DatasetConfig):
        """Initialize the downloader.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download(self) -> Dict[str, np.ndarray]:
        """Download the dataset from Hugging Face Hub.
        
        Returns:
            dict: Dictionary containing the downloaded arrays
            
        Raises:
            ValueError: If no valid files are found to download
            RuntimeError: If download fails
        """
        # Get repo ID and files to download
        repo_id = self._get_repo_id()
        files_to_download = self._get_files_to_download(repo_id)
        
        if not files_to_download:
            raise ValueError(
                f"No files found to download from repo '{repo_id}'. "
                f"Please check the repository and file specifications."
            )
        
        # Download each file
        downloaded_data = {}
        for filename in files_to_download:
            file_path = self._download_file(repo_id, filename)
            if file_path.suffix == '.npz':
                data = self._load_cached_dataset(file_path)
                downloaded_data.update(data)
            else:
                # For non-npz files, store the file path
                downloaded_data[filename] = str(file_path)
                
        return downloaded_data
    
    def _get_repo_id(self) -> str:
        """Get the repository ID to download from.
        
        Returns:
            Repository ID string
        """
        if self.config.repo_id:
            return self.config.repo_id
            
        # Check predefined mappings
        if self.config.name in self.DATASET_MAPPINGS:
            return self.DATASET_MAPPINGS[self.config.name]
            
        # Default to time-series organization
        return f"time-series/{self.config.name}"
    
    def _get_files_to_download(self, repo_id: str) -> List[str]:
        """Get list of files to download.
        
        Args:
            repo_id: Repository ID
            
        Returns:
            List of filenames to download
        """
        if self.config.files:
            # Handle both string and list inputs
            if isinstance(self.config.files, str):
                return [self.config.files]
            return self.config.files
            
        # If no specific files are specified, try to find appropriate files
        try:
            all_files = list_repo_files(
                repo_id,
                revision=self.config.revision,
                repo_type="dataset"
            )
            
            if self.config.subset:
                # Look for subset-specific files
                subset_files = [
                    f for f in all_files 
                    if self.config.subset in f and f.endswith('.npz')
                ]
                if subset_files:
                    return subset_files
                    
            # Default to any .npz files
            npz_files = [f for f in all_files if f.endswith('.npz')]
            if npz_files:
                return npz_files
                
            # If no .npz files found, return all files
            return all_files
            
        except Exception as e:
            # If listing files fails, try default filename
            if self.config.subset:
                return [f"{self.config.subset}.npz"]
            return ["data.npz"]
    
    def _download_file(self, repo_id: str, filename: str) -> Path:
        """Download a single file from the repository.
        
        Args:
            repo_id: Repository ID
            filename: Name of file to download
            
        Returns:
            Path to downloaded file
            
        Raises:
            RuntimeError: If download fails
        """
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.cache_dir),
                repo_type="dataset",
                revision=self.config.revision
            )
            return Path(downloaded_path)
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to download file '{filename}' from repo '{repo_id}': {str(e)}"
            )
    
    def _load_cached_dataset(self, path: Path) -> Dict[str, np.ndarray]:
        """Load a cached .npz dataset.
        
        Args:
            path: Path to the cached dataset
            
        Returns:
            dict: Dictionary containing the dataset arrays
        """
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    
    @staticmethod
    def list_available_datasets() -> Dict[str, Any]:
        """List available time series datasets on Hugging Face Hub.
        
        Returns:
            dict: Dictionary of available datasets and their metadata
        """
        return {
            "UCR": {
                "repo_id": "timeseriesAI/UCR",
                "subsets": ["ECG200", "GunPoint", "CBF"],
                "description": "UCR Time Series Classification Archive"
            },
            "M4": {
                "repo_id": "timeseriesAI/M4",
                "subsets": ["Hourly", "Daily", "Weekly"],
                "description": "M4 Forecasting Competition Dataset"
            }
        } 