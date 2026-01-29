"""
Utility functions for the UniRig FAL App.

This module contains helper functions for:
- Model weight downloading
- File handling
- Seed generation
"""

import os
from pathlib import Path
from typing import List, Optional


def get_seed(seed: int | None) -> int:
    """
    Get seed value, generating a random one if not provided.
    
    Args:
        seed: Optional seed value
        
    Returns:
        The seed to use for generation
    """
    if seed is None:
        import random
        return random.randint(0, 2**32 - 1)
    return seed


def safe_snapshot_download(
    repo_id: str,
    revision: str = "main",
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    local_dir: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """
    Safely download a HuggingFace snapshot with retry logic.
    
    Args:
        repo_id: The HuggingFace repo ID (e.g., "VAST-AI/UniRig")
        revision: The revision/branch to download
        allow_patterns: List of patterns to include
        ignore_patterns: List of patterns to exclude
        local_dir: Local directory to download to
        max_retries: Maximum number of retry attempts
        
    Returns:
        Path to the downloaded snapshot
    """
    from huggingface_hub import snapshot_download
    import time
    
    last_error = None
    for attempt in range(max_retries):
        try:
            return snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_dir=local_dir,
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Download attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    raise RuntimeError(f"Failed to download {repo_id} after {max_retries} attempts: {last_error}")


def validate_file_format(filename: str, supported_formats: List[str]) -> str:
    """
    Validate that a file has a supported format.
    
    Args:
        filename: The filename to validate
        supported_formats: List of supported format extensions (without dots)
        
    Returns:
        The file extension (lowercase, without dot)
        
    Raises:
        ValueError: If the format is not supported
    """
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in supported_formats:
        raise ValueError(
            f"Unsupported format: .{ext}. Supported formats: {', '.join(supported_formats)}"
        )
    return ext


def download_file(url: str, dest_path: Path, timeout: int = 300) -> Path:
    """
    Download a file from URL to local path.
    
    Args:
        url: The URL to download from
        dest_path: The destination path
        timeout: Timeout in seconds
        
    Returns:
        Path to the downloaded file
        
    Raises:
        RuntimeError: If download fails
    """
    import requests
    
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        if not dest_path.exists() or dest_path.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty")
        
        return dest_path
        
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download file: {e}")


def cleanup_directory(path: Path, ignore_errors: bool = True) -> None:
    """
    Clean up a directory and all its contents.
    
    Args:
        path: Path to the directory to clean up
        ignore_errors: Whether to ignore errors during cleanup
    """
    import shutil
    shutil.rmtree(path, ignore_errors=ignore_errors)


def ensure_symlink(source: Path, target: Path) -> None:
    """
    Ensure a symlink exists from target to source.
    
    Args:
        source: The source path (what the symlink points to)
        target: The target path (where the symlink is created)
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return  # Symlink already exists and points to correct location
        target.unlink()  # Remove existing file/symlink
    
    target.symlink_to(source)


class OrderedBaseModel:
    """
    Base model class that maintains field order in JSON output.
    This is a simple placeholder - in production, inherit from Pydantic's BaseModel
    with proper configuration.
    """
    pass
