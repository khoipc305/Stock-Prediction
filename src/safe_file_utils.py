"""
Utility functions for safe file operations in OneDrive-synced folders.

OneDrive sync can cause timeout errors when reading or writing files directly.
These utilities use temporary file copies or direct operations to avoid sync conflicts.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Union, Any
import matplotlib.pyplot as plt
import pandas as pd
import time


def safe_savefig(fig: plt.Figure, filepath: Union[str, Path], **kwargs) -> None:
    """
    Safely save a matplotlib figure to avoid OneDrive timeout issues.
    
    Args:
        fig: Matplotlib figure object
        filepath: Destination path for the figure
        **kwargs: Additional arguments passed to fig.savefig()
    
    Example:
        >>> import matplotlib.pyplot as plt
        >>> from src.safe_file_utils import safe_savefig
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> safe_savefig(fig, '../reports/figures/myplot.png', dpi=150)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine file format from extension
    suffix = filepath.suffix
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=suffix) as tmp_file:
        tmp_path = tmp_file.name
        fig.savefig(tmp_path, **kwargs)
    
    # Move temp file to final destination
    shutil.move(tmp_path, filepath)


def safe_savefig_plt(filepath: Union[str, Path], **kwargs) -> None:
    """
    Safely save the current pyplot figure to avoid OneDrive timeout issues.
    
    This is a drop-in replacement for plt.savefig() that handles OneDrive sync.
    
    Args:
        filepath: Destination path for the figure
        **kwargs: Additional arguments passed to plt.savefig()
    
    Example:
        >>> import matplotlib.pyplot as plt
        >>> from src.safe_file_utils import safe_savefig_plt
        >>> plt.plot([1, 2, 3])
        >>> safe_savefig_plt('../reports/figures/myplot.png', dpi=150, bbox_inches='tight')
    """
    fig = plt.gcf()  # Get current figure
    safe_savefig(fig, filepath, **kwargs)


def safe_write_text(filepath: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
    """
    Safely write text to a file to avoid OneDrive timeout issues.
    
    Args:
        filepath: Destination path for the file
        content: Text content to write
        encoding: Text encoding (default: 'utf-8')
    
    Example:
        >>> from src.safe_file_utils import safe_write_text
        >>> safe_write_text('../reports/summary.txt', 'Model accuracy: 95%')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=filepath.suffix, encoding=encoding) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(content)
    
    # Move temp file to final destination
    shutil.move(tmp_path, filepath)


def safe_write_binary(filepath: Union[str, Path], content: bytes) -> None:
    """
    Safely write binary data to a file to avoid OneDrive timeout issues.
    
    Args:
        filepath: Destination path for the file
        content: Binary content to write
    
    Example:
        >>> from src.safe_file_utils import safe_write_binary
        >>> safe_write_binary('../data/output.bin', b'\\x00\\x01\\x02')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=filepath.suffix) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(content)
    
    # Move temp file to final destination
    shutil.move(tmp_path, filepath)


def safe_to_csv(df, filepath: Union[str, Path], **kwargs) -> None:
    """
    Safely save a pandas DataFrame to CSV to avoid OneDrive timeout issues.
    
    This is a drop-in replacement for df.to_csv() that handles OneDrive sync.
    
    Args:
        df: Pandas DataFrame to save
        filepath: Destination path for the CSV file
        **kwargs: Additional arguments passed to df.to_csv()
    
    Example:
        >>> import pandas as pd
        >>> from src.safe_file_utils import safe_to_csv
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> safe_to_csv(df, '../reports/data.csv', index=False)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
        tmp_path = tmp_file.name
        df.to_csv(tmp_path, **kwargs)
    
    # Move temp file to final destination
    shutil.move(tmp_path, filepath)


def safe_read_csv(filepath: Union[str, Path], max_retries: int = 3, **kwargs) -> pd.DataFrame:
    """
    Safely read a CSV file to avoid OneDrive timeout issues.
    
    This is a drop-in replacement for pd.read_csv() that handles OneDrive sync.
    Uses retry logic to handle OneDrive sync locks.
    
    Args:
        filepath: Path to the CSV file to read
        max_retries: Maximum number of retry attempts (default: 3)
        **kwargs: Additional arguments passed to pd.read_csv()
    
    Returns:
        pandas DataFrame
    
    Example:
        >>> from src.safe_file_utils import safe_read_csv
        >>> df = safe_read_csv('../data/raw/finviz_news.csv')
    
    Note:
        If this still fails, try:
        1. Pause OneDrive sync temporarily
        2. Move the file outside OneDrive folder temporarily
        3. Use: pd.read_csv(str(filepath).replace('OneDrive', 'OneDrive.old'))
    """
    filepath = Path(filepath)
    
    for attempt in range(max_retries):
        try:
            # Read file content in chunks to avoid OneDrive timeout during copy
            # Then write to temp file and read from there
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
                tmp_path = tmp_file.name
                
                # Read source file in smaller chunks (256KB at a time) with retry
                chunk_size = 256 * 1024  # 256KB chunks
                with open(filepath, 'rb') as src:
                    while True:
                        chunk = src.read(chunk_size)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
            
            # Read from temp file
            df = pd.read_csv(tmp_path, **kwargs)
            
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
            
            return df
            
        except (TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"⚠️  OneDrive timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                # Clean up temp file if it exists
                if 'tmp_path' in locals():
                    Path(tmp_path).unlink(missing_ok=True)
            else:
                print(f"\n❌ Failed to read file after {max_retries} attempts.")
                print(f"OneDrive is blocking access to: {filepath}")
                print("\n💡 Workarounds:")
                print("1. Pause OneDrive sync (right-click OneDrive icon → Pause syncing)")
                print("2. Move file to non-OneDrive location temporarily")
                print("3. Wait for OneDrive to finish syncing the file")
                raise


def safe_read_parquet(filepath: Union[str, Path], max_retries: int = 3, **kwargs) -> pd.DataFrame:
    """
    Safely read a parquet file to avoid OneDrive timeout issues.
    
    This is a drop-in replacement for pd.read_parquet() that handles OneDrive sync.
    Uses retry logic to handle OneDrive sync locks.
    
    Args:
        filepath: Path to the parquet file to read
        max_retries: Maximum number of retry attempts (default: 3)
        **kwargs: Additional arguments passed to pd.read_parquet()
    
    Returns:
        pandas DataFrame
    
    Example:
        >>> from src.safe_file_utils import safe_read_parquet
        >>> df = safe_read_parquet('../data/raw/prices.parquet')
    """
    filepath = Path(filepath)
    
    for attempt in range(max_retries):
        try:
            # Read file content in chunks to avoid OneDrive timeout during copy
            # Then write to temp file and read from there
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.parquet') as tmp_file:
                tmp_path = tmp_file.name
                
                # Read source file in smaller chunks (256KB at a time)
                chunk_size = 256 * 1024  # 256KB chunks
                with open(filepath, 'rb') as src:
                    while True:
                        chunk = src.read(chunk_size)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
            
            # Read from temp file
            df = pd.read_parquet(tmp_path, **kwargs)
            
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
            
            return df
            
        except (TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"⚠️  OneDrive timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                # Clean up temp file if it exists
                if 'tmp_path' in locals():
                    Path(tmp_path).unlink(missing_ok=True)
            else:
                print(f"\n❌ Failed to read file after {max_retries} attempts.")
                print(f"OneDrive is blocking access to: {filepath}")
                print("\n💡 Workarounds:")
                print("1. Pause OneDrive sync (right-click OneDrive icon → Pause syncing)")
                print("2. Move file to non-OneDrive location temporarily")
                print("3. Wait for OneDrive to finish syncing the file")
                raise


def safe_to_parquet(df, filepath: Union[str, Path], **kwargs) -> None:
    """
    Safely save a pandas DataFrame to parquet to avoid OneDrive timeout issues.
    
    This is a drop-in replacement for df.to_parquet() that handles OneDrive sync.
    
    Args:
        df: Pandas DataFrame to save
        filepath: Destination path for the parquet file
        **kwargs: Additional arguments passed to df.to_parquet()
    
    Example:
        >>> import pandas as pd
        >>> from src.safe_file_utils import safe_to_parquet
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> safe_to_parquet(df, '../data/processed/data.parquet', index=False)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.parquet') as tmp_file:
        tmp_path = tmp_file.name
        df.to_parquet(tmp_path, **kwargs)
    
    # Move temp file to final destination
    shutil.move(tmp_path, filepath)
