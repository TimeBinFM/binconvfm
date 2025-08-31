"""CLI script for downloading time series datasets."""

import typer
from pathlib import Path
import hydra
from omegaconf import DictConfig

from ..preprocessing.config import DatasetConfig
from ..preprocessing.downloader.huggingface import HuggingFaceDownloader

app = typer.Typer()


@app.command()
def list_datasets():
    """List available datasets and their subsets."""
    datasets = HuggingFaceDownloader.list_available_datasets()
    
    typer.echo("Available datasets:")
    for name, info in datasets.items():
        typer.echo(f"\n{name}:")
        typer.echo(f"  Description: {info['description']}")
        typer.echo("  Available subsets:")
        for subset in info["subsets"]:
            typer.echo(f"    - {subset}")


@app.command()
def download(
    dataset: str = typer.Argument(..., help="Name of the dataset to download"),
    subset: str = typer.Option(None, help="Specific subset of the dataset"),
    cache_dir: str = typer.Option(
        "data/cache",
        help="Directory to cache downloaded datasets"
    )
):
    """Download a dataset from Hugging Face Hub."""
    try:
        config = DatasetConfig(
            name=dataset,
            subset=subset,
            cache_dir=cache_dir
        )
        
        downloader = HuggingFaceDownloader(config)
        
        typer.echo(f"Downloading dataset: {dataset}")
        if subset:
            typer.echo(f"Subset: {subset}")
            
        data = downloader.download()
        
        typer.echo("Download complete!")
        typer.echo(f"Data shape: {data['data'].shape}")
        if 'targets' in data:
            typer.echo(f"Targets shape: {data['targets'].shape}")
            
    except Exception as e:
        typer.echo(f"Error downloading dataset: {str(e)}", err=True)
        raise typer.Exit(1)


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for the download script."""
    app() 
