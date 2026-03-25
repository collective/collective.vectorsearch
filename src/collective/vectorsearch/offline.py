# -*- coding: utf-8 -*-
"""Offline model management for FastEmbed.

This module provides utilities to pre-download embedding models
for use in offline environments.

Usage:
    # From command line (after installing with entry_point)
    vectorsearch-download

    # Or programmatically
    from collective.vectorsearch.offline import download_all_models
    download_all_models()
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("collective.vectorsearch")

# Default cache directory for FastEmbed
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fastembed"

# Models supported by this package
SUPPORTED_MODELS = [
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "MiniLM L6 v2 - lightweight English model (~90 MB)",
        "provider_id": "all-minilm-l6",
        "custom_registration": None,
    },
    {
        "name": "intfloat/multilingual-e5-base",
        "description": "E5 Base Multilingual - 100+ languages (~1.1 GB)",
        "provider_id": "e5-base-multilingual",
        "custom_registration": {
            "dim": 768,
            "pooling": "mean",
            "normalization": True,
            "model_file": "onnx/model.onnx",
            "additional_files": ["sentencepiece.bpe.model"],
        },
    },
]


def _register_custom_model_if_needed(model_name: str) -> None:
    """Register a custom model with FastEmbed if it requires custom registration.

    Looks up the model in SUPPORTED_MODELS and, if it has custom_registration
    config, delegates to model_providers._register_fastembed_custom_model().
    """
    for model_info in SUPPORTED_MODELS:
        if model_info["name"] == model_name and model_info.get("custom_registration"):
            from fastembed.common.model_description import PoolingType

            from collective.vectorsearch.model_providers import (
                _register_fastembed_custom_model,
            )

            reg = model_info["custom_registration"]
            pooling_map = {"mean": PoolingType.MEAN, "cls": PoolingType.CLS}
            _register_fastembed_custom_model(
                model_name=model_name,
                dim=reg["dim"],
                pooling=pooling_map[reg["pooling"]],
                normalization=reg["normalization"],
                model_file=reg["model_file"],
                additional_files=reg.get("additional_files"),
            )
            return


def get_cache_dir() -> Path:
    """Get the FastEmbed cache directory.

    Respects FASTEMBED_CACHE_PATH environment variable if set.

    Returns:
        Path to cache directory
    """
    env_path = os.environ.get("FASTEMBED_CACHE_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_CACHE_DIR


def download_model(model_name: str, cache_dir: Optional[Path] = None) -> Path:
    """Download a FastEmbed model for offline use.

    Args:
        model_name: Model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        cache_dir: Cache directory (default: ~/.cache/fastembed)

    Returns:
        Path to cached model directory

    Raises:
        ImportError: If fastembed is not installed
        Exception: If download fails
    """
    try:
        from fastembed import TextEmbedding
    except ImportError as err:
        raise ImportError(
            "fastembed is not installed. Install it with: pip install fastembed"
        ) from err

    cache_dir = cache_dir or get_cache_dir()
    os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir)

    # Register custom model if needed (e.g., intfloat/multilingual-e5-base
    # is not in FastEmbed's built-in supported list)
    _register_custom_model_if_needed(model_name)

    logger.info(f"Downloading model {model_name} to {cache_dir}")
    print(f"Downloading: {model_name}")
    print(f"Cache directory: {cache_dir}")

    # This triggers the download
    _model = TextEmbedding(model_name=model_name)

    # Return the expected cache path
    model_cache_path = cache_dir / model_name.replace("/", "_")
    print(f"Download complete: {model_cache_path}")

    return model_cache_path


def list_cached_models(cache_dir: Optional[Path] = None) -> List[str]:
    """List all cached FastEmbed models.

    Args:
        cache_dir: Cache directory (default: ~/.cache/fastembed)

    Returns:
        List of cached model names
    """
    cache_dir = cache_dir or get_cache_dir()
    if not cache_dir.exists():
        return []

    # FastEmbed stores models in subdirectories
    models = []
    for item in cache_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            models.append(item.name)

    return sorted(models)


def download_all_models(cache_dir: Optional[Path] = None) -> None:
    """Download all supported models for offline use.

    This is the main entry point for the CLI command.

    Args:
        cache_dir: Cache directory (default: ~/.cache/fastembed)
    """
    print("=" * 60)
    print("collective.vectorsearch - Model Downloader")
    print("=" * 60)
    print()

    cache_dir = cache_dir or get_cache_dir()
    print(f"Cache directory: {cache_dir}")
    print()

    # Check if fastembed is available
    try:
        from fastembed import TextEmbedding  # noqa: F401
    except ImportError:
        print("ERROR: fastembed is not installed.")
        print("Install it with: pip install fastembed")
        sys.exit(1)

    print(f"Found {len(SUPPORTED_MODELS)} supported models:")
    for i, model in enumerate(SUPPORTED_MODELS, 1):
        print(f"  {i}. {model['name']}")
        print(f"     {model['description']}")
    print()

    # Check existing cache
    cached = list_cached_models(cache_dir)
    if cached:
        print("Already cached:")
        for name in cached:
            print(f"  - {name}")
        print()

    # Download each model
    success_count = 0
    for model in SUPPORTED_MODELS:
        print("-" * 60)
        try:
            download_model(model["name"], cache_dir)
            success_count += 1
        except Exception as e:
            print(f"ERROR downloading {model['name']}: {e}")
            logger.error(f"Failed to download {model['name']}: {e}")

    print()
    print("=" * 60)
    print(f"Download complete: {success_count}/{len(SUPPORTED_MODELS)} models")
    print("=" * 60)

    if success_count < len(SUPPORTED_MODELS):
        sys.exit(1)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download embedding models for offline use."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Cache directory for model files. "
            "Defaults to FASTEMBED_CACHE_PATH env var, "
            "or ~/.cache/fastembed if not set."
        ),
    )
    args = parser.parse_args()

    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    download_all_models(cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
