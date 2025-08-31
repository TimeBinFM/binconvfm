# Time Series Preprocessing Library

A comprehensive library for building streaming data pipelines for time series datasets, providing tools for downloading, transforming, and combining time series data in real-time.

## Features

- **Streaming Dataset Transforms**: Build composable data pipelines that process time series data as it streams
- **Dataset Downloaders**: Download datasets from Hugging Face Hub with caching support
- **Synthetic Data Generation**: Generate synthetic time series data for testing and development
- **Flexible Data Pipeline Builder**: Chain multiple transforms together using a fluent builder pattern
- **PyTorch Integration**: Full compatibility with PyTorch's IterableDataset interface
- **Type Safety**: Comprehensive type annotations for better development experience

## Installation

Using poetry:
```bash
poetry install
```

Using pip:
```bash
pip install .
```

## Core Components

### Dataset Transforms

The library provides a rich set of streaming dataset transforms that can be chained together:

#### Basic Transforms
- **`TransformingDataset`**: Apply arbitrary functions to each item in a dataset
- **`BatchingIterableDataset`**: Group items into batches with configurable batch sizes
- **`UnbatchingIterableDataset`**: Flatten batched datasets back to individual items
- **`SlidingWindowIterableDataset`**: Create sliding windows over time series data

#### Combining Transforms
- **`ConcatDataset`**: Concatenate multiple datasets sequentially
- **`CombiningDataset`**: Combine multiple datasets using custom operations (e.g., element-wise addition)
- **`ProbabilisticMixingDataset`**: Mix datasets with configurable probabilities and seeding

#### Pipeline Builder
- **`Builder`**: Fluent interface for building complex data pipelines

### Dataset Downloaders

- **`HuggingFaceDownloader`**: Download datasets from Hugging Face Hub with automatic caching
- **`GiftEvalWrapperDataset`**: Wrapper for Salesforce's GiftEval pretraining datasets

### Synthetic Data

- **`LinearTrendDataset`**: Generate synthetic time series with linear trends and configurable noise

### Serialization

- **`serialize_tensor_stream`**: Save tensor streams to disk in sharded format
- **`SerializedTensorDataset`**: Load serialized tensor streams with lazy loading support

## Usage Examples

### Basic Transform Pipeline

```python
from preprocessing.transform.dataset_builder import Builder
from preprocessing.transform.batching_dataset import BatchingIterableDataset
from preprocessing.transform.transforming_dataset import TransformingDataset

# Create a simple pipeline
pipeline = (
    Builder(your_dataset)
    .batch(batch_size=32)
    .map(lambda x: x * 2)  # Double all values
    .build()
)

# Iterate over the pipeline
for batch in pipeline:
    print(batch.shape)  # (32, sequence_length, features)
```

### Combining Multiple Datasets

```python
from preprocessing.transform.combining_dataset import CombiningDataset

# Combine two datasets element-wise
def add_operation(x, y):
    return x + y

combined = CombiningDataset([dataset1, dataset2], op=add_operation)

for result in combined:
    print(result)  # x + y for each pair
```

### Concatenating Datasets

```python
from preprocessing.transform.concat_dataset import ConcatDataset

# Concatenate datasets sequentially
concatenated = ConcatDataset([dataset1, dataset2, dataset3])

for item in concatenated:
    # Items from dataset1, then dataset2, then dataset3
    print(item)
```

### Probabilistic Mixing

```python
from preprocessing.transform.probabilistic_mixing_dataset import ProbabilisticMixingDataset

# Mix datasets with custom probabilities
mixed = ProbabilisticMixingDataset(
    datasets={"train": train_data, "val": val_data},
    probabilities={"train": 0.8, "val": 0.2},
    seed=42  # For reproducibility
)

for item in mixed:
    # 80% chance from train, 20% chance from val
    print(item)
```

### Sliding Windows

```python
from preprocessing.transform.sliding_window_dataset import SlidingWindowIterableDataset

# Create sliding windows over time series
windowed = SlidingWindowIterableDataset(
    dataset=your_dataset,
    window_size=100,
    step=50
)

for window in windowed:
    print(window.shape)  # (100, features)
```

### Downloading Datasets

```python
from preprocessing.downloader.huggingface import HuggingFaceDownloader
from preprocessing.config import DatasetConfig

# Configure dataset download
config = DatasetConfig(
    name="air-passengers",
    repo_id="duol/airpassengers",
    files=["AP.csv"],
    cache_dir="data/cache"
)

# Download dataset
downloader = HuggingFaceDownloader(config)
data = downloader.download()
```

### Synthetic Data Generation

```python
from preprocessing.synthetic.linear_trend import LinearTrendDataset

# Generate synthetic time series
synthetic_data = LinearTrendDataset(
    sequence_length=1000,
    num_sequences=100,
    trend_slope=0.01,
    noise_std=0.1,
    seed=42
)

for sequence in synthetic_data:
    print(sequence.shape)  # (1000, 1)
```

### Serialization

```python
from preprocessing.serialization.serialize import serialize_tensor_stream
from preprocessing.serialization.deserialize import SerializedTensorDataset

# Save dataset to disk
serialize_tensor_stream(
    dataset=your_dataset,
    output_dir="data/serialized",
    max_tensors_per_file=1000
)

# Load dataset from disk
loaded_dataset = SerializedTensorDataset(
    filepaths=["data/serialized/shard_00000.pt"],
    lazy=True  # Load on-demand
)
```

## Project Structure

```
preprocessing/
├── common/              # Common types and utilities
│   └── tensor_dataset.py
├── config/              # Configuration management
│   ├── __init__.py
│   └── examples/
├── downloader/          # Dataset downloaders
│   ├── huggingface.py
│   └── gift_eval.py
├── serialization/       # Data serialization
│   ├── serialize.py
│   └── deserialize.py
├── synthetic/           # Synthetic data generation
│   └── linear_trend.py
└── transform/           # Dataset transforms
    ├── batching_dataset.py
    ├── combining_dataset.py
    ├── concat_dataset.py
    ├── dataset_builder.py
    ├── probabilistic_mixing_dataset.py
    ├── sliding_window_dataset.py
    ├── transforming_dataset.py
    └── unbatching_dataset.py
```

## Development

### Setup Development Environment
```bash
poetry install --with dev
```

### Run Tests
```bash
poetry run pytest
```

### Run Tests with Coverage
```bash
poetry run pytest --cov=preprocessing
```

## Key Design Principles

1. **Streaming First**: All transforms work with streaming data, enabling processing of large datasets that don't fit in memory
2. **Composability**: Transforms can be easily chained together using the Builder pattern
3. **Type Safety**: Comprehensive type annotations for better development experience
4. **PyTorch Integration**: Full compatibility with PyTorch's data loading ecosystem
5. **Reproducibility**: Built-in support for random seeds and deterministic operations
6. **Flexibility**: Support for custom operations and transformations

## License

MIT License - see LICENSE file for details 
