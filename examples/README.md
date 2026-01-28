# CCUB2-Agent Examples

This directory contains example scripts demonstrating various use cases of CCUB2-Agent.

## Prerequisites

Before running these examples, ensure you have:

1. Installed all dependencies: `pip install -r requirements.txt`
2. Initialized dataset: `python scripts/01_setup/init_dataset.py --country korea`
3. Built indices:
   ```bash
   python scripts/indexing/build_clip_image_index.py --country korea
   python scripts/indexing/integrate_knowledge_to_rag.py --country korea
   ```

## Examples Overview

| Example | Description | Difficulty |
|---------|-------------|------------|
| [basic_usage.py](basic_usage.py) | Simple cultural image generation | Beginner |
| [vlm_evaluation.py](vlm_evaluation.py) | Standalone VLM cultural evaluation | Beginner |
| [clip_retrieval.py](clip_retrieval.py) | Retrieve reference images with CLIP | Intermediate |
| [custom_model.py](custom_model.py) | Add a custom image editing model | Advanced |
| [batch_processing.py](batch_processing.py) | Process multiple prompts in batch | Intermediate |

## Running Examples

### Basic Usage

Generate a culturally-accurate image with automatic correction:

```bash
python examples/basic_usage.py
```

**Output**: Image saved to `output/basic_example.png`

### VLM Evaluation

Evaluate an existing image for cultural accuracy:

```bash
python examples/vlm_evaluation.py --image-path my_image.png --country korea
```

**Output**: Cultural and prompt scores (1-10), detected issues

### CLIP Retrieval

Find similar reference images from your dataset:

```bash
python examples/clip_retrieval.py --image-path my_image.png --country korea --top-k 5
```

**Output**: Paths to top-5 most similar reference images

### Custom Model

Learn how to integrate your own T2I or I2I model:

```bash
python examples/custom_model.py
```

**Output**: Demonstrates custom model integration

### Batch Processing

Process multiple prompts efficiently:

```bash
python examples/batch_processing.py --prompts-file prompts.txt --country korea
```

**Output**: Generated images for each prompt

## Common Issues

### GPU Memory Error

If you encounter "CUDA out of memory":

```bash
# Enable 4-bit quantization
export LOAD_IN_4BIT=1
python examples/basic_usage.py
```

### Missing Indices

If you see "Index not found":

```bash
# Rebuild indices
python scripts/indexing/build_clip_image_index.py --country korea
python scripts/indexing/integrate_knowledge_to_rag.py --country korea
```

### Firebase Connection Failed

If Firebase is not configured, system will fall back to local CSV data. Most examples will still work.

## Modifying Examples

All examples are designed to be easily modified for your use case:

1. **Change country**: Update `country="korea"` to your target country
2. **Change models**: Update `model_type="qwen"` to try different models
3. **Adjust parameters**: Modify `max_iterations`, `width`, `height` as needed

## Next Steps

After running these examples:

1. Read [ARCHITECTURE.md](../ARCHITECTURE.md) for system design
2. Check [FAQ.md](../FAQ.md) for common questions
3. See [CONTRIBUTING.md](../CONTRIBUTING.md) to add your own examples

## Need Help?

- Open an issue: https://github.com/cmubig/ccub2-agent/issues
- Email: chans@andrew.cmu.edu
