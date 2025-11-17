# Frequently Asked Questions (FAQ)

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Firebase](#firebase)
- [Models & Performance](#models--performance)
- [Countries & Data](#countries--data)
- [Common Errors](#common-errors)
- [Features & Usage](#features--usage)
- [Contributing](#contributing)

---

## Installation & Setup

### Q: Do I need a GPU to run CCUB2-Agent?

**A**: Not strictly required, but **highly recommended**.

- **With GPU** (NVIDIA 8GB+ VRAM): 2-5 minutes per image
- **Without GPU** (CPU only): 30-90 minutes per image (10-50x slower)

For CPU-only usage, expect long wait times. Consider using Google Colab or cloud GPU instances.

### Q: How much disk space do I need?

**A**: Minimum **50GB** free space:

- Models: ~20GB (Qwen3-VL, SDXL, FLUX, etc.)
- Data: ~1GB per country (images + indices)
- Cache: ~20GB (HuggingFace cache, intermediate files)
- Room for outputs: ~5GB

### Q: Which operating systems are supported?

**A**: Officially tested on:
- ✅ **Ubuntu 20.04+** (primary development platform)
- ✅ **macOS 11+** (works but slower without NVIDIA GPU)
- ⚠️ **Windows 10/11** (via WSL2 Ubuntu)

Native Windows support not tested. Use WSL2 for best results.

### Q: How long does initialization take?

**A**: Depends on internet speed and whether models are cached:

| Task | First Time | Subsequent |
|------|------------|------------|
| Install dependencies | 5 min | instant |
| Download dataset | 10-30 min | 1 min |
| Download models | Auto (during first generation) | instant |
| Build indices | 5-10 min | 1 min |

**Total first-time setup**: 30-60 minutes

### Q: Can I use this without internet?

**A**: After initial setup, yes (mostly):
- Models are cached locally
- Data is stored locally
- Firebase requires internet (but is optional)

Internet needed only for:
- Initial model downloads
- Firebase sync (optional)
- Getting new data

### Q: What Python version is required?

**A**: Python **3.10 or higher** required.

```bash
python3 --version  # Should show 3.10.x or higher
```

Python 3.9 or lower will cause compatibility issues with some dependencies.

---

## Firebase

### Q: Do I need Firebase to use CCUB2-Agent?

**A**: **No**, Firebase is optional.

- **Without Firebase**: Uses local CSV data (338 Korea images included)
- **With Firebase**: Real-time data sync, access to all countries, job creation

Most features work fine without Firebase.

### Q: How do I get Firebase credentials?

**A**: Email **chans@andrew.cmu.edu** with:

- Subject: "CCUB2 Firebase Access"
- Include: Your name, institution, intended use case
- Expected response time: 1-2 business days

You'll receive:
- `firebase-service-account.json` (admin SDK credentials)
- `.firebase_config.json` (app configuration)

### Q: Where should I put Firebase credentials?

**A**: Place both files in the **project root directory**:

```
ccub2-agent/
├── firebase-service-account.json  ← Here
├── .firebase_config.json           ← Here
├── README.md
├── scripts/
└── ...
```

**Security**: These files are in `.gitignore` and won't be committed.

### Q: How do I verify Firebase is working?

**A**: Run the connection test:

```bash
python scripts/05_utils/test_firebase_connection.py
```

**Expected output**:
- ✅ With credentials: `✓ Firebase connection successful`
- ✗ Without: `✗ Firebase connection failed`

### Q: What if Firebase connection fails?

**A**: Check these common issues:

1. **Credentials missing**: Ensure files are in project root
2. **Invalid JSON**: Open files and verify valid JSON format
3. **Expired credentials**: Contact admin for updated credentials
4. **Firewall/proxy**: Ensure port 443 (HTTPS) is open
5. **Internet down**: Firebase requires internet connection

System will fall back to local CSV data if Firebase fails.

---

## Models & Performance

### Q: Which models are supported?

**A**: CCUB2-Agent supports **6+ models** with universal adapter pattern:

**Text-to-Image (T2I)**:
- `sdxl` - Stable Diffusion XL (fast, reliable)
- `flux` - FLUX.1-dev (best quality, slower)
- `sd35` - SD 3.5 Medium (balanced)
- `gemini` - Gemini 2.5 Flash Image (API-based, photorealistic)

**Image-to-Image (I2I)**:
- `qwen` - Qwen-Image-Edit-2509 (**recommended for cultural accuracy**)
- `flux` - FLUX ControlNet (style preservation)
- `sdxl` - SDXL InstructPix2Pix (fast)
- `sd35` - SD 3.5 Medium (versatile)
- `gemini` - Gemini API (photorealistic)

### Q: Which model should I use?

**A**: For best cultural accuracy:

- **T2I**: `flux` (quality) or `sd35` (speed)
- **I2I**: `qwen` (best at preserving cultural details)

For fastest results:
- **T2I**: `sd35`
- **I2I**: `sdxl`

### Q: How much VRAM do I need?

**A**: Depends on quantization mode:

| Mode | VRAM | Quality | Speed |
|------|------|---------|-------|
| 4-bit | 8GB+ | Good | Fast |
| 8-bit | 12GB+ | Better | Medium |
| Full precision | 24GB+ | Best | Slower |

**Recommendation**: Use 4-bit mode (`--load-in-4bit`) for 8GB GPUs.

### Q: Can I run this on multiple GPUs?

**A**: Not currently supported out-of-the-box.

Workaround: Run separate processes on different GPUs:
```bash
CUDA_VISIBLE_DEVICES=0 python script.py &  # GPU 0
CUDA_VISIBLE_DEVICES=1 python script.py &  # GPU 1
```

### Q: How can I speed up generation?

**A**: Several options:

1. **Use smaller models**: `sd35` instead of `flux`
2. **Enable quantization**: `--load-in-4bit`
3. **Reduce iterations**: `--max-iterations 2`
4. **Lower resolution**: `--width 512 --height 512`
5. **Use GPU**: Much faster than CPU
6. **Batch processing**: Process multiple countries sequentially

### Q: Why is the first run so slow?

**A**: Models download automatically on first use (~20GB):

- Qwen3-VL-8B (~16GB)
- SDXL/FLUX/SD3.5 (~7-12GB each)
- CLIP (~1GB)

Models are cached in `~/.cache/huggingface/` for future use.

### Q: Can I use custom models?

**A**: Yes! The universal adapter pattern makes it easy.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guide on adding new models. Requires:
1. Implement `BaseImageEditor` interface
2. Register in `ImageEditingAdapter`
3. Add prompt adaptation strategy

---

## Countries & Data

### Q: Which countries are supported?

**A**: Currently **10 countries** with data:

- 🇰🇷 Korea (338 images)
- 🇯🇵 Japan (201 images)
- 🇨🇳 China (187 images)
- 🇺🇸 USA (542 images)
- 🇳🇬 Nigeria (460 images)
- 🇰🇪 Kenya (95 images)
- 🇲🇽 Mexico (108 images)
- 🇮🇹 Italy (92 images)
- 🇫🇷 France (68 images)
- 🇩🇪 Germany (64 images)

### Q: Can I add support for my country?

**A**: Yes! Two ways:

**1. Use WorldCCUB App** (recommended):
- Download from [worldccub.com](https://worldccub.com)
- Contribute authentic cultural images
- Images automatically integrated into system

**2. Manual Dataset Creation**:
- See [scripts/README.md](scripts/README.md) for data format
- Create dataset JSON with images
- Build indices with provided scripts

### Q: How do I initialize multiple countries?

**A**: Run init script for each country:

```bash
python scripts/01_setup/init_dataset.py --country korea
python scripts/01_setup/init_dataset.py --country japan
python scripts/01_setup/init_dataset.py --country china
```

Or process all at once:
```bash
python scripts/build_all_country_indices.py --countries korea japan china
```

### Q: What categories are available?

**A**: Common categories:

- `traditional_clothing`
- `food`
- `architecture`
- `festivals`
- `art`
- `general` (no specific category)

Categories are flexible - you can use any category when generating.

---

## Common Errors

### Q: "ModuleNotFoundError: No module named 'transformers'"

**A**: Dependencies not installed. Run:

```bash
pip install -r requirements.txt
```

Ensure your virtual environment is activated first.

### Q: "CUDA out of memory"

**A**: GPU VRAM insufficient. Solutions:

1. **Use 4-bit quantization**: `--load-in-4bit`
2. **Close other GPU programs**: Check with `nvidia-smi`
3. **Reduce batch size**: Use default settings
4. **Use smaller model**: Try `sd35` instead of `flux`
5. **Restart Python**: Clear GPU memory

### Q: "RuntimeError: Expected all tensors to be on the same device"

**A**: Model loading issue. Try:

```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Then run again
python your_script.py
```

### Q: "FileNotFoundError: data/country_packs/korea/approved_dataset.json"

**A**: Data not initialized. Run:

```bash
python scripts/01_setup/init_dataset.py --country korea
```

### Q: "ImportError: cannot import name 'CLIPImageRAG'"

**A**: Using wrong import path. Ensure you're in project root:

```bash
cd /path/to/ccub2-agent
python scripts/your_script.py
```

### Q: Models downloading very slowly

**A**: HuggingFace downloads can be slow. Options:

1. **Wait patiently**: First-time only, cached afterwards
2. **Use faster internet**: University/corporate networks often faster
3. **Download manually**: Use `huggingface-cli download`
4. **Use mirror**: Set `HF_ENDPOINT` environment variable

### Q: "ValueError: Prompt is too long (512 tokens)"

**A**: FLUX has 512 token limit. Solutions:

1. **Use shorter prompt**: Be concise
2. **Use different model**: SDXL/SD3.5 have higher limits
3. **Edit prompt adapter**: Modify in `ccub2_agent/modules/prompt_adapter.py`

---

## Features & Usage

### Q: How does cultural bias correction work?

**A**: 4-step process:

1. **Generate** initial image with T2I model
2. **Evaluate** cultural accuracy with VLM (Qwen3-VL)
3. **Retrieve** authentic reference images via CLIP RAG
4. **Edit** image with I2I model using references
5. **Repeat** until cultural score ≥ 8/10

### Q: What is the VLM scoring system?

**A**: Dual 1-10 scale:

- **Cultural Score** (1-10): Authenticity of cultural elements
- **Prompt Score** (1-10): Alignment with original prompt

**Threshold**: Cultural score < 8 triggers iterative refinement.

### Q: How many iterations does it run?

**A**: Default maximum: **5 iterations**

Stops early if:
- Cultural score ≥ 8 AND no issues detected
- Maximum iterations reached

Configure with: `--max-iterations N`

### Q: Can I use this programmatically (Python API)?

**A**: Yes! Example:

```python
from ccub2_agent.modules.vlm_detector import create_vlm_detector
from ccub2_agent.adapters.image_editing_adapter import create_adapter

# Initialize components
vlm = create_vlm_detector(...)
adapter = create_adapter(model_type="qwen")

# Generate and evaluate
image = adapter.generate("Traditional Korean hanbok", 1024, 1024)
cultural_score, prompt_score = vlm.score_cultural_quality(image, ...)
```

See [scripts/](scripts/) for complete examples.

### Q: Does it work for text rendering (e.g., Korean/Chinese characters)?

**A**: Partially. Current models struggle with non-Latin scripts.

**Workaround**: Focus on visual cultural elements (clothing, architecture, food) rather than text.

### Q: Can I use this offline?

**A**: After initial setup, mostly yes:
- ✅ Image generation (models cached)
- ✅ VLM evaluation (model cached)
- ✅ RAG search (indices local)
- ❌ Firebase sync (requires internet)
- ❌ Downloading new models (requires internet)

---

## Contributing

### Q: How can I contribute?

**A**: Several ways:

1. **Add cultural data**: Via WorldCCUB app or directly
2. **Report bugs**: Open GitHub issues
3. **Fix bugs**: Submit pull requests
4. **Add features**: New models, countries, categories
5. **Improve docs**: Fix typos, add examples
6. **Share results**: Show what you've created!

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Q: Do I need to sign a CLA?

**A**: No Contributor License Agreement required. Just follow the contribution guidelines.

### Q: Where can I discuss ideas?

**A**: Use GitHub Discussions for:
- Feature requests
- Questions
- Showcase your results
- General discussion

---

## Still Have Questions?

- **Read the docs**: [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md)
- **Open an issue**: [GitHub Issues](https://github.com/cmubig/ccub2-agent/issues)
- **Start a discussion**: [GitHub Discussions](https://github.com/cmubig/ccub2-agent/discussions)
- **Email maintainer**: chans@andrew.cmu.edu

---

**Last Updated**: November 2025
