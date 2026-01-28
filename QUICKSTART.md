# Quick Start Guide

Get CCUB2-Agent running in 30 minutes or less!

## What is CCUB2-Agent?

CCUB2-Agent is an AI system that automatically detects and corrects cultural biases in AI-generated images. It uses Vision-Language Models (VLM) and Retrieval-Augmented Generation (RAG) to ensure cultural accuracy.

**Example**: If an AI generates a Korean hanbok (traditional clothing) with incorrect collar styles, CCUB2-Agent detects the error and automatically fixes it using authentic reference images.

---

## Prerequisites

Before you begin, ensure you have:

- **Operating System**: Ubuntu 20.04+ or macOS 11+
- **Python**: 3.10 or higher
- **Disk Space**: 50GB free (20GB for models, 1GB for data, rest for cache)
- **GPU** (optional but recommended): NVIDIA with 8GB+ VRAM
- **Internet**: For downloading models (~20GB first time)

### Quick Check

```bash
python3 --version  # Should be 3.10+
nvidia-smi         # Check GPU (optional)
df -h .            # Check free disk space
```

---

## Installation (10 minutes)

### Step 1: Clone Repository

```bash
git clone https://github.com/cmubig/ccub2-agent.git
cd ccub2-agent
```

✅ **Expected**: Repository cloned (~100MB)

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n ccub2 python=3.10
conda activate ccub2

# OR using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

✅ **Expected**: Virtual environment activated

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

⏱️ **Duration**: ~5 minutes
✅ **Expected**: ~100 packages installed, no errors

**Common Issue**: If you see `ERROR: Failed building wheel for...`, try:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 4: Get Firebase Credentials (Optional)

Firebase enables real-time data updates. Skip this for basic testing.

**To enable Firebase**:
1. Email `chans@andrew.cmu.edu` with subject "CCUB2 Firebase Access"
2. Save received files to project root:
   - `firebase-service-account.json`
   - `.firebase_config.json`

**To skip Firebase**: System will work with local CSV data.

### Step 5: Verify Installation

```bash
python scripts/05_utils/test_firebase_connection.py
```

✅ **With Firebase**: `✓ Firebase connection successful`
✅ **Without Firebase**: `✗ Firebase connection failed (using local data)`

---

## First Run (20 minutes)

### Initialize Data for Korea

```bash
python scripts/01_setup/init_dataset.py --country korea
```

⏱️ **First time**: 10-30 minutes (downloads 338 images, ~700MB)
⏱️ **Subsequent runs**: ~1 minute

✅ **Expected output**:
```
✓ Downloaded 338 images
✓ Saved to data/country_packs/korea/
✓ Dataset initialized successfully
```

**What's happening?**:
- Downloads authentic Korean cultural images from Firebase
- Creates local dataset for reference-based editing
- Images include traditional clothing, food, architecture, etc.

### Build Cultural Indices (Optional but Recommended)

```bash
# Build CLIP index for image similarity search
python scripts/indexing/build_clip_image_index.py --country korea

# Build cultural knowledge index for text retrieval
python scripts/indexing/integrate_knowledge_to_rag.py --country korea
```

⏱️ **Duration**: 5-10 minutes
✅ **Expected**: Index files created in `data/clip_index/korea/` and `data/cultural_index/korea/`

**Skip if**: You want to test quickly without full RAG capabilities.

---

## Your First Cultural Image Generation

### Interactive Mode (Easiest)

```bash
python scripts/04_testing/test_model_agnostic_editing.py
```

**You'll be prompted for**:
1. **T2I Model**: Choose `sd35` (fastest) or `flux` (best quality)
2. **I2I Model**: Choose `qwen` (recommended for cultural accuracy)
3. **Country**: Type `korea`
4. **Category**: Type `traditional_clothing` or press Enter for none
5. **Prompt**: Try `"A person wearing traditional Korean hanbok"`

⏱️ **Duration**: 2-5 minutes per image

✅ **Expected output**:
```
Step 0: Generated initial image (initial.png)
VLM Evaluation: Cultural score 6/10 → needs improvement
Step 1: Retrieved reference images from CLIP
Step 1: Editing image with cultural corrections...
Step 1: VLM Evaluation: Cultural score 9/10 ✓ Acceptable!
✓ Pipeline completed successfully
```

**Output location**: `test_outputs/<timestamp>/`

### Command-Line Mode

```bash
python scripts/04_testing/test_model_agnostic_editing.py \
  --prompt "Traditional Korean palace" \
  --country korea \
  --category architecture \
  --t2i-model sd35 \
  --i2i-model qwen \
  --max-iterations 3
```

---

## View Your Results

Generated images are saved to `test_outputs/<timestamp>/`:

```bash
ls -lh test_outputs/*/
# step_0_initial.png     # Original AI-generated image
# step_1_edited.png      # After cultural correction
# summary.json           # VLM scores and detected issues
```

**Compare Before/After**:
- `step_0_initial.png`: Original generation (may have cultural inaccuracies)
- `step_1_edited.png`: Culturally corrected version

---

## Troubleshooting

### "CUDA out of memory"

**Solution**: Use 4-bit quantization to reduce VRAM usage

```bash
python scripts/04_testing/test_model_agnostic_editing.py --load-in-4bit
```

### "Firebase connection failed"

**Solution**: This is normal if you haven't setup Firebase. System will use local CSV data.

### "ModuleNotFoundError"

**Solution**: Ensure virtual environment is activated and dependencies installed

```bash
conda activate ccub2  # or: source venv/bin/activate
pip install -r requirements.txt
```

### Models downloading slowly

**First run**: Models download from HuggingFace (~20GB). This is one-time only.

**Speed up**: Use fast internet or wait patiently. Models are cached for future use.

### "No such file or directory: firebase-service-account.json"

**Solution**: Either get Firebase credentials (email chans@andrew.cmu.edu) or ignore - system works without it.

---

## What's Next?

### Try Different Countries

```bash
# Initialize data for other countries
python scripts/01_setup/init_dataset.py --country japan
python scripts/01_setup/init_dataset.py --country china

# Then use them in generation
python scripts/04_testing/test_model_agnostic_editing.py
# Select japan or china when prompted
```

### Use the GUI (Visual Interface)

```bash
# Start backend
cd gui/backend
python main.py &  # Runs on http://localhost:8000

# Start frontend (new terminal)
cd gui/frontend
npm install
npm run dev  # Opens http://localhost:3000
```

### Add Your Own Data

Contribute authentic cultural images via the WorldCCUB app or contact the team.

### Learn More

- **Full Documentation**: [README.md](README.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **FAQ**: [FAQ.md](FAQ.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **API Reference**: [docs/](docs/)

---

## Performance Tips

### Speed Up Generation

1. **Use smaller models**: `sd35` is faster than `flux`
2. **Enable 4-bit mode**: `--load-in-4bit`
3. **Reduce iterations**: `--max-iterations 2`
4. **Use GPU**: NVIDIA GPU is 10-50x faster than CPU

### Save Disk Space

1. **Clear HuggingFace cache** (after setup):
   ```bash
   rm -rf ~/.cache/huggingface/hub/*.tmp
   ```

2. **Remove test outputs**:
   ```bash
   rm -rf test_outputs/
   ```

### Optimize Quality

1. **Use best models**: `flux` (T2I) + `qwen` (I2I)
2. **Allow more iterations**: `--max-iterations 5`
3. **Use specific categories**: More targeted cultural corrections

---

## Summary

| Task | Command | Duration |
|------|---------|----------|
| Install | `pip install -r requirements.txt` | ~5 min |
| Init data | `python scripts/01_setup/init_dataset.py --country korea` | 10-30 min (first time) |
| Build indices | `python scripts/indexing/build_clip_image_index.py --country korea` | 5-10 min |
| Generate | `python scripts/04_testing/test_model_agnostic_editing.py` | 2-5 min/image |

**Total first-time setup**: 30-60 minutes
**Subsequent usage**: 2-5 minutes per generation

---

## Need Help?

- **Issues**: Check [FAQ.md](FAQ.md) first
- **Bugs**: Open an issue on [GitHub](https://github.com/cmubig/ccub2-agent/issues)
- **Questions**: Email chans@andrew.cmu.edu or open a [Discussion](https://github.com/cmubig/ccub2-agent/discussions)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Happy cultural image generation!** 🎉
