# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete GUI system with React Flow visualization
- Real-time WebSocket updates for pipeline status
- Job proposal and approval system
- Multi-country support (10 countries)
- GPU monitoring and system stats

### Changed
- Upgraded VLM model from Qwen2-VL-7B to Qwen3-VL-8B
- Improved memory management with 4-bit quantization
- Enhanced error handling across all modules

### Fixed
- Schema mismatch in cultural index configuration files
- Korean language content removed from all public-facing code
- TODO comments resolved or clarified

## [2.0.0] - 2025-11-17

### Added
- **Self-Improving Loop**: Automatic data gap detection and Firebase job creation
- **Model-Agnostic Architecture**: Universal adapter pattern for 6+ T2I and I2I models
- **Dual RAG System**: Text KB (FAISS) + CLIP Image RAG for cultural context
- **VLM-Based Evaluation**: Qwen3-VL-8B for accurate cultural scoring (1-10 scale)
- **Reference-Based Editing**: CLIP similarity + semantic keyword matching
- **Multi-Country Support**: Scalable to any country without code changes
- **Firebase Integration**: Real-time data sync with WorldCCUB app
- **Web GUI**: Production-ready interface with pipeline visualization

### Supported Models

**T2I (Text-to-Image):**
- Stable Diffusion XL
- FLUX.1-dev
- SD 3.5 Medium
- Gemini 2.5 Flash Image

**I2I (Image-to-Image):**
- Qwen-Image-Edit-2509 (recommended)
- FLUX ControlNet
- SDXL ControlNet
- SD 3.5 Medium
- Gemini 2.5 Flash Image

### Technical Highlights
- **Zero Hardcoding**: All cultural knowledge retrieved dynamically from RAG
- **Iteration-Aware Scoring**: VLM compares to previous iterations
- **Sequential Fixing**: One issue per iteration for better results
- **Memory Efficient**: 4-bit quantization, sequential CPU offload
- **Fast Indexing**: 89% faster with incremental FAISS updates

### Performance
- Cultural accuracy improved from 30-40% → 70-90%
- Self-improving loop: Data gaps → Jobs → User uploads → Better RAG
- Supports up to 5 iterative refinements per image

## [1.0.0] - 2024-11-04

### Added
- Initial release of CCUB2-Agent
- Basic VLM cultural detection
- CLIP-based image similarity search
- Simple prompt adaptation
- Single-model support (SDXL)
- CSV-based data storage

### Changed
- Foundation architecture established
- Basic pipeline: T2I → VLM → Edit → Evaluate

### Known Issues
- Limited model support
- No self-improving mechanism
- Manual data collection required
- Korean language in documentation

---

## Version History

- **2.0.0** (2025-11-17): Major release with self-improving loop and multi-model support
- **1.0.0** (2024-11-04): Initial release

## Migration Guides

### Migrating from 1.x to 2.0

**Breaking Changes:**
1. Configuration schema updated for cultural indices
2. VLM model changed from Qwen2-VL-7B to Qwen3-VL-8B
3. Firebase integration replaces CSV storage

**Migration Steps:**
1. Update index configuration files to include `model_name` key
2. Rebuild FAISS indices with new schema:
   ```bash
   python scripts/03_indexing/integrate_knowledge_to_rag.py --country korea
   ```
3. Update Firebase credentials configuration
4. Install new dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Deprecation Notices

- **CSV Storage** (deprecated in 2.0.0): Use Firebase integration instead
- **Single-model adapters** (deprecated in 2.0.0): Use universal adapter pattern

## Roadmap

See [docs/FUTURE_WORK.md](docs/FUTURE_WORK.md) for planned features.

---

For detailed information about each release, see the [GitHub Releases](https://github.com/yourusername/ccub2-agent/releases) page.
