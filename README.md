# SignSpeak

**Text-to-sign-language video platform using NLP grammar conversion and motion capture synthesis.**

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://dataedd.github.io/SignSpeak)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Try the Live Demo →](https://dataedd.github.io/SignSpeak)**

SignSpeak translates English text into American Sign Language (ASL) video output. It applies research-backed ASL grammar transformations, maps words to a verified sign database, and composites sign clips into smooth video sequences.

---

## Features

- **NLP-Powered Grammar Conversion** -- Applies ASL-specific grammar rules: Time-Topic-Comment ordering, article/be-verb removal, WH-question reordering, negation placement
- **Sign Video Database** -- 100+ verified ASL signs with metadata, quality scoring, and a verification workflow
- **Video Compositing Pipeline** -- Stitches individual sign clips with crossfade transitions into continuous sequences
- **REST API** -- FastAPI backend serving translation, video streaming, and sign management endpoints
- **React Web Frontend** -- Dark-themed UI with text input, ASL gloss display, and video playback
- **3D Avatar Rendering** -- SMPL-X based body model animation from motion capture data
- **CLI Tool** -- Command-line interface for translation, sign management, and database operations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│               (Text Input → Video Playback)                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP/REST
┌──────────────────────────▼──────────────────────────────────────┐
│                      FastAPI Backend                             │
│         POST /api/translate  →  { video_url, glosses }          │
│         GET  /api/videos/:id →  MP4 stream                      │
│         GET  /api/health     →  { status }                      │
├─────────────┬──────────────┬──────────────┬─────────────────────┤
│ Translation │   Database   │    Video     │      Avatar         │
│   Package   │   Package    │   Package    │     Package         │
│             │              │              │                     │
│ Grammar     │ Sign Store   │ Compositor   │ SMPL-X Renderer     │
│ Rules       │ Importer     │ Transitions  │ Motion Blender      │
│ Gloss       │ Verifier     │ Clip Manager │ Pose Extractor      │
│ Converter   │ Search       │ Exporter     │                     │
└─────────────┴──────────────┴──────────────┴─────────────────────┘
                           │
              ┌────────────▼────────────┐
              │      Sign Database      │
              │  (Verified Video Clips) │
              └─────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- FFmpeg (for video processing)

### Backend

```bash
# Clone the repository
git clone https://github.com/DataEdd/SignSpeak.git
cd SignSpeak

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[api,cli]"

# Download data (follow printed instructions)
python scripts/download_data.py

# Start the API server
python -m packages.api
# → API running at http://localhost:8000
# → Docs at http://localhost:8000/docs
```

### Frontend

```bash
cd apps/web
npm install
npm run dev
# → Frontend running at http://localhost:5173
```

### CLI

```bash
# Interactive translation
python translate.py --interactive

# Translate a sentence
python translate.py "Hello, how are you?"

# Show grammar transformation (no video)
python translate.py --grammar "Where did you go yesterday?"

# Demo mode
python demo.py "Thank you"
```

## API Reference

### Translation

```http
POST /api/translate
Content-Type: application/json

{
  "text": "Hello, how are you?",
  "options": {
    "speed": "normal",
    "format": "mp4"
  }
}
```

**Response:**
```json
{
  "glosses": ["HELLO", "HOW", "YOU"],
  "video_url": "/api/videos/abc123.mp4",
  "confidence": 0.92,
  "quality": "high",
  "missing_signs": []
}
```

### Video Streaming

```http
GET /api/videos/{video_id}  →  MP4 stream
```

### Health Check

```http
GET /api/health  →  { "status": "healthy", "version": "2.0.0" }
```

### Sign Management

```http
GET  /api/signs              # List verified signs
GET  /api/signs/{gloss}      # Get sign details
POST /api/signs              # Add new sign
PUT  /api/signs/{gloss}/verify  # Verify a pending sign
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **NLP / Grammar** | Custom ASL grammar engine (rule-based + research-backed) |
| **Backend** | Python 3.10+, FastAPI, Pydantic, OpenCV |
| **Frontend** | React 18, Vite, Axios |
| **Video** | OpenCV, FFmpeg, NumPy |
| **3D Rendering** | SMPL-X, PyRender, Trimesh |
| **Testing** | pytest, pytest-asyncio, pytest-cov |
| **Data** | WLASL dataset, How2Sign corpus |

## Project Structure

```
SignSpeak/
├── packages/
│   ├── core/           # Shared types, config, utilities
│   ├── translation/    # English → ASL grammar conversion
│   ├── database/       # Sign storage, import, verification
│   ├── video/          # Clip compositing, transitions, export
│   ├── avatar/         # SMPL-X 3D avatar rendering
│   └── api/            # FastAPI REST endpoints
├── apps/
│   ├── cli/            # Command-line interface
│   └── web/            # React frontend
├── data/
│   ├── signs/          # Verified sign video clips + metadata
│   └── wlasl_mapping.json
├── scripts/            # Data download and setup utilities
├── research/           # ASL grammar research and bibliography
├── translate.py        # Main translation entry point
├── demo.py             # Interactive demo script
└── pyproject.toml      # Project configuration
```

## ASL Grammar Rules

SignSpeak implements research-backed ASL grammar transformations:

| Rule | English | ASL Gloss |
|------|---------|-----------|
| **Article Removal** | *The* cat sat | CAT SIT |
| **Be-Verb Removal** | She *is* happy | SHE HAPPY |
| **Time-First** | I go *tomorrow* | TOMORROW I GO |
| **WH-Movement** | *Where* do you live? | YOU LIVE WHERE |
| **Negation** | I don't understand | I UNDERSTAND NOT |
| **Verb Simplification** | She *was running* | SHE RUN |

Based on research from Valli & Lucas (2000), Baker & Cokely (1980), and others. See [`research/bibliography.md`](research/bibliography.md) for full citations.

## Data Setup

SignSpeak requires external datasets not included in the repository:

1. **WLASL** (Word-Level ASL) -- Video clips of individual signs
   - Source: [dxli94.github.io/WLASL](https://dxli94.github.io/WLASL/)
   - Place in: `data/signs/verified/<GLOSS>/video.mp4`

2. **SMPL-X** (optional, for 3D avatar) -- Body model parameters
   - Source: [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/)
   - Place in: `data/models/smplx/`

Run `python scripts/download_data.py` for full setup instructions.

## Testing

```bash
# Run all tests
pytest packages/*/tests apps/*/tests

# Run specific package tests
pytest packages/translation/tests -v

# With coverage
pytest --cov=packages packages/*/tests
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/new-signs`)
3. Add tests for new functionality
4. Run the test suite (`pytest`)
5. Submit a pull request

## License

[MIT](LICENSE)
