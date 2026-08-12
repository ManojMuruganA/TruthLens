# TruthLens — AI-Generated Image Detection for Social Media

## Overview

TruthLens is a deep learning-based web application that detects whether images on social media are **real** or **AI-generated**. Users paste an Instagram, Twitter/X, or TikTok URL, and the system extracts the image and classifies it using a fine-tuned Convolutional Neural Network (CNN) and Vision Transformer (ViT).

This project was developed as part of the **Engineering Science Capstone Design 1** course at **Jeonbuk National University**.

---

## Research Findings

| Finding | Detail |
|---------|--------|
| **Lab vs Reality Gap** | CNN scored 95.39% on CIFAKE benchmark but dropped to 65% on real Instagram images — a 30% accuracy gap |
| **Fine-Tuning Works** | After fine-tuning on just 320 social media images, accuracy recovered to **98.75%** |
| **ViT Overconfidence** | Vision Transformer (85M params) labeled everything as AI-generated when undertrained; CNN (845K params) was more reliable |
| **Compression Impact** | Instagram's JPEG 40% compression reduced detection confidence by up to 46%; one AI image flipped to "Real" after compression |

---

## Project Structure

    TruthLens/
    ├── backend/ # Flask API server
    │ ├── app.py # Main Flask application
    │ ├── ai_detector_v2.py # CNN + ViT detection engine (ensemble)
    │ ├── scraper.py # Social media image extraction
    │ ├── config.py # Configuration settings
    │ ├── models.py # SQLite database models
    │ ├── requirements.txt # Python dependencies
    │ └── models/ # Trained model files (download separately)
    ├── frontend/ # React user interface
    │ ├── public/
    │ ├── src/
    │ │ ├── App.jsx # Main React component
    │ │ ├── App.css # Professional minimal styling
    │ │ └── components/ # SearchBar, ResultCard, HistoryPanel
    │ ├── package.json # Node.js dependencies
    │ └── tailwind.config.js # TailwindCSS configuration
    ├── research/ # Training scripts and results
    │ ├── train_cnn.py # CNN training on CIFAKE dataset
    │ ├── train_vit.py # ViT fine-tuning script
    │ ├── finetune_on_social.py # Fine-tune CNN on Instagram images
    │ ├── test_instagram.py # Test models on compressed images
    │ ├── generate_ppt_charts.py # Generate presentation charts
    │ ├── split_social_dataset.py # Dataset splitting utility
    │ ├── download_ai_faces.py # AI face image downloader
    │ └── results/ # Training charts, metrics, confusion matrices
    ├── .gitignore
    └── README.md

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python, Flask, PyTorch, timm |
| **Frontend** | React, TailwindCSS |
| **Models** | Custom CNN (845,602 params), ViT-Base/16 (85,800,194 params) |
| **Dataset** | CIFAKE (120,000 images) + Custom Social Media Dataset (400 images) |
| **Database** | SQLite |

---

## Setup Instructions

### Prerequisites

- **Python 3.9+** installed on your system
- **Node.js 18+** installed on your system
- **Git** installed (for cloning)

### Step 1: Clone the Repository

    git clone https://github.com/ManojMuruganA/TruthLens.git
    cd TruthLens

### Step 2: Download Trained Models

The trained model files are too large for GitHub. Download them from Google Drive:

🔗 [**Download Models from Google Drive**](https://drive.google.com/drive/folders/1BImV1bq1RTMxy6t1mEdgNhsvyrHJTcf2)

Place the downloaded files in:

    TruthLens/backend/models/

Required files:

- `cnn_model.pth` — CNN trained on 100K CIFAKE images (95.39% accuracy)
- `cnn_social_finetuned.pth` — CNN fine-tuned on Instagram images (98.75% accuracy)
- `vit_model.pth` — Vision Transformer fine-tuned (94.50% accuracy)

### Step 3: Setup Backend

    cd backend
    python -m venv venv
    venv\Scripts\Activate        # On Windows
    # source venv/bin/activate   # On Mac/Linux
    pip install -r requirements.txt
    pip install timm
    python app.py

The backend will start at `http://127.0.0.1:5000`.

### Step 4: Setup Frontend

Open a new terminal:

    cd frontend
    npm install
    npm start

The frontend will open at `http://localhost:3000`.

### Step 5: Use the Application

1. Open your browser to `http://localhost:3000`
2. Paste any Instagram, Twitter/X, or TikTok URL
3. Click **Analyze**
4. View the result — **Real** or **AI-Generated** with confidence score

---

## API Endpoints

| **Method** | **Endpoint**            | **Description**          |
| ---------- | ----------------------- | ------------------------ |
| `GET`      | `/api/health`           | Health check             |
| `POST`     | `/api/detect`           | Submit URL for detection |
| `GET`      | `/api/result/<task_id>` | Get detection result     |
| `GET`      | `/api/history`          | View detection history   |

### Example: Submit a URL

    curl -X POST http://127.0.0.1:5000/api/detect \
      -H "Content-Type: application/json" \
      -d '{"url": "https://www.instagram.com/p/example/"}'

---

## Model Performance

| **ModelDataset** | **Dataset**     | **Accuracy** | **Parameters** | **Training Time** |
| ---------------- | --------------- | ------------ | -------------- | ----------------- |
| CNN (Original)   | CIFAKE (100K)   | 95.39%       | 845,602        | 54 min (CPU)      |
| CNN (Fine-tuned) | Instagram (320) | **98.75%**   | 845,602        | 7 min (CPU)       |
| ViT (Fine-tuned) | CIFAKE (5K)     | 94.50%       | 85,800,194     | 103 min (CPU)     |

---

## Key Features

- 🔍 **Social Media URL Support** — Instagram, Twitter/X, TikTok, Facebook
- 🧠 **Ensemble Detection** — CNN + ViT with weighted voting
- 📊 **Confidence Scores** — Transparent percentage for every prediction
- 📝 **Detection History** — View past analyses
- 🎨 **Professional UI** — Minimal, clean interface built with React + TailwindCSS
- ⚡ **Fast Inference** — Under 1 second per image on CPU

---

## Limitations & Future Work

### Current Limitations

- ViT trained on only 5,000 images due to CPU-only training (Python 3.13 CUDA incompatibility)
- Social media dataset limited to 400 images (needs 50,000+ for production)
- Instagram scraping requires authentication for live testing
- Only tested on Instagram; Twitter/TikTok performance not yet validated

### Future Roadmap

- Full ViT training on GPU with complete 100K CIFAKE dataset
- Expand social media dataset to 50,000+ images across platforms
- Multi-modal detection using image captions + visual features
- Browser extension for real-time Instagram verification
- Cross-generator testing with Midjourney, DALL-E, Adobe Firefly

---

## Acknowledgments

- **CIFAKE Dataset:** Bird, J.J. & Lotfi, A. (2024), *IEEE Access*
- **CIFAR-10:** Krizhevsky & Hinton (2009)
- **ThisPersonDoesNotExist:** StyleGAN-generated faces for social media dataset
- **HuggingFace:** Pre-trained model weights and infrastructure
