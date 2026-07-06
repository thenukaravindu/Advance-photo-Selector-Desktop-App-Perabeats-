# 📸 AI-Powered Bulk Photo Selector (NIMA Aesthetic Evaluator)

An automated Image Quality Assessment (IQA) tool built with **PyTorch** and **MobileNetV2**. This script helps student clubs, creative societies, and professional photographers automate the tedious process of "culling" (selecting the best photos) from a massive bulk event folder.

It scans a folder containing **JPG, PNG, WebP**, and even high-end camera **RAW formats (`.cr2`, `.nef`, `.arw`, etc.)**, evaluates each image using Google's NIMA (Neural Image Assessment) approach, and filters out the top-scoring photos into a separate folder automatically.

---

## ✨ Features
* **Aesthetic & Technical Scoring:** Evaluates photos on a scale of 1-10 based on lighting, composition, blurriness, and noise.
* **Camera RAW Support:** Native decoding for professional camera RAW files using `rawpy`.
* **High Performance:** Fast embedded-JPEG thumbnail extraction for RAW images to speed up evaluation.
* **Hardware Acceleration:** Automatically utilizes NVIDIA CUDA GPU if available; falls back safely to CPU.
* **Interactive CLI:** Prompts for paths dynamically with automatic quotes stripping (supports drag-and-drop).
* **Progress Tracking:** Includes a neat visual progress bar powered by `tqdm`.

---
