# Instagram/Facebook News Automation

A Python automation pipeline for turning daily news into Instagram and Facebook-ready content.

The project collects news candidates from Google News, uses Gemini to select relevant articles, extracts article bodies, generates Korean social media captions, creates poster images, uploads final images to Cloudflare R2, and prepares Instagram/Facebook publishing through the Meta Graph API.

## Current Status

This project is still in development.

- News collection, article selection, body extraction, caption generation, image generation, poster overlay, and Cloudflare R2 upload have been implemented.
- Cloudflare R2 image upload has been tested successfully.
- Meta/Instagram/Facebook publishing functions are implemented, but the required Meta API credentials have not been added yet.
- The social publishing flow has not been fully tested because the Meta API setup is not complete yet.

## Pipeline

1. Collect fresh news candidates from Google News.
2. Exclude articles already recorded in `history.jsonl`.
3. Send only `id`, `category`, `title`, and `source` to Gemini.
4. Ask Gemini to choose a primary and backup article for each category.
5. Resolve selected Google News links into original publisher URLs.
6. Extract article body text using `requests` and `trafilatura`.
7. Generate Korean social media captions with Gemini.
8. Generate SDXL image prompts with Gemini.
9. Generate poster images through Hugging Face.
10. Add Korean title text and a bottom gradient overlay with Pillow.
11. Upload final images to Cloudflare R2.
12. Save local run outputs under `outputs/YYYY-MM-DD/`.
13. Optionally publish to Instagram and Facebook once Meta API credentials are configured.

## Features

- Google News candidate collection by category
- Gemini-based article selection with primary/backup fallback
- Google News URL decoding
- Article body extraction
- Korean social media caption generation
- SDXL image prompt generation
- Hugging Face image generation
- Poster title overlay with Korean font support
- Cloudflare R2 image hosting
- Prepared Instagram and Facebook publishing functions
- Daily publish limit guard
- Duplicate publish prevention through `history.jsonl`

## Requirements

- Python 3.11+
- Gemini API key
- Hugging Face token
- Cloudflare R2 bucket and API credentials
- Meta Developer app and access tokens for Instagram/Facebook publishing

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
