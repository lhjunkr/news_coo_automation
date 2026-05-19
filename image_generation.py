import os
from typing import TypedDict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from constants import (
    IMAGE_GENERATION_STATUS_FAILED,
    IMAGE_GENERATION_STATUS_SKIPPED_NO_PROMPT,
    STATUS_SUCCESS,
)
from models import Article


class HuggingFaceImageModelConfig(TypedDict):
    model: str
    num_inference_steps: int
    guidance_scale: float


HUGGINGFACE_IMAGE_MODELS: list[HuggingFaceImageModelConfig] = [
    {
        "model": "stabilityai/stable-diffusion-3.5-large-turbo",
        "num_inference_steps": 10,
        "guidance_scale": 7.5,
    },
    {
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
    },
    {
        "model": "black-forest-labs/FLUX.1-schnell",
        "num_inference_steps": 4,
        "guidance_scale": 3.5,
    },
]


def generate_huggingface_image(article: Article, run_dir) -> Article:
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(".env 파일에 HF_TOKEN을 먼저 입력하세요.")

    if not article.sdxl_image_prompt:
        article.image_path = ""
        article.image_generation_status = IMAGE_GENERATION_STATUS_SKIPPED_NO_PROMPT
        return article

    output_dir = run_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"article_{article.id}.png"

    client = InferenceClient(token=hf_token)
    last_error = ""

    # 외부 이미지 provider는 독립적으로 timeout이 날 수 있으므로,
    # 기사 실패 처리 전에 다음 모델을 시도해 카테고리 backup 기회를 보존합니다.
    for image_model_config in HUGGINGFACE_IMAGE_MODELS:
        image_model = image_model_config["model"]

        try:
            print(f" -> 이미지 모델 시도: {image_model}")

            image = client.text_to_image(
                prompt=article.sdxl_image_prompt,
                negative_prompt=(
                    "text, watermark, logo, low quality, blurry, distorted face, "
                    "extra fingers, oversaturated, artificial glow"
                ),
                model=image_model,
                width=1024,
                height=1280,
                num_inference_steps=image_model_config["num_inference_steps"],
                guidance_scale=image_model_config["guidance_scale"],
            )

            image.save(image_path)

            article.image_path = str(image_path)
            article.image_generation_status = STATUS_SUCCESS
            article.image_generation_model = image_model
            article.image_generation_error = ""
            return article

        except Exception as e:
            last_error = str(e)
            print(f" -> 이미지 모델 실패: {image_model} ({e})")

    article.image_path = ""
    article.image_generation_status = IMAGE_GENERATION_STATUS_FAILED
    article.image_generation_model = ""
    article.image_generation_error = last_error

    return article


def generate_huggingface_images(selected_articles: list[Article], run_dir) -> list[Article]:
    for article in selected_articles:
        print(f"Hugging Face 이미지 생성 중: {article.title[:30]}...")
        generate_huggingface_image(article, run_dir)

    return selected_articles