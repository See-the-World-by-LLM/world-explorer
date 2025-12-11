#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub",
#     "pillow",
#     "python-dotenv",
#     "requests",
# ]
# ///
from __future__ import annotations

import io
import os
import random
import subprocess
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

API_URL = "https://huggingface.co/api/models"
GITHUB_API = "https://api.github.com"
DEFAULT_PIPELINE_TAG = "text-generation"
DEFAULT_INFERENCE_PROVIDER = "all"
MAX_RETRIES = 10
MIN_DOWNLOADS = 10_000
MAX_GENERATION_TRIES = 10
MAX_IMAGE_GENERATION_TRIES = 10
SLEEP_SECONDS = 1.0
DEFAULT_PROMPT_EN = (
    "Write a travel blog post about {city_en}, {country}. "
    "The tone should be engaging, humorous, and informative. "
    "Include the following sections:\n"
    "1. A playful welcome.\n"
    "2. At least 3 fun facts about the city.\n"
    "3. Local food you must try.\n"
    "4. A one-day itinerary (Where to go if you only have 24 hours).\n"
    "5. Expectation vs. Reality (humorous comparison).\n"
    "6. The Local’s Cheat Sheet (tips on transport, etiquette, or hidden gems).\n"
    "7. An encouraging conclusion.\n\n"
    "Format the output exactly as follows:\n"
    "SUMMARY: [A 2-sentence summary of the post]\n"
    "CONTENT:\n"
    "[The full blog post content in Markdown format. Use headers ## for the sections above, lists, and bold text.]"
)

DEFAULT_PROMPT_ZH = (
    "写一篇关于{country}{city_zh}的旅游博客文章。"
    "文章风格应该是幽默、风趣且信息丰富的。"
    "请包含以下部分：\n"
    "1. 一个俏皮的欢迎语。\n"
    "2. 至少3个关于这座城市的有趣冷知识。\n"
    "3. 必尝的当地美食。\n"
    "4. 一日游攻略 (如果只有24小时该去哪里)。\n"
    "5. 理想 vs. 现实的幽默的对比。\n"
    "6. 本地人秘籍 (关于交通、礼仪或隐藏景点的建议)。\n"
    "7. 鼓励读者前往的结尾。\n\n"
    "请严格按照以下格式输出：\n"
    "CITY_ZH: [城市的中文名称]\n"
    "COUNTRY_ZH: [国家的中文名称]\n"
    "SUMMARY: [2句话的文章摘要]\n"
    "CONTENT:\n"
    "[完整的Markdown格式博客文章。使用##标题对应上述部分、列表和粗体文本。]"
)

GITHUB_REPO = os.getenv("GH_REPO", "See-the-World-by-LLM/see-the-world-by-llm")
POSTS_DIR = "src/data/posts"
CITY_LIST_PATH = os.getenv("CITY_LIST_PATH", "src/data/cities.txt")
DEFAULT_CITY_EN = os.getenv("CITY_EN", "Tokyo")
DEFAULT_CITY_ZH = os.getenv("CITY_ZH", "东京")
DEFAULT_COUNTRY = os.getenv("COUNTRY", "Japan")
DEFAULT_PHOTO = os.getenv("PHOTO_URL", "/images/cities/tokyo.jpg")

MIN_CONTENT_LEN = 200
IMAGE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
LOCAL_REPO_PATH = Path("see-the-world-by-llm")


def ensure_local_repo(token: Optional[str]) -> None:
    """
    Ensures the local repository is present and up-to-date.
    """
    repo_path = LOCAL_REPO_PATH.resolve()
    if repo_path.exists():
        print(f"Updating local repository at {repo_path}...")
        try:
            subprocess.run(["git", "pull"], cwd=repo_path, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to pull repo: {e}. Continuing with existing files.")
    else:
        print(f"Cloning repository to {repo_path}...")
        if token:
            repo_url = f"https://x-access-token:{token}@github.com/{GITHUB_REPO}.git"
        else:
            repo_url = f"https://github.com/{GITHUB_REPO}.git"

        try:
            subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repository: {e}")


def fetch_models(
    token: Optional[str] = None,
) -> List[str]:
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    session = requests.Session()
    params: Dict[str, Any] = {
        "pipeline_tag": DEFAULT_PIPELINE_TAG,
        "inference_provider": DEFAULT_INFERENCE_PROVIDER,
    }

    payload: List[Dict[str, Any]]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(API_URL, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                detail = ""
                if hasattr(exc, "response") and exc.response is not None:
                    detail = f" Response: {exc.response.text}"
                raise RuntimeError(
                    f"Request failed after {attempt} attempts: {exc}.{detail}"
                ) from exc
            time.sleep(SLEEP_SECONDS * attempt)

    if not isinstance(payload, list):
        raise RuntimeError(
            f"Unexpected response format: {type(payload)}; expected a list of models."
        )

    popular = [m for m in payload if m.get("downloads", 0) > MIN_DOWNLOADS]

    model_ids: List[str] = []
    print(f"Fetched {len(payload)} models.")
    print(f"Models with downloads > {MIN_DOWNLOADS}: {len(popular)}")
    for idx, model in enumerate(popular, start=1):
        mid = model.get("id")
        if isinstance(mid, str):
            model_ids.append(mid)
            downloads = model.get("downloads", 0)
            print(f"{idx}: {mid} (downloads={downloads})")

    return model_ids


def load_city_pool() -> List[Dict[str, Any]]:
    # Try local file first
    local_path = LOCAL_REPO_PATH / "src/data/cities.txt"
    if local_path.exists():
        print(f"Loading cities from local file: {local_path}")
        text = local_path.read_text(encoding="utf-8")
    else:
        raise RuntimeError(f"No local cities.txt found at {local_path}")

    cities: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        city_en, country = parts[0], parts[1]
        if not city_en or not country:
            continue
        slug = city_en.lower().replace(",", "").replace(" ", "-")
        photo = f"/images/cities/{slug}.jpg"
        cities.append(
            {
                "en": city_en,
                "zh": city_en,  # no zh provided; fall back to en
                "country": country,
                "photoUrl": photo,
            }
        )
    if cities:
        return cities
    raise RuntimeError("Parsed zero cities from text list")


def load_used_cities() -> List[str]:
    used: List[str] = []

    # Try local directory first
    local_posts_dir = LOCAL_REPO_PATH / "src/data/posts"
    if local_posts_dir.exists():
        print(f"Loading used cities from local directory: {local_posts_dir}")

        # Directory based Markdown posts (e.g. tokyo/en.md)
        for city_dir in local_posts_dir.iterdir():
            if city_dir.is_dir():
                # Assume directory name is the slug, but we need the city name.
                # We can read the en.md frontmatter or just assume the slug is "used".
                # Better to read the frontmatter to be accurate, or just use the slug if that's how we match.
                # The choose_city logic compares city['en'].lower() with used list.
                # So we need the city name.
                en_md = city_dir / "en.md"
                if en_md.exists():
                    try:
                        content = en_md.read_text(encoding="utf-8")
                        # Simple parsing for "city: Tokyo"
                        for line in content.splitlines():
                            if line.startswith("city:"):
                                name = line.split(":", 1)[1].strip()
                                used.append(name.lower())
                                break
                    except Exception as exc:
                        print(f"Skip unreadable local post {en_md}: {exc}")

    return used


def choose_city(cities: List[Dict[str, Any]], used: List[str]) -> Dict[str, Any]:
    available = [c for c in cities if c.get("en", "").lower() not in used]
    if not available:
        raise RuntimeError("No unused cities remain in the list.")
    return random.choice(available)


def generate_city_image(token: str, city_en: str, country: str) -> Optional[bytes]:
    client = InferenceClient(api_key=token)

    # Randomize anime girl characteristics
    hair = random.choice(
        ["pink", "blue", "silver", "blonde", "purple", "black", "brown", "red"]
    )
    style = random.choice(
        ["long flowing", "short bob", "twin tails", "ponytail", "wavy"]
    )
    outfit = random.choice(
        [
            "casual hoodie",
            "summer dress",
            "stylish jacket",
            "traditional outfit",
            "school uniform",
        ]
    )

    prompt = (
        f"A cute anime girl with {hair} {style} hair, wearing {outfit}, "
        f"holding a sign board that says '{city_en}, {country}'. "
        f"Background is a scenic view of {city_en}. "
        "Anime style, high quality, detailed, 8k resolution, 16:9 aspect ratio."
    )

    for attempt in range(1, MAX_IMAGE_GENERATION_TRIES + 1):
        print(
            f"Generating image for {city_en} (Attempt {attempt}/{MAX_IMAGE_GENERATION_TRIES})..."
        )
        try:
            image = client.text_to_image(
                prompt,
                width=1600,
                height=900,
                model=IMAGE_MODEL,
            )
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG", quality=85)
            print("Image generation successful.")
            return img_byte_arr.getvalue()
        except Exception as e:
            print(f"Image generation failed (Attempt {attempt}): {e}")
            if attempt < MAX_IMAGE_GENERATION_TRIES:
                time.sleep(SLEEP_SECONDS * attempt)

    print("All image generation attempts failed.")
    return None


def generate_blog_content(
    model_ids: List[str],
    token: str,
    city_en: str,
    city_zh: str,
    country: str,
    lang: str,
    preferred_model: Optional[str] = None,
) -> Tuple[str, str, str, Optional[str], Optional[str]]:
    if not token:
        raise RuntimeError("HF_TOKEN is required for generation.")

    client = InferenceClient(api_key=token)

    # Build an ordered list of models to try:
    # 1) preferred_model (if provided), 2) random order of fetched models.
    models_to_try: List[str] = []
    if preferred_model:
        models_to_try.append(preferred_model)

    pool = model_ids.copy() if model_ids else []
    random.shuffle(pool)
    for m in pool:
        if m != preferred_model:
            models_to_try.append(m)

    if not models_to_try:
        raise RuntimeError("No models available to generate from (preferred model missing and fetch failed).")

    last_error: Optional[Exception] = None
    max_tries = min(MAX_GENERATION_TRIES, len(models_to_try))

    prompt_template = DEFAULT_PROMPT_EN if lang == "en" else DEFAULT_PROMPT_ZH
    prompt = prompt_template.format(city_en=city_en, city_zh=city_zh, country=country)

    for attempt, model_id in enumerate(models_to_try[:max_tries], start=1):
        try:
            print(
                f"Generating {lang.upper()} content with model ({attempt}/{max_tries}): {model_id}"
            )

            completion = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
            )
            text = completion.choices[0].message.content

            # Parse the output based on markers
            summary = ""
            content = ""
            parsed_city_zh = None
            parsed_country_zh = None

            if "SUMMARY:" in text and "CONTENT:" in text:
                # Extract metadata if present (only for ZH currently)
                if "CITY_ZH:" in text:
                    try:
                        parsed_city_zh = (
                            text.split("CITY_ZH:")[1].split("\n")[0].strip()
                        )
                    except Exception:
                        pass
                if "COUNTRY_ZH:" in text:
                    try:
                        parsed_country_zh = (
                            text.split("COUNTRY_ZH:")[1].split("\n")[0].strip()
                        )
                    except Exception:
                        pass

                parts = text.split("CONTENT:")
                summary_part = parts[0].split("SUMMARY:")[1].strip()
                content_part = parts[1].strip()
                summary = summary_part
                content = content_part
            else:
                # Fallback parsing if model ignores format
                lines = text.strip().split("\n")
                summary = lines[0]
                content = "\n".join(lines[1:])

            if len(content) < MIN_CONTENT_LEN:
                raise ValueError("Generated content too short")

            print(f"Success with model: {model_id}")
            return summary, content, model_id, parsed_city_zh, parsed_country_zh

        except Exception as exc:
            last_error = exc
            msg = str(exc)
            if "model_not_supported" in msg:
                print(f"Model failed: {model_id} -> Not supported by provider.")
            else:
                print(f"Model failed: {model_id} -> {msg}")
            time.sleep(SLEEP_SECONDS)

    raise RuntimeError(
        f"Generation failed for {lang} after {max_tries} tries. Last error: {last_error}"
    )


def build_post_payload(
    date_str: str,
    city_en: str,
    city_zh: str,
    country: str,
    photo_url: str,
    model_name: str,
    content: Dict[str, str],
    slug: str,
) -> Dict[str, Any]:
    return {
        "date": date_str,
        "city": {"en": city_en, "zh": city_zh, "country": country},
        "slug": slug,
        "photoUrl": photo_url,
        "summaryEn": content["summaryEn"],
        "summaryZh": content["summaryZh"],
        "contentEn": content["contentEn"],
        "contentZh": content["contentZh"],
        "model": model_name,
    }


def commit_file_to_github(
    repo: str, path: str, content_bytes: bytes, token: str, message: str
) -> None:
    # Deprecated: Using git push instead
    pass


def commit_post_to_github(
    repo: str, path: str, content: Dict[str, Any], token: str
) -> None:
    # Deprecated: Using git push instead
    pass


def push_to_github(city_name: str) -> None:
    """
    Commits and pushes changes in the local repository to GitHub.
    """
    repo_path = LOCAL_REPO_PATH.resolve()
    print(f"\n--- Pushing changes to GitHub from {repo_path} ---")

    try:
        # Configure git user
        subprocess.run(
            ["git", "config", "user.name", "World Explorer Bot"],
            cwd=repo_path,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "config",
                "user.email",
                "generate-post-script@noreply.world-explorer",
            ],
            cwd=repo_path,
            check=True,
        )

        # Check if there are changes
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

        if not status.stdout.strip():
            print("No changes to commit.")
            return

        # Stage all changes
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)

        # Commit
        commit_message = f"Add post for {city_name}"
        subprocess.run(
            ["git", "commit", "-m", commit_message], cwd=repo_path, check=True
        )

        # Push
        subprocess.run(["git", "push"], cwd=repo_path, check=True)
        print("Successfully pushed to GitHub.")

    except subprocess.CalledProcessError as e:
        print(f"Git operation failed: {e}")
    except Exception as e:
        print(f"An error occurred during git push: {e}")


def main() -> None:
    # Load .env from the working directory so HF_TOKEN is picked up
    # automatically.
    load_dotenv(dotenv_path=".env")
    hf_token = os.getenv("HF_TOKEN")
    gh_token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")

    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for generation")

    # Ensure local repo is up to date
    ensure_local_repo(gh_token)

    try:
        model_ids = fetch_models(token=hf_token)
    except Exception as e:
        print(f"Warning: Failed to fetch models from API: {e}")
        print("Proceeding with preferred model (if any).")
        model_ids = []

    city_override = os.getenv("CITY_EN")
    if city_override:
        city_en = city_override
        city_zh = os.getenv("CITY_ZH", DEFAULT_CITY_ZH)
        country = os.getenv("COUNTRY", DEFAULT_COUNTRY)
        photo_url = os.getenv("PHOTO_URL", DEFAULT_PHOTO)
    else:
        city_pool = load_city_pool()
        used_cities = load_used_cities()
        chosen_city = choose_city(city_pool, used_cities)
        city_en = chosen_city.get("en", DEFAULT_CITY_EN)
        city_zh = chosen_city.get("zh", DEFAULT_CITY_ZH)
        country = chosen_city.get("country", DEFAULT_COUNTRY)
        photo_url = chosen_city.get("photoUrl", DEFAULT_PHOTO)

    date_str = os.getenv("POST_DATE") or date.today().isoformat()
    slug = city_en.lower().replace(",", "").replace(" ", "-")

    # Generate Image
    image_bytes = generate_city_image(hf_token, city_en, country)
    if image_bytes:
        # Save image locally
        local_image_path = LOCAL_REPO_PATH / "public/images/cities" / f"{slug}.jpg"
        print(f"Saving image to {local_image_path}...")
        try:
            local_image_path.parent.mkdir(parents=True, exist_ok=True)
            local_image_path.write_bytes(image_bytes)
            photo_url = f"/images/cities/{slug}.jpg"
        except Exception as e:
            print(f"Failed to save local image: {e}")
            photo_url = DEFAULT_PHOTO
    else:
        print("Using default photo URL due to generation failure.")

    # Generate English Content
    print("\n--- Generating English Content ---")
    preferred_model = os.getenv("PREFERRED_MODEL")
    summary_en, content_en, model_en, _, _ = generate_blog_content(
        model_ids,
        token=hf_token,
        city_en=city_en,
        city_zh=city_zh,
        country=country,
        lang="en",
        preferred_model=preferred_model,
    )

    # Generate Chinese Content
    print("\n--- Generating Chinese Content ---")
    summary_zh, content_zh, model_zh, parsed_city_zh, parsed_country_zh = (
        generate_blog_content(
            model_ids,
            token=hf_token,
            city_en=city_en,
            city_zh=city_zh,
            country=country,
            lang="zh",
            preferred_model=preferred_model,
        )
    )

    # Use parsed Chinese names if available, otherwise fallback to existing
    final_city_zh = parsed_city_zh if parsed_city_zh else city_zh
    final_country_zh = parsed_country_zh if parsed_country_zh else country

    # Save post locally as Markdown
    # Structure: src/data/posts/[slug]/en.md and src/data/posts/[slug]/zh.md

    created_at = int(time.time() * 1000)

    # Helper to format frontmatter
    def create_markdown(
        lang_content: str, lang_summary: str, lang_code: str, model_name: str
    ) -> str:
        # Ensure content is a string (handle list case from previous errors)
        if isinstance(lang_content, list):
            lang_content = "\n".join(lang_content)
        if isinstance(lang_summary, list):
            lang_summary = "\n".join(lang_summary)

        return (
            "---\n"
            f"title: {city_en if lang_code == 'en' else final_city_zh}\n"
            f"date: {date_str}\n"
            f"createdAt: {created_at}\n"
            f"city: {city_en}\n"
            f"city_zh: {final_city_zh}\n"
            f"country: {country}\n"
            f"country_zh: {final_country_zh}\n"
            f"slug: {slug}\n"
            f"photoUrl: {photo_url}\n"
            f"model: {model_name}\n"
            f'summary: "{lang_summary.replace('"', '\\"')}"\n'
            "---\n\n"
            f"{lang_content}\n"
        )

    posts_dir = LOCAL_REPO_PATH / "src/data/posts" / slug
    print(f"Saving posts to {posts_dir}...")

    try:
        posts_dir.mkdir(parents=True, exist_ok=True)

        # Save English
        en_md = create_markdown(content_en, summary_en, "en", model_en)
        (posts_dir / "en.md").write_text(en_md, encoding="utf-8")

        # Save Chinese
        zh_md = create_markdown(content_zh, summary_zh, "zh", model_zh)
        (posts_dir / "zh.md").write_text(zh_md, encoding="utf-8")

    except Exception as e:
        print(f"Failed to save local post: {e}")

    print("\n--- Post Created ---")
    print(f"Local Path: {posts_dir}")
    print(f"Models Used: EN={model_en}, ZH={model_zh}")

    # Push to GitHub
    push_to_github(city_en)


if __name__ == "__main__":
    main()
