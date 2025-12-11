# World Explorer Automation

This project contains the automation scripts for the [See the World by LLM](https://github.com/See-the-World-by-LLM/see-the-world-by-llm) blog. It automatically generates bilingual travel blog posts and anime-style city illustrations using various AI models, then publishes them to the blog repository.

## 🚀 Features

-   **Auto-Discovery**: Fetches trending text-generation models from Hugging Face to ensure variety.
-   **Bilingual Content**: Generates engaging travel guides in both English and Chinese.
-   **Visuals**: Creates custom anime-style illustrations for each city using text-to-image models.
-   **GitOps**: Automatically handles repository cloning, syncing, committing, and pushing changes.
-   **Smart Selection**: Tracks visited cities to avoid duplicates.

## 📋 Prerequisites

-   **Python 3.11+**
-   **[uv](https://github.com/astral-sh/uv)**: An extremely fast Python package installer and resolver.
-   **Git**: For repository operations.

## ⚙️ Configuration

1.  **Clone this repository** (if you haven't already):
    ```bash
    git clone https://github.com/See-the-World-by-LLM/world-explorer
    cd world-explorer
    ```

2.  **Create a `.env` file** in the root directory:
    ```bash
    touch .env
    ```

3.  **Add your API tokens** to `.env`:
    ```env
    # [Required] Hugging Face Token for inference API
    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # [Required] GitHub Token for cloning and pushing to the blog repo
    GH_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    # [Optional] Overrides
    # CITY_EN=Paris
    # POST_DATE=2025-01-01
    ```

## 🏃 Usage

The script is designed to be run with `uv`, which handles dependencies automatically.

### One-off Run

```bash
uv run generate_post.py
```

### How it Works

When you execute the script, it performs the following steps:

1.  **Repo Sync**: Checks for the `see-the-world-by-llm` directory.
    *   If missing: Clones the repository using your `GH_TOKEN`.
    *   If present: Pulls the latest changes.
2.  **City Selection**: Reads `src/data/cities.txt` from the repo and picks a random city that hasn't been generated yet.
3.  **Content Generation**:
    *   **Image**: Generates a 16:9 anime-style city view.
    *   **Text**: Uses a trending LLM to write a structured blog post in English and Chinese.
4.  **File Creation**: Saves the content as Markdown files with Frontmatter in `src/data/posts/[city-slug]/`.
5.  **Publish**: Commits the new files and pushes them to the remote GitHub repository.

## ⏰ Automation (Cron)

To run this script automatically (e.g., every 8 hours), add a cron job.

1.  Open crontab:
    ```bash
    crontab -e
    ```

2.  Add the schedule:
    ```cron
    # Run at minute 0 past every 8th hour (00:00, 08:00, 16:00)
    0 */8 * * * cd /path/to/world-explorer && /path/to/uv run generate_post.py >> /tmp/world_explorer.log 2>&1
    ```

    *Replace `/path/to/world-explorer` and `/path/to/uv` with your absolute paths.*

## ☁️ Hugging Face Jobs

You can run this script via Hugigng Face Jobs as well.

```bash
hf jobs uv run --secrets-file .env generate_post.py
```

You can also run this script as a scheduled job on Hugging Face.

Run the script at 0:00, 8:00, and 16:00 UTC daily:
```bash
hf jobs scheduled uv run --secrets-file .env '0 0,8,16 * * *' generate_post.py
```

## 🛠️ Development

If you want to run it without `uv`, you can set up a standard virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_post.py
```
