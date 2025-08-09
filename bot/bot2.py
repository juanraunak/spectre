import json
import time
import asyncio
from openai import AzureOpenAI
from pathlib import Path
import re

# --- Azure OpenAI Configuration ---
client = AzureOpenAI(
    api_key="2be1544b3dc14327b60a870fe8b94f35",
    api_version="2024-06-01",
    azure_endpoint="https://notedai.openai.azure.com"
)
DEPLOYMENT_NAME = "gpt-4o"

# --- Directory containing individual JSON files ---
PROFILE_DIR = Path("profiles")

# --- Build prompt ---
def build_prompt(profile):
    return f"""
You are a data extractor. Convert the following raw LinkedIn profile into structured JSON.

Extract these fields if available:
- name
- pronouns (optional)
- profile_url
- current_position (title, company, location, start_date, end_date, duration)
- previous_positions (same fields as above)
- education (institution, degree, field_of_study, years_attended)
- connections
- followers
- skills (list)
- interests (list of names or companies)

Only include fields that are present. Format the output in clean, valid JSON. Do not guess missing data.

Profile URL: {profile.get('url', '')}

RAW TEXT:
---
{profile.get('raw_text', '')}

SKILLS TEXT:
---
{profile.get('skills_text', '')}
"""

# --- Process one file ---
async def process_file(filepath: Path):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "cleaned_profile" in data:
        print(f"⏭️ Already processed: {filepath.name}")
        return None

    prompt = build_prompt(data)

    try:
        print(f"🧠 Processing: {filepath.name}")
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        result = response.choices[0].message.content
        usage = response.usage

        try:
            # If GPT wrapped the output in ```json ... ```
            match = re.search(r"```json\n(.*)\n```", result, re.DOTALL)
            if match:
                cleaned = json.loads(match.group(1))
            else:
                cleaned = json.loads(result)
        except json.JSONDecodeError:
            cleaned = {"raw_output": result}


        # Update original file with cleaned data
        data["cleaned_profile"] = cleaned
        data["tokens_used"] = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved: {filepath.name}")

        return usage.total_tokens

    except Exception as e:
        print(f"❌ Error on {filepath.name}: {e}")
        return None

# --- Main runner ---
async def run_all():
    files = list(PROFILE_DIR.glob("*.json"))
    total_tokens = 0

    for file in files:
        tokens = await process_file(file)
        if tokens:
            total_tokens += tokens
        await asyncio.sleep(1)  # avoid hitting rate limits

    print(f"\n📊 Total tokens used across all files: {total_tokens}")
    estimated_cost = (total_tokens / 1000) * 0.005 + (total_tokens / 1000) * 0.015
    print(f"💰 Estimated cost (USD): ${estimated_cost:.4f}")

# --- Entry point ---
if __name__ == "__main__":
    asyncio.run(run_all())
