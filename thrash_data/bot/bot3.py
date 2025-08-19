import json
from pathlib import Path

PROFILE_DIR = Path("profiles")
output_file = "xto10x_employees.json"

xto10x_employees = []

for json_file in PROFILE_DIR.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = data.get("cleaned_profile")
    if not cleaned:
        continue

    current_position = cleaned.get("current_position")
    company = current_position.get("company", "") if current_position else ""

    if "xto10x" in company.lower():
        xto10x_employees.append(cleaned)
    else:
        print(f"⚠️ Skipping {json_file.name} - Company: {company}")

# Save filtered employees
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(xto10x_employees, f, ensure_ascii=False, indent=2)

print(f"✅ Done! Saved {len(xto10x_employees)} xto10x profiles to {output_file}")
