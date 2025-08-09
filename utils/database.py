import os
import json
from datetime import datetime

# 🧱 Ensure company folder exists under /db
def ensure_company_db_folder(company_id: str) -> str:
    base_path = os.path.join("db", company_id.lower())
    os.makedirs(base_path, exist_ok=True)
    return base_path

# 💾 Save a single JSON file to company folder
def save_json(company_id: str, filename: str, data: dict | list):
    folder = ensure_company_db_folder(company_id)
    file_path = os.path.join(folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[✅] Saved: {filename} in {folder}")

# 📦 Save all 3 outputs from Head One in one go
def save_all_outputs(company_id: str, company_data: dict, employee_data: list, merged_data: dict):
    save_json(company_id, "company_report.json", company_data)
    save_json(company_id, "employees_enriched.json", employee_data)
    save_json(company_id, "hydra_head_one_output.json", merged_data)

# 🧾 Optional metadata generator
def save_metadata(company_id: str):
    meta = {
        "company_id": company_id.lower(),
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "version": "hydra-head-one-v1"
    }
    save_json(company_id, "metadata.json", meta)
