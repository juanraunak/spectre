import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import difflib
import time

# Azure OpenAI (v1 SDK)
from openai import AzureOpenAI

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EmployeeMatcher")

# -------------------- Data Models --------------------
@dataclass
class EmployeeProfile:
    employee_id: str
    name: str
    position: str
    department: str
    company: str
    is_target: bool
    raw_data: Dict = None

@dataclass
class FuzzyMatch:
    target_employee: Dict
    candidate_employee: Dict
    similarity_score: float
    reason: str

@dataclass
class VerifiedMatch:
    target_employee: Dict
    candidate_employee: Dict
    gpt_is_match: bool
    gpt_confidence: float   # 0..1
    gpt_reason: str
    fuzzy_score: float

# -------------------- Matcher --------------------
class EmployeeMatcher:
    """
    Two-stage employee matcher:
      1) Fuzzy/heuristic candidate generation
      2) GPT verification of candidate pairs
    """

    def __init__(
        self,
        target_company: str = "Manipal Fintech",
        *,
        openai_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment_id: Optional[str] = None,
        fuzzy_min_score: float = 0.42,
        max_candidates_per_target: int = 5,
        gpt_enabled_default: bool = True,
        gpt_max_pairs_per_target: int = 8,
        gpt_match_threshold: float = 0.55
    ):
        self.target_company = target_company
        self.target_employees: List[EmployeeProfile] = []
        self.competitor_employees: List[EmployeeProfile] = []
        self.all_employees: List[EmployeeProfile] = []
        self.raw_employee_data: Dict[str, Dict] = {}

        self.fuzzy_min_score = fuzzy_min_score
        self.max_candidates_per_target = max_candidates_per_target
        self.gpt_max_pairs_per_target = gpt_max_pairs_per_target
        self.gpt_match_threshold = gpt_match_threshold

        # Position keyword buckets for boosts
        self.position_keywords = {
            'manager': ['manager', 'lead', 'head', 'supervisor', 'team lead'],
            'senior': ['senior', 'sr', 'principal', 'lead'],
            'analyst': ['analyst', 'associate', 'executive'],
            'engineer': ['engineer', 'developer', 'programmer', 'architect'],
            'director': ['director', 'vp', 'vice president', 'head of'],
            'specialist': ['specialist', 'expert', 'consultant', 'advisor'],
            'coordinator': ['coordinator', 'organizer', 'administrator'],
            'officer': ['officer', 'executive', 'representative'],
            'sales': ['sales', 'business development', 'bd', 'bde', 'bdr'],
            'finance': ['finance', 'credit', 'lending', 'collections', 'underwriting']
        }

        # --- Azure OpenAI setup ---
        self.use_gpt = bool(openai_api_key and azure_endpoint and azure_api_version and azure_deployment_id and gpt_enabled_default)
        self.azure_deployment_id = azure_deployment_id
        if self.use_gpt:
            try:
                self.openai_client = AzureOpenAI(
                    api_key=openai_api_key,
                    azure_endpoint=azure_endpoint,
                    api_version=azure_api_version
                )
                logger.info("🤖 Azure OpenAI client ready")
                logger.info(f"  • Endpoint: {azure_endpoint}")
                logger.info(f"  • Deployment: {azure_deployment_id}")
                logger.info(f"  • Version: {azure_api_version}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to init Azure OpenAI client, falling back to fuzzy-only. Error: {e}")
                self.use_gpt = False
                self.openai_client = None
        else:
            logger.warning("⚠️ GPT verification disabled (missing credentials or turned off).")
            self.openai_client = None

    # -------------------- Loaders --------------------
    def load_normalized_data(self, normalized_dir: str):
        """Robust loader: reads ANY .json in each company folder, accepts multiple shapes."""
        base = Path(normalized_dir).resolve()
        logger.info(f"📂 Loading normalized employee data from: {base}")
        if not base.exists():
            logger.error(f"❌ Directory not found: {base}")
            return

        company_dirs = [p for p in base.iterdir() if p.is_dir()]
        if not company_dirs:
            logger.warning("⚠️ No company folders found under normalized_departments")
            return

        logger.info(f"🏢 Company folders: {len(company_dirs)} -> {[d.name for d in company_dirs]}")
        total_loaded = 0

        for company_path in company_dirs:
            company_name = company_path.name.replace('_', ' ').title()
            is_target = self._is_target_company(company_name)
            json_files = sorted(company_path.glob("*.json"))
            if not json_files:
                logger.warning(f"⚠️ No .json files in {company_path}")
                continue

            logger.info(f"   📁 {company_path.name}: {len(json_files)} JSON files")
            company_count = 0

            for jf in json_files:
                try:
                    with jf.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to parse {jf.name}: {e}")
                    continue

                # ---- Flexible extraction of employees list ----
                employees = []
                if isinstance(data, dict):
                    if isinstance(data.get("employees"), list):
                        employees = data["employees"]
                    elif isinstance(data.get("data"), list):
                        employees = data["data"]
                    elif isinstance(data.get("items"), list):
                        employees = data["items"]
                    elif "employee_intelligence" in data and isinstance(data["employee_intelligence"].get("employees"), list):
                        employees = data["employee_intelligence"]["employees"]
                    else:
                        # try to find a list of dicts that looks like employees
                        for v in data.values():
                            if isinstance(v, list) and v and isinstance(v[0], dict):
                                employees = v
                                break
                elif isinstance(data, list):
                    if data and isinstance(data[0], dict):
                        employees = data

                if not employees:
                    logger.warning(f"   ⚠️ {jf.name}: no 'employees' list detected (keys: {list(data)[:6] if isinstance(data, dict) else 'top-level list/other'})")
                    continue

                # ---- Normalize and load ----
                file_count = 0
                for emp in employees:
                    if not isinstance(emp, dict):
                        continue

                    employee_id = str(emp.get('employee_id') or emp.get('id') or emp.get('uuid') or "").strip()

                    name = (emp.get('name')
                            or emp.get('full_name')
                            or f"{emp.get('first_name','')} {emp.get('last_name','')}".strip())
                    # Fallback if name is missing/blank → use ID or synthetic
                    if not name or not name.strip():
                        name = employee_id or f"Unknown_{file_count+1}"

                    position = (emp.get('position') or emp.get('title') or emp.get('role') or "").strip()
                    department = (emp.get('department') or emp.get('dept') or emp.get('function') or "").strip()
                    company = (emp.get('company') or company_name).strip()

                    profile = EmployeeProfile(
                        employee_id=employee_id,
                        name=name.strip(),
                        position=position,
                        department=department,
                        company=company,
                        is_target=is_target,
                        raw_data=None
                    )

                    self.all_employees.append(profile)
                    (self.target_employees if is_target else self.competitor_employees).append(profile)
                    file_count += 1

                company_count += file_count
                total_loaded += file_count
                logger.info(f"   ✅ {jf.name}: loaded {file_count} employees")

            logger.info(f"📦 Company total — {company_name}: {company_count}")

        logger.info(f"📊 FINAL LOAD: target={len(self.target_employees)}, competitor={len(self.competitor_employees)}, total={total_loaded}")
        if not total_loaded:
            logger.error("❌ Loaded 0 employees. Check working directory/path and JSON shapes.")

    def load_raw_employee_data(self, raw_data_dir: str):
        """Optional: load richer raw JSONs (to give GPT more context)."""
        logger.info("📂 Loading raw employee data (optional)...")
        if not os.path.exists(raw_data_dir):
            logger.warning(f"Raw data directory {raw_data_dir} does not exist")
            return

        json_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.json')]
        for jf in json_files:
            try:
                with open(os.path.join(raw_data_dir, jf), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    if "employee_intelligence" in data and "employees" in data["employee_intelligence"]:
                        employees = data["employee_intelligence"]["employees"]
                    else:
                        employees = data.get("employees", [])
                else:
                    employees = data if isinstance(data, list) else []

                for emp in employees:
                    if not isinstance(emp, dict):
                        continue

                    name = (emp.get('name')
                            or emp.get('full_name')
                            or f"{emp.get('first_name', '')} {emp.get('last_name', '')}").strip()
                    emp_id = str(emp.get('employee_id') or emp.get('id') or emp.get('uuid') or "").strip()

                    if name:
                        self.raw_employee_data[name.lower()] = emp
                    if emp_id:
                        self.raw_employee_data[emp_id.lower()] = emp
            except Exception as e:
                logger.warning(f"⚠️ Failed to load raw data from {jf}: {e}")

        logger.info(f"✅ Raw profiles loaded: {len(self.raw_employee_data)}")

    # -------------------- Stage 1: Fuzzy candidate generation --------------------
    def run_fuzzy_matching(self) -> List[FuzzyMatch]:
        logger.info("🔍 Running fuzzy candidate generation...")
        matches: List[FuzzyMatch] = []

        for t in self.target_employees:
            scored: List[Tuple[float, EmployeeProfile, str]] = []
            for c in self.competitor_employees:
                score, reason = self._fuzzy_similarity(t, c)
                if score >= self.fuzzy_min_score:
                    scored.append((score, c, reason))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[: self.max_candidates_per_target]
            for score, c, reason in top:
                matches.append(FuzzyMatch(
                    target_employee=self._pack_basic(t),
                    candidate_employee=self._pack_basic(c),
                    similarity_score=round(score, 3),
                    reason=reason
                ))

        logger.info(f"🏁 Fuzzy matching produced {len(matches)} candidate pairs")
        return matches

    def _fuzzy_similarity(self, a: EmployeeProfile, b: EmployeeProfile) -> Tuple[float, str]:
        """Heuristic similarity over department, position, and a tiny name cue (ignored if ID-like)."""
        dep_score = self._string_ratio(a.department, b.department)
        pos_score = self._position_similarity(a.position, b.position)
        name_hint = self._name_hint(a.name, b.name)

        score = 0.45 * dep_score + 0.5 * pos_score + 0.05 * name_hint
        reason_bits = []
        if dep_score >= 0.6: reason_bits.append("same/close department")
        if pos_score >= 0.55: reason_bits.append("similar role")
        if name_hint >= 0.7: reason_bits.append("name pattern overlap")
        if not reason_bits:
            reason_bits.append("text similarity")

        # keyword overlap boost
        boost = self._keyword_boost(a.position, b.position)
        score = min(1.0, score + boost)
        if boost > 0:
            reason_bits.append("keyword overlap")

        return score, ", ".join(reason_bits)

    @staticmethod
    def _string_ratio(x: str, y: str) -> float:
        if not x or not y:
            return 0.0
        return difflib.SequenceMatcher(None, x.lower(), y.lower()).ratio()

    def _position_similarity(self, p1: str, p2: str) -> float:
        if not p1 or not p2:
            return 0.0
        return difflib.SequenceMatcher(None, p1.lower(), p2.lower()).ratio()

    def _keyword_boost(self, p1: str, p2: str) -> float:
        p1l, p2l = p1.lower(), p2.lower()
        bonus = 0.0
        for kws in self.position_keywords.values():
            if any(k in p1l for k in kws) and any(k in p2l for k in kws):
                bonus += 0.06
        return min(bonus, 0.18)

    @staticmethod
    def _name_hint(n1: str, n2: str) -> float:
        # Ignore if names look like IDs or are too short
        def looks_like_id(s: str) -> bool:
            s = (s or "").lower()
            return bool(re.search(r'\b([a-z]{2,}_?\d{2,}|[a-z]{2,}\d{2,}|^\d+)$', s)) or len(s) < 3

        if looks_like_id(n1) or looks_like_id(n2):
            return 0.0

        try:
            l1 = n1.strip().lower().split()[-1]
            l2 = n2.strip().lower().split()[-1]
            if not l1 or not l2:
                return 0.0
            return 1.0 if l1 == l2 else difflib.SequenceMatcher(None, l1, l2).ratio() * 0.6
        except Exception:
            return 0.0

    @staticmethod
    def _pack_basic(e: EmployeeProfile) -> Dict[str, Any]:
        return {
            "employee_id": e.employee_id,
            "name": e.name,
            "position": e.position,
            "department": e.department,
            "company": e.company
        }

    # -------------------- Stage 2: GPT verification --------------------
    def verify_with_gpt(self, fuzzy_matches: List[FuzzyMatch]) -> List[VerifiedMatch]:
        if not self.use_gpt:
            logger.warning("⚠️ GPT disabled. Skipping verification; returning fuzzy matches as 'unverified'.")
            return [
                VerifiedMatch(
                    target_employee=m.target_employee,
                    candidate_employee=m.candidate_employee,
                    gpt_is_match=False,
                    gpt_confidence=0.0,
                    gpt_reason="GPT disabled; fuzzy-only result",
                    fuzzy_score=m.similarity_score
                )
                for m in fuzzy_matches
            ]

        logger.info("🧠 Verifying candidate pairs with GPT...")
        verified: List[VerifiedMatch] = []

        # Group pairs by target to keep output balanced
        by_target: Dict[str, List[FuzzyMatch]] = defaultdict(list)
        for m in fuzzy_matches:
            by_target[m.target_employee["employee_id"]].append(m)

        for target_id, pairs in by_target.items():
            pairs = sorted(pairs, key=lambda x: x.similarity_score, reverse=True)[: self.gpt_max_pairs_per_target]
            for p in pairs:
                vr = self._gpt_verify_one(p)
                if vr:
                    verified.append(vr)

        logger.info(f"🏁 GPT verified {len(verified)} pairs")
        return verified

    def _gpt_verify_one(self, pair: FuzzyMatch, retries: int = 2, backoff_sec: float = 2.0) -> Optional[VerifiedMatch]:
        # Lookup raw by name or employee_id
        tgt_name = (pair.target_employee.get("name") or "").lower()
        tgt_id = (pair.target_employee.get("employee_id") or "").lower()
        cand_name = (pair.candidate_employee.get("name") or "").lower()
        cand_id = (pair.candidate_employee.get("employee_id") or "").lower()

        target_raw = self.raw_employee_data.get(tgt_name) or self.raw_employee_data.get(tgt_id) or {}
        cand_raw   = self.raw_employee_data.get(cand_name) or self.raw_employee_data.get(cand_id) or {}

        system_msg = (
            "You are an expert HR analyst. Determine if a competitor employee is the best cross-company "
            "match for a target employee based on role/department/level/context. "
            "Output strict JSON with keys: is_match (true/false), confidence (0..1), reason (short). "
            "Be conservative: only mark true if the candidate clearly represents the same function/scope."
        )

        user_payload = {"target": {"basic": pair.target_employee, "raw": target_raw},
                        "candidate": {"basic": pair.candidate_employee, "raw": cand_raw}}

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Decide match for the following JSON:\n{json.dumps(user_payload, ensure_ascii=False)}\n"
                                        "Return ONLY JSON: {\"is_match\": bool, \"confidence\": number, \"reason\": string}."}
        ]

        for attempt in range(retries + 1):
            try:
                resp = self.openai_client.chat.completions.create(
                    model=self.azure_deployment_id,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=150
                )
                text = resp.choices[0].message.content.strip()
                json_text = self._extract_json(text)
                data = json.loads(json_text)

                is_match = bool(data.get("is_match", False))
                conf = float(data.get("confidence", 0.0))
                reason = str(data.get("reason", "")).strip()

                if is_match and conf < self.gpt_match_threshold:
                    is_match = False

                return VerifiedMatch(
                    target_employee=pair.target_employee,
                    candidate_employee=pair.candidate_employee,
                    gpt_is_match=is_match,
                    gpt_confidence=round(conf, 3),
                    gpt_reason=reason or "—",
                    fuzzy_score=pair.similarity_score
                )
            except Exception as e:
                if attempt < retries:
                    wait = backoff_sec * (2 ** attempt)
                    logger.warning(f"GPT verify failed (attempt {attempt+1}/{retries+1}): {e} → retrying in {wait:.1f}s")
                    time.sleep(wait)
                else:
                    logger.error(f"GPT verify failed permanently: {e}")
                    return VerifiedMatch(
                        target_employee=pair.target_employee,
                        candidate_employee=pair.candidate_employee,
                        gpt_is_match=False,
                        gpt_confidence=0.0,
                        gpt_reason=f"GPT error: {e}",
                        fuzzy_score=pair.similarity_score
                    )

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        return '{"is_match": false, "confidence": 0.0, "reason": "Malformed response"}'

    # -------------------- Export --------------------
    def export_reports(
        self,
        fuzzy_matches: List[FuzzyMatch],
        verified_matches: List[VerifiedMatch],
        out_dir: str = "output/matching_results"
    ):
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Fuzzy (stage 1) report
        fuzzy_path = os.path.join(out_dir, "stage1_fuzzy_matches.json")
        with open(fuzzy_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "target_company": self.target_company,
                    "target_employees": len(self.target_employees),
                    "competitor_employees": len(self.competitor_employees),
                    "candidate_pairs": len(fuzzy_matches),
                    "fuzzy_min_score": self.fuzzy_min_score,
                    "max_candidates_per_target": self.max_candidates_per_target
                },
                "matches": [asdict(m) for m in fuzzy_matches]
            }, f, indent=2, ensure_ascii=False)

        # Final verified (stage 2) report
        final_rows = []
        for vm in verified_matches:
            row = {
                "target": vm.target_employee,
                "candidate": vm.candidate_employee,
                "fuzzy_score": vm.fuzzy_score,
                "gpt_is_match": vm.gpt_is_match,
                "gpt_confidence": vm.gpt_confidence,
                "gpt_reason": vm.gpt_reason
            }
            final_rows.append(row)

        verified_path = os.path.join(out_dir, "stage2_verified_matches.json")
        with open(verified_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "target_company": self.target_company,
                    "verified_pairs": len(verified_matches),
                    "gpt_enabled": self.use_gpt,
                    "gpt_match_threshold": self.gpt_match_threshold
                },
                "verified_matches": final_rows
            }, f, indent=2, ensure_ascii=False)

        # Compact grouped report
        final_map: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"target": None, "confirmed_matches": []})
        for vm in verified_matches:
            tid = vm.target_employee["employee_id"]
            if final_map[tid]["target"] is None:
                final_map[tid]["target"] = vm.target_employee
            if vm.gpt_is_match:
                final_map[tid]["confirmed_matches"].append({
                    "candidate": vm.candidate_employee,
                    "confidence": vm.gpt_confidence,
                    "reason": vm.gpt_reason,
                    "fuzzy_score": vm.fuzzy_score
                })

        compact_path = os.path.join(out_dir, "final_matching_report.json")
        with open(compact_path, "w", encoding="utf-8") as f:
            json.dump({
                "target_company": self.target_company,
                "targets_covered": len(final_map),
                "report": list(final_map.values())
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Exported:\n  • {fuzzy_path}\n  • {verified_path}\n  • {compact_path}")

    # -------------------- Utils --------------------
    @staticmethod
    def _is_target_company(company_name: str) -> bool:
        s = re.sub(r'[^a-z0-9]', '', company_name.lower())
        # Accept any folder containing both 'manipal' and 'fin'
        return ('manipal' in s) and ('fin' in s)

# -------------------- Main --------------------
def main():
    # Azure env (required for GPT verification)
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")

    matcher = EmployeeMatcher(
        target_company="Manipal",
        openai_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment_id=azure_deployment_id,
        fuzzy_min_score=0.42,
        max_candidates_per_target=5,
        gpt_enabled_default=True,
        gpt_max_pairs_per_target=8,
        gpt_match_threshold=0.55
    )

    matcher.load_normalized_data("normalized_departments")
    matcher.load_raw_employee_data("employee_data")  # optional but helps GPT

    # Stage 1
    fuzzy = matcher.run_fuzzy_matching()

    # Stage 2
    verified = matcher.verify_with_gpt(fuzzy)

    # Export
    matcher.export_reports(fuzzy, verified, out_dir="output/matching_results")

    logger.info("🏁 Matching pipeline completed.")

if __name__ == "__main__":
    main()
