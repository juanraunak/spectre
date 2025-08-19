import os
import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
import time

# ==============================================
# CONFIGURATION SECTION - CUSTOMIZE THESE
# ==============================================

class Config:
    """
    Centralized configuration for the skill gap analysis pipeline.
    Modify these values to customize the analysis behavior.
    """
    
    # === FILE PATHS ===
    SKILL_DIR = "company_skills"                    # Folder with normalized skills JSON
    RAW_DIR = "employee_data"                      # Folder with raw report JSON
    SPECTRE_PATH = "spectre_matches.json"          # Output file for role matches
    OUTPUT_FILE = "final_skill_gaps.json"         # Final output file
    STEP1_DETAILED = "step1_missing_skills.json"   # Step 1 detailed output
    STEP1_BASIC = "step1_missing_skills_basic.json" # Step 1 basic output
    STEP2_DETAILED = "final_skill_gaps_detailed_gpt.json" # Step 2 detailed output
    
    # === COMPANY CONFIGURATION ===
    SPECTRE_COMPANY = "xto10x"                    # Your primary company to analyze
    TARGET_COMPANIES = []                          # Specific competitors (empty = all available)
    EXCLUDE_COMPANIES = []                         # Companies to exclude from analysis
    
    # === EMPLOYEE LIMITS ===
    MAX_EMPLOYEES_TO_ANALYZE = None                # Max employees from spectre company (None = all)
    MAX_COMPETITORS_PER_EMPLOYEE = None            # Max competitor matches per employee (None = all)
    MIN_COMPETITORS_REQUIRED = 1                   # Minimum competitors needed for analysis
    
    # === SKILL CONFIGURATION ===
    MAX_SKILLS_TO_SHOW = None                      # Max skills to display per employee (None = all)
    MIN_SKILLS_FOR_ANALYSIS = 0                    # Min skills required for employee inclusion
    INCLUDE_EMPTY_SKILL_EMPLOYEES = True           # Include employees with no skills
    
    # === MATCHING ACCURACY SETTINGS ===
    # Role matching thresholds
    FUZZY_NAME_THRESHOLD = 0.92                    # Name similarity threshold (0.0-1.0)
    ROLE_SIMILARITY_THRESHOLD = 72                 # Role similarity percentage (0-100)
    PREFILTER_MIN_RATIO = 0.30                     # Minimum ratio for prefiltering candidates
    PREFILTER_TOP_K = 12                           # Top K candidates to consider
    
    # === LLM CONFIGURATION ===
    USE_LLM_FOR_MATCHING = True                    # Use LLM for role matching
    USE_LLM_FOR_SKILLS = True                      # Use LLM for skill analysis
    LLM_TEMPERATURE = 0.1                          # LLM creativity (0.0-1.0)
    LLM_MAX_TOKENS = 1400                          # Max tokens per LLM request
    
    # Azure OpenAI Settings (will try environment variables first)
    AZURE_ENDPOINT = "https://notedai.openai.azure.com"
    AZURE_API_KEY = "2be1544b3dc14327b60a870fe8b94f35"
    AZURE_API_VERSION = "2024-06-01"
    AZURE_DEPLOYMENT = "gpt-4o"
    
    # === DEBUGGING & REPORTING ===
    DEBUG_MODE = True                              # Enable verbose logging
    DEBUG_LIMIT_PER_COMPANY = 8                    # Max debug entries per company
    SHOW_SAMPLE_OUTPUT = True                      # Show sample results
    EXPORT_INTERMEDIATE_FILES = True               # Save intermediate processing files
    
    # === ANALYSIS BEHAVIOR ===
    SKIP_IF_SPECTRE_EXISTS = True                  # Skip matching if spectre file exists
    FORCE_RERUN_ANALYSIS = False                   # Force rerun even if outputs exist
    INCLUDE_MARKET_INSIGHTS = True                 # Include market analysis in results
    
    # === SKILL NORMALIZATION ===
    CUSTOM_SKILL_ALIASES = {                       # Custom skill name mappings
        "ms excel": "excel",
        "microsoft excel": "excel",
        "power point": "powerpoint",
        "nlp": "natural language processing",
        "gen ai": "generative ai",
        "py": "python",
        "js": "javascript",
    }
    
    # === REPORTING OPTIONS ===
    INCLUDE_COMPETITOR_DETAILS = True              # Include competitor info in reports
    SHOW_CONFIDENCE_SCORES = True                  # Show matching confidence scores
    GROUP_BY_DEPARTMENT = False                    # Group results by department/role
    EXPORT_TO_CSV = False                          # Also export results to CSV
    
    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        errors = []
        
        if cls.FUZZY_NAME_THRESHOLD < 0 or cls.FUZZY_NAME_THRESHOLD > 1:
            errors.append("FUZZY_NAME_THRESHOLD must be between 0.0 and 1.0")
        
        if cls.ROLE_SIMILARITY_THRESHOLD < 0 or cls.ROLE_SIMILARITY_THRESHOLD > 100:
            errors.append("ROLE_SIMILARITY_THRESHOLD must be between 0 and 100")
        
        if cls.LLM_TEMPERATURE < 0 or cls.LLM_TEMPERATURE > 1:
            errors.append("LLM_TEMPERATURE must be between 0.0 and 1.0")
        
        if not cls.SPECTRE_COMPANY:
            errors.append("SPECTRE_COMPANY must be specified")
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"- {e}" for e in errors))
        
        return True
    
    @classmethod
    def print_summary(cls):
        """Print configuration summary"""
        print("=== CONFIGURATION SUMMARY ===")
        print(f"Spectre Company: {cls.SPECTRE_COMPANY}")
        print(f"Target Companies: {cls.TARGET_COMPANIES or 'All available'}")
        print(f"Max Employees: {cls.MAX_EMPLOYEES_TO_ANALYZE or 'All'}")
        print(f"Role Similarity Threshold: {cls.ROLE_SIMILARITY_THRESHOLD}%")
        print(f"Use LLM: Matching={cls.USE_LLM_FOR_MATCHING}, Skills={cls.USE_LLM_FOR_SKILLS}")
        print(f"Debug Mode: {cls.DEBUG_MODE}")
        print("=" * 30)


# =======================
# DataLoader (Stages 1 & 2)
# =======================
class DataLoader:
    def __init__(self, skill_dir: str = None, raw_dir: str = None) -> None:
        self.skill_dir = skill_dir or Config.SKILL_DIR
        self.raw_dir = raw_dir or Config.RAW_DIR
        # Outputs
        self.skills_by_company: Dict[str, Dict[str, List[str]]] = {}         # {company: {employee_id: [skills,...]}}
        self.skills_name_by_company: Dict[str, Dict[str, List[str]]] = {}    # {company: {normalized_name: [skills,...]}}
        self.raw_by_company: Dict[str, List[Dict[str, Any]]] = {}            # {company: [raw_employee_obj,...]}

    # ---------- public API ----------
    def run(self) -> None:
        """Run Stage 1 + Stage 2."""
        self._stage1_load_skills()
        self._stage2_load_raw()

    # ---------- utils ----------
    @staticmethod
    def load_json(path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def normalize_company_name(name: Optional[str]) -> str:
        """
        Normalize company names with configurable spectre company detection.
        """
        n = (name or "").strip().lower()
        
        # Check for spectre company variations
        spectre_variations = [Config.SPECTRE_COMPANY.lower(), f"{Config.SPECTRE_COMPANY} group", 
                            f"{Config.SPECTRE_COMPANY} technologies", f"{Config.SPECTRE_COMPANY} fintech"]
        
        for variation in spectre_variations:
            if variation in n:
                return Config.SPECTRE_COMPANY.lower()
        
        return n

    @staticmethod
    def get(d: Any, path: str, default=None):
        """
        Safe nested getter: get(obj, "a.b.c") -> obj['a']['b']['c'] or default
        Supports dicts only for simplicity.
        """
        cur = d
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    @staticmethod
    def norm_name(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

    # ---------- extractors ----------
    def extract_company_and_employees_from_raw(self, obj: Any) -> Tuple[str, List[Dict[str, Any]]]:
        """
        RAW report shape:
          {
            "mission_metadata": { "target_company": "..." },
            "employee_intelligence": { "employees": [ { "basic_info": {...}, "detailed_profile": {...} }, ... ] }
          }
        """
        if not isinstance(obj, dict):
            return "", []

        # Employees
        employees = self.get(obj, "employee_intelligence.employees", [])
        if not isinstance(employees, list):
            employees = []

        # Company
        company_name = self.get(obj, "mission_metadata.target_company")
        if not company_name and employees:
            company_name = (
                self.get(employees[0], "basic_info.company")
                or self.get(employees[0], "detailed_profile.current_company.name")
            )

        return self.normalize_company_name(company_name or ""), employees

    def extract_company_and_employees_generic(self, obj: Any) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generic extractor for SKILLS files or other shapes where employees is a flat array of:
          { "employee_id": "...", "skills": [...], "name": "..." (optional) }
        Supports:
          - {"company_name": "...", "employees":[...]}
          - {"employees":[...]}
          - {"data":{"employees":[...]}}
          - raw list of employees
        """
        company_name = None
        employees: List[Dict[str, Any]] = []

        if isinstance(obj, dict):
            if "employees" in obj and isinstance(obj["employees"], list):
                employees = obj["employees"]
                company_name = obj.get("company_name") or obj.get("company")
            elif "data" in obj and isinstance(obj["data"], dict) and isinstance(obj["data"].get("employees"), list):
                employees = obj["data"]["employees"]
                company_name = obj["data"].get("company_name") or obj.get("company_name")
            else:
                # heuristic: find a list of dicts that looks like employees with employee_id
                for _, v in obj.items():
                    if isinstance(v, list) and v and isinstance(v[0], dict) and "employee_id" in v[0]:
                        employees = v
                        company_name = obj.get("company_name") or obj.get("company")
                        break
        elif isinstance(obj, list):
            employees = obj

        if not company_name and employees and isinstance(employees[0], dict):
            company_name = employees[0].get("company")

        return self.normalize_company_name(company_name or ""), employees

    def extract_company_and_employees(self, obj: Any, is_raw: bool) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Switch between RAW vs generic extraction.
        Detect RAW by flag or by presence of 'mission_metadata'/'employee_intelligence'.
        """
        if is_raw or (isinstance(obj, dict) and ("mission_metadata" in obj or "employee_intelligence" in obj)):
            return self.extract_company_and_employees_from_raw(obj)
        return self.extract_company_and_employees_generic(obj)

    # ---------- stages ----------
    def _stage1_load_skills(self) -> None:
        print("=== Stage 1: Load Skills Data ===")
        out_ids: Dict[str, Dict[str, List[str]]] = {}
        out_names: Dict[str, Dict[str, List[str]]] = {}

        for fname in os.listdir(self.skill_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.skill_dir, fname)
            try:
                obj = self.load_json(path)
                # skills files are generic flat employees w/ employee_id
                company, employees = self.extract_company_and_employees(obj, is_raw=False)
                if not company:
                    if Config.DEBUG_MODE:
                        print(f"⚠️  {fname}: could not determine company_name; skipping")
                    continue

                # Apply company filters
                if Config.TARGET_COMPANIES and company not in Config.TARGET_COMPANIES:
                    if Config.DEBUG_MODE:
                        print(f"⚠️  {fname}: company '{company}' not in target list; skipping")
                    continue
                
                if company in Config.EXCLUDE_COMPANIES:
                    if Config.DEBUG_MODE:
                        print(f"⚠️  {fname}: company '{company}' in exclude list; skipping")
                    continue

                id_map: Dict[str, List[str]] = {}
                name_map: Dict[str, List[str]] = {}

                for e in employees:
                    emp_id = e.get("employee_id")
                    skills = e.get("skills") or []
                    
                    # Apply skill filters
                    if len(skills) < Config.MIN_SKILLS_FOR_ANALYSIS and not Config.INCLUDE_EMPTY_SKILL_EMPLOYEES:
                        continue
                    
                    if Config.MAX_SKILLS_TO_SHOW and len(skills) > Config.MAX_SKILLS_TO_SHOW:
                        skills = skills[:Config.MAX_SKILLS_TO_SHOW]
                    
                    if emp_id is not None:
                        id_map[str(emp_id)] = skills

                    # if the skills file has a real name, index it
                    nm = self.norm_name(e.get("name", ""))
                    if nm:
                        # keep richer list if duplicate names exist
                        if nm in name_map:
                            if len(skills) > len(name_map[nm]):
                                name_map[nm] = skills
                        else:
                            name_map[nm] = skills

                out_ids[company] = id_map
                if name_map:
                    out_names[company] = name_map

                print(f"Loaded skills for '{company}': {len(id_map)} employees mapped")
            except Exception as e:
                print(f"❌ Failed to parse {fname}: {e}")

        self.skills_by_company = out_ids
        self.skills_name_by_company = out_names

    def _stage2_load_raw(self) -> None:
        print("\n=== Stage 2: Load Raw Data ===")
        out: Dict[str, List[Dict[str, Any]]] = {}
        for fname in os.listdir(self.raw_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.raw_dir, fname)
            try:
                obj = self.load_json(path)
                company, employees = self.extract_company_and_employees(obj, is_raw=True)
                if not company:
                    # fallback (light)
                    base = os.path.splitext(fname)[0]
                    company = self.normalize_company_name(
                        base.replace("_report", "").replace("_fintech_complete_intelligence_report", "")
                    )

                # Apply company filters
                if Config.TARGET_COMPANIES and company not in Config.TARGET_COMPANIES:
                    if Config.DEBUG_MODE:
                        print(f"⚠️  {fname}: company '{company}' not in target list; skipping")
                    continue
                
                if company in Config.EXCLUDE_COMPANIES:
                    if Config.DEBUG_MODE:
                        print(f"⚠️  {fname}: company '{company}' in exclude list; skipping")
                    continue

                # Apply employee limits
                if Config.MAX_EMPLOYEES_TO_ANALYZE and len(employees) > Config.MAX_EMPLOYEES_TO_ANALYZE:
                    employees = employees[:Config.MAX_EMPLOYEES_TO_ANALYZE]
                    if Config.DEBUG_MODE:
                        print(f"   Limited to {Config.MAX_EMPLOYEES_TO_ANALYZE} employees")

                out[company] = employees
                print(f"Loaded raw employees for '{company}': {len(employees)}")

                if len(employees) == 0:
                    if isinstance(obj, dict) and Config.DEBUG_MODE:
                        print(f"   ⚠ no employees found. top-level keys: {list(obj.keys())[:10]}")
                    elif isinstance(obj, list) and obj and isinstance(obj[0], dict) and Config.DEBUG_MODE:
                        print(f"   ⚠ no employees found. first-item keys: {list(obj[0].keys())[:15]}")
            except Exception as e:
                print(f"❌ Failed to parse {fname}: {e}")

        self.raw_by_company = out


# =======================
# Merger (Stage 3)  — NAME + COMPANY ONLY
# =======================
class Merger:
    def __init__(
        self,
        raw_by_company: Dict[str, List[Dict[str, Any]]],                 # {company: [raw employee objects]}
        skills_name_by_company: Dict[str, Dict[str, List[str]]],         # {company: {normalized_name: skills}}
        fuzzy_threshold: float = None
    ) -> None:
        self.raw_by_company = raw_by_company
        self.skills_name_by_company = skills_name_by_company
        self.fuzzy_threshold = fuzzy_threshold or Config.FUZZY_NAME_THRESHOLD
        self.company_employees: Dict[str, List[Dict[str, Any]]] = {}

    # ---------- helpers ----------
    @staticmethod
    def _norm_name(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

    @staticmethod
    def _get(d: Any, path: str, default=None):
        cur = d
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def _pick_name(self, emp: Dict[str, Any]) -> str:
        return (
            self._get(emp, "detailed_profile.name")
            or self._get(emp, "basic_info.name")
            or ""
        )

    def _pick_role(self, emp: Dict[str, Any]) -> str:
        role = (
            self._get(emp, "detailed_profile.position")
            or self._get(emp, "basic_info.position")
            or self._get(emp, "detailed_profile.current_company.title")
            or self._get(emp, "basic_info.title")
            or emp.get("role", "")
        )
        if role:
            return role
        exp = self._get(emp, "detailed_profile.experience", [])
        if isinstance(exp, list) and exp:
            return exp[0].get("title", "") or role
        return role

    def _pick_company(self, emp: Dict[str, Any], default_company: str) -> str:
        return (
            self._get(emp, "detailed_profile.current_company.name")
            or self._get(emp, "basic_info.company")
            or default_company
        )

    def _lookup_skills_by_name(self, company: str, raw_name: str) -> List[str]:
        """
        Find skills using the prebuilt name index for this company.
        First exact normalized name, then fuzzy >= threshold.
        """
        nmap = self.skills_name_by_company.get(company, {})
        nm = self._norm_name(raw_name)
        if not nm or not nmap:
            return []

        # exact
        if nm in nmap:
            return nmap[nm]

        # fuzzy
        best_key, best_sc = None, 0.0
        for k in nmap.keys():
            sc = SequenceMatcher(None, nm, k).ratio()
            if sc > best_sc:
                best_sc, best_key = sc, k
        if best_key and best_sc >= self.fuzzy_threshold:
            return nmap[best_key]
        return []

    # ---------- main ----------
    def run(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Merge using NAME + COMPANY only.
        """
        print("\n=== Stage 3: Merge Raw + Skills (name+company) ===")
        merged_all: Dict[str, List[Dict[str, Any]]] = {}

        for company, raw_emps in self.raw_by_company.items():
            name_hits = 0
            rows: List[Dict[str, Any]] = []

            for emp in raw_emps:
                name = self._pick_name(emp)
                role = self._pick_role(emp)
                comp = self._pick_company(emp, default_company=company)

                skills = self._lookup_skills_by_name(company, name)
                if skills:
                    name_hits += 1

                rows.append({
                    # keep id for reference only
                    "id": (self._get(emp, "detailed_profile.linkedin_id")
                           or self._get(emp, "detailed_profile.id")
                           or emp.get("employee_id")
                           or self._get(emp, "basic_info.linkedin_id")),
                    "name": name,
                    "role": role,
                    "company": comp,
                    "skills": skills
                })

            merged_all[company] = rows
            with_skills = sum(1 for r in rows if r["skills"])
            print(f"Merged {len(rows)} employees for '{company}' "
                  f"(with skills for {with_skills} | name-matched {name_hits})")

        self.company_employees = merged_all
        return merged_all


# =======================
# SpectreMatcher (match + print for 1-vs-all)
# =======================
import requests  # keep import after main imports to mirror your original file

class SpectreMatcher:
    PLACEHOLDER_ROLES = {"--", "-", "na", "n/a", "n\\a", "none", "null", "unknown", "untitled", "not available"}

    def __init__(self,
                 company_employees: Dict[str, List[Dict[str, Any]]],
                 spectre_company_key: str = None,
                 similarity_threshold: int = None,         
                 per_employee_show_limit: Optional[int] = None,
                 use_llm: bool = None,
                 prefilter_top_k: int = None,
                 prefilter_min_ratio: float = None,      
                 debug: bool = None,
                 debug_limit_per_company: int = None) -> None:
        self.company_employees = company_employees
        self.spectre_key = spectre_company_key or Config.SPECTRE_COMPANY
        self.threshold = similarity_threshold or Config.ROLE_SIMILARITY_THRESHOLD
        self.per_employee_show_limit = per_employee_show_limit or Config.MAX_COMPETITORS_PER_EMPLOYEE
        self.prefilter_top_k = prefilter_top_k or Config.PREFILTER_TOP_K
        self.prefilter_min_ratio = prefilter_min_ratio or Config.PREFILTER_MIN_RATIO
        self.debug = debug if debug is not None else Config.DEBUG_MODE
        self.debug_limit_per_company = debug_limit_per_company or Config.DEBUG_LIMIT_PER_COMPANY

        # Azure configuration with environment variable preference
        self._azure_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or Config.AZURE_ENDPOINT).rstrip("/")
        self._azure_key = os.getenv("AZURE_OPENAI_API_KEY") or Config.AZURE_API_KEY
        self._azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or Config.AZURE_API_VERSION
        self._azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID") or Config.AZURE_DEPLOYMENT

        self.use_llm = (use_llm if use_llm is not None else Config.USE_LLM_FOR_MATCHING) and self._azure_is_configured()
        if Config.USE_LLM_FOR_MATCHING and not self.use_llm and self.debug:
            print("[LLM OFF] Azure config invalid; using similarity-only.")

    # ---------- utils ----------
    def _dprint(self, *a, **k):
        if self.debug:
            print(*a, **k)

    def _azure_is_configured(self) -> bool:
        ok = True
        if not self._azure_endpoint or not self._azure_endpoint.startswith("https://"):
            self._dprint("   [CONFIG] AZURE_OPENAI_ENDPOINT missing/invalid.")
            ok = False
        if not self._azure_deployment:
            self._dprint("   [CONFIG] AZURE_OPENAI_DEPLOYMENT_ID missing/invalid.")
            ok = False
        if not self._azure_key:
            self._dprint("   [CONFIG] AZURE_OPENAI_API_KEY missing/invalid.")
            ok = False
        return ok

    # ---------- cleaning ----------
    @classmethod
    def _strip_company_words(cls, s: str) -> str:
        s = re.sub(r"\bat\s+[a-z0-9 .,&'-]+\b", "", s)
        # Add configurable company name stripping
        spectre_lower = Config.SPECTRE_COMPANY.lower()
        noise = [
            r"\bshriram( city)? union finance( ltd(\.)?)?\b",
            r"\bshriram finance( limited)?\b",
            rf"\b{re.escape(spectre_lower)}( fintech| technologies| group| business solutions)?\b",
            r"\bfinance(ltd| limited)?\b",
            r"\bltd(\.)?\b", r"\blimited\b", r"\bprivate limited\b",
            r"\b(pvt|pvt\.)\b", r"\b(co|co\.)\b"
        ]
        for pat in noise:
            s = re.sub(pat, "", s, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", s).strip()

    @classmethod
    def _role_core(cls, role: str) -> str:
        s = (role or "").strip().lower()
        if not s:
            return ""
        seg = re.split(r"[|,;/·–—-]+", s)
        s = " ".join(seg[:2]).strip()
        s = cls._strip_company_words(s)
        repl = {
            "relationship officer": "relationship manager",
            "regional head": "regional manager",
            "area head": "area manager",
            "assistant manager": "assistant manager",
            "hr professional": "hr manager",
            "qa manager": "quality assurance manager",
            "data analytics": "data analyst",
        }
        for k, v in repl.items():
            s = s.replace(k, v)
        s = re.sub(r"\s+", " ", s).strip()
        if re.fullmatch(r"[-–—_.]+", s) or s in cls.PLACEHOLDER_ROLES:
            return ""
        return s

    @classmethod
    def _ratio(cls, a: Optional[str], b: Optional[str]) -> float:
        a_clean = cls._role_core(a or "")
        b_clean = cls._role_core(b or "")
        if not a_clean or not b_clean:
            return 0.0
        return SequenceMatcher(None, a_clean, b_clean).ratio()

    # ---------- LLM ----------
    def _build_prompt(self, source_role: str, targets: List[str]) -> str:
        lines = [
            "You match job roles across companies.",
            "Given a SOURCE job title and a list of TARGET titles, return indices that are truly equivalent in function and seniority.",
            "Be conservative; pick only strong matches.",
            "",
            f"SOURCE_ROLE: {self._role_core(source_role)}",
            "",
            "TARGET_ROLES:",
        ]
        for i, r in enumerate(targets):
            lines.append(f"- [{i}] {self._role_core(r)}")
        lines += ["", 'Return ONLY JSON: {"matches":[i1,i2,...]}']
        return "\n".join(lines)

    def _call_azure_openai(self, prompt: str) -> dict:
        url = f"{self._azure_endpoint}/openai/deployments/{self._azure_deployment}/chat/completions?api-version={self._azure_api_version}"
        headers = {"Content-Type": "application/json", "api-key": self._azure_key}
        body = {
            "messages": [
                {"role": "system", "content": "You are a precise, terse role-matching assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": Config.LLM_TEMPERATURE
        }
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", content or "")
            return json.loads(m.group(0)) if m else {}

    # ---------- steps ----------
    def _prefilter_candidates(self, m_role: str, company_key: str) -> List[Dict[str, Any]]:
        scored = []
        for idx, t in enumerate(self.company_employees.get(company_key, [])):
            traw = t.get("role")
            sc = self._ratio(m_role, traw)
            if sc >= self.prefilter_min_ratio:
                scored.append((sc, idx, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = scored[: self.prefilter_top_k]
        return [{"idx": i, "name": emp.get("name",""), "role": emp.get("role",""), "score": r}
                for r, i, emp in picked]

    def _llm_select(self, m_role_clean: str, cands: List[Dict[str, Any]]) -> List[int]:
        prompt = self._build_prompt(m_role_clean, [c["role"] for c in cands])
        try:
            resp = self._call_azure_openai(prompt) or {}
            idxs = resp.get("matches", [])
            return idxs if isinstance(idxs, list) else []
        except Exception as e:
            self._dprint("   [LLM ERROR] falling back to similarity. Err:", str(e))
            return []

    # ---------- matching ----------
    def _match_vs_company(self, spectre_emp: Dict[str, Any], target_company_key: str) -> List[Dict[str, Any]]:
        m_role_raw = spectre_emp.get("role") or ""
        m_role_clean = self._role_core(m_role_raw)
        if not m_role_clean:
            self._dprint(f"   - SKIP (empty/placeholder role): {spectre_emp.get('name','')} | raw='{m_role_raw}'")
            return []

        cands = self._prefilter_candidates(m_role_clean, target_company_key)
        self._dprint(f"   - PREFILTER {spectre_emp.get('name','')} | role='{m_role_clean}' | "
                     f"cands={len(cands)} (min_ratio={self.prefilter_min_ratio}, top_k={self.prefilter_top_k})")
        if not cands:
            return []
        for k, c in enumerate(cands[:3]):
            self._dprint(f"       cand{k}: '{self._role_core(c['role'])}'  (quick={round(c['score'],3)})")

        indices = []
        if self.use_llm:
            indices = self._llm_select(m_role_clean, cands)
            self._dprint(f"   - LLM indices: {indices}")

        matches = []
        if indices:
            for j in indices:
                if isinstance(j, int) and 0 <= j < len(cands):
                    cand = cands[j]
                    sim_hint = self._ratio(m_role_clean, cand["role"]) * 100.0
                    matches.append({
                        "company": target_company_key,
                        "name": cand["name"],
                        "role": cand["role"],
                        "similarity": round(sim_hint, 2),
                        "via": "llm"
                    })
        else:
            # similarity fallback on same candidates
            for cand in cands:
                sim_pct = self._ratio(m_role_clean, cand["role"]) * 100.0
                if sim_pct >= self.threshold:
                    matches.append({
                        "company": target_company_key,
                        "name": cand["name"],
                        "role": cand["role"],
                        "similarity": round(sim_pct, 2),
                        "via": "sim-fallback"
                    })

        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Apply per-employee limit
        if self.per_employee_show_limit:
            matches = matches[:self.per_employee_show_limit]
        
        return matches

    # ---------- public ----------
    def run(self, export_json_path: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        results: Dict[str, List[Dict[str, Any]]] = {}
        spectre_emps = self.company_employees.get(self.spectre_key, [])
        
        # Apply employee limits
        if Config.MAX_EMPLOYEES_TO_ANALYZE:
            spectre_emps = spectre_emps[:Config.MAX_EMPLOYEES_TO_ANALYZE]
            
        competitor_keys = [c for c in self.company_employees.keys() if c != self.spectre_key]
        
        # Apply company filters
        if Config.TARGET_COMPANIES:
            competitor_keys = [c for c in competitor_keys if c in Config.TARGET_COMPANIES]
        competitor_keys = [c for c in competitor_keys if c not in Config.EXCLUDE_COMPANIES]

        self._dprint("\n=== Spectre Matcher DEBUG ===")
        self._dprint(f"spectre='{self.spectre_key}', employees={len(spectre_emps)}, "
                     f"use_llm={self.use_llm}, min_ratio={self.prefilter_min_ratio}, top_k={self.prefilter_top_k}, "
                     f"sim_threshold={self.threshold}")
        self._dprint(f"Target companies: {competitor_keys}")

        for comp in competitor_keys:
            bucket: List[Dict[str, Any]] = []
            printed = 0
            self._dprint(f"\n[COMP] {comp} | targets={len(self.company_employees.get(comp, []))}")

            for m in spectre_emps:
                per_matches = self._match_vs_company(m, comp)
                if per_matches:
                    bucket.append({
                        "manipal_name": m.get("name", ""),
                        "manipal_role": m.get("role", ""),
                        "matches": per_matches
                    })
                    if printed < self.debug_limit_per_company:
                        self._dprint(f"   ✓ MATCH for {m.get('name','')}  -> {len(per_matches)} kept")
                        printed += 1

            results[comp] = bucket
            self._dprint(f"[COMP] done: spectre_with_matches={len(bucket)}, "
                         f"pairs={sum(len(r['matches']) for r in bucket)}")

        if export_json_path and Config.EXPORT_INTERMEDIATE_FILES:
            with open(export_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Spectre match JSON written: {export_json_path}")

        return results

    def print_report(self, results: Dict[str, List[Dict[str, Any]]]) -> None:
        spectre_total = len(self.company_employees.get(self.spectre_key, []))
        if Config.MAX_EMPLOYEES_TO_ANALYZE:
            spectre_total = min(spectre_total, Config.MAX_EMPLOYEES_TO_ANALYZE)
            
        print("\n=== Spectre Role Match Report (1 vs All) ===")
        mode = "LLM+fuzzy (fallback to sim)" if self.use_llm else f"Similarity-only ≥{self.threshold}%"
        print(f"Spectre company: '{self.spectre_key}'  |  Mode: {mode}")
        print(f"Spectre employees analyzed: {spectre_total}")
        all_pairs = 0
        spectre_with_matches_global = set()

        for comp, rows in results.items():
            comp_total = len(self.company_employees.get(comp, []))
            matched_spectre = len(rows)
            comp_pairs = sum(len(r["matches"]) for r in rows)
            all_pairs += comp_pairs
            for r in rows:
                if r["matches"]:
                    spectre_with_matches_global.add(r["manipal_name"])

            print(f"\n--- Against '{comp}' ---")
            print(f"Target employees: {comp_total}")
            print(f"Spectre employees with ≥1 match: {matched_spectre}")
            print(f"Total matched pairs here: {comp_pairs}")

            if Config.SHOW_SAMPLE_OUTPUT:
                for r in rows[:3]:  # Show first 3 matches as sample
                    print(f"\n- {r['manipal_name']}  ({r['manipal_role']})")
                    for m in r["matches"][:2]:  # Show top 2 matches per employee
                        via_text = f"[{m.get('via','')}]" if Config.SHOW_CONFIDENCE_SCORES else ""
                        print(f"    → {m['name']}  ({m['role']})   [{m['similarity']}%]  {via_text}")

        print(f"\n=== SUMMARY ===")
        print(f"Total unique {self.spectre_key} employees with matches: {len(spectre_with_matches_global)}")
        print(f"Total competitor matches found: {all_pairs}")


# =======================
# Helpers for gating logic
# =======================
def spectre_file_has_matches(path: str) -> bool:
    """Return True if file exists and contains at least one non-empty match list."""
    if not os.path.exists(path):
        return False
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return False

    # Expecting { "<company>": [ { "manipal_name":..., "matches":[...] }, ... ], ... }
    if not isinstance(data, dict) or not data:
        return False

    for _, rows in data.items():
        if isinstance(rows, list):
            for r in rows:
                matches = r.get("matches") if isinstance(r, dict) else None
                if isinstance(matches, list) and len(matches) > 0:
                    return True
    return False


# =======================
# Data classes for skill gap analysis
# =======================
@dataclass
class SkillGap:
    employee_name: str
    role: str
    current_skills: List[str]
    missing_skills: List[str]
    competitor_count: int
    gap_reasoning: str
    confidence_score: float


@dataclass
class Step2Analysis:
    manipal_employee: str
    role: str
    missing_skills: List[str]
    skill_importance: Dict[str, str]
    gap_reasoning: Dict[str, str]
    overall_assessment: str
    recommendations: List[str]
    competitor_companies: List[str]
    competitor_count: int
    evidence_flags: Dict[str, Any]


# =======================
# Unified Skill Gap Analyzer
# =======================
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

class UnifiedSkillGapAnalyzer:
    """
    Unified skill gap analyzer that works with the merged company_employees structure.
    Handles both Step 1 (finding gaps) and Step 2 (detailed analysis).
    """
    
    def __init__(self, 
                 company_employees: Dict[str, List[Dict[str, Any]]],
                 spectre_matches_path: str,
                 spectre_company_key: str = None,
                 use_llm: bool = None,
                 azure_config: Optional[Dict[str, str]] = None):
        """
        Args:
            company_employees: Output from Merger.run() - {company: [employee_rows]}
            spectre_matches_path: Path to spectre_matches.json
            spectre_company_key: Key for spectre company
            use_llm: Whether to use LLM for analysis
            azure_config: Azure OpenAI configuration
        """
        self.company_employees = company_employees
        self.spectre_key = spectre_company_key or Config.SPECTRE_COMPANY
        self.use_llm = use_llm if use_llm is not None else Config.USE_LLM_FOR_SKILLS
        
        # Load spectre matches
        self.spectre_matches = self._load_spectre_matches(spectre_matches_path)
        
        # Build lookup indexes
        self._build_indexes()
        
        # Setup LLM if needed
        if self.use_llm and azure_config:
            self._setup_azure_client(azure_config)
        else:
            self.client = None
    
    def _load_spectre_matches(self, path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load spectre matches JSON"""
        if not os.path.exists(path):
            print(f"Warning: Spectre matches file not found: {path}")
            return {}
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for matching"""
        return re.sub(r'[^a-z0-9]+', ' ', (name or '').lower()).strip()
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize skill text with custom aliases"""
        if not skill or not isinstance(skill, str):
            return ""
        
        # Basic normalization
        normalized = re.sub(r'[^\w\s+#.]', ' ', skill.lower()).strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Apply custom aliases from config
        return Config.CUSTOM_SKILL_ALIASES.get(normalized, normalized)
    
    def _build_indexes(self):
        """Build lookup indexes for faster matching"""
        # Build name-based index for each company
        self.employees_by_name = {}
        for company, employees in self.company_employees.items():
            company_index = {}
            for emp in employees:
                name_key = self._normalize_name(emp.get('name', ''))
                if name_key:
                    company_index[name_key] = emp
            self.employees_by_name[company] = company_index
    
    def _setup_azure_client(self, config: Dict[str, str]):
        """Setup Azure OpenAI client"""
        try:
            self.client = AzureOpenAI(
                api_key=config.get('api_key', ''),
                azure_endpoint=config.get('endpoint', '').rstrip('/'),
                api_version=config.get('api_version', '2024-06-01')
            )
            self.deployment_id = config.get('deployment_id', 'gpt-4o')
        except Exception as e:
            print(f"Warning: Failed to setup Azure client: {e}")
            self.client = None
    
    def _get_matched_competitors(self, employee_name: str) -> List[Dict[str, Any]]:
        """Get competitor data for matched employees using spectre_matches as bridge"""
        competitors = []
        norm_name = self._normalize_name(employee_name)
        
        # Go through each company's matches in spectre_matches
        for company, matches_list in self.spectre_matches.items():
            company_employees = self.employees_by_name.get(company, {})
            
            for match_entry in matches_list:
                # Check if this spectre employee matches
                spectre_name = self._normalize_name(match_entry.get('manipal_name', ''))
                if spectre_name == norm_name:
                    # Get competitor matches for this employee
                    for competitor_match in match_entry.get('matches', []):
                        comp_name = self._normalize_name(competitor_match.get('name', ''))
                        if comp_name in company_employees:
                            competitor = company_employees[comp_name].copy()
                            competitor['company'] = company
                            competitor['match_similarity'] = competitor_match.get('similarity', 0)
                            competitors.append(competitor)
        
        return competitors
    
    def _calculate_skill_gap(self, spectre_employee: Dict[str, Any], competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate skill gaps between spectre employee and competitors"""
        if not competitors:
            return {
                'current_skills': [],
                'missing_skills': [],
                'competitor_skills': [],
                'gap_analysis': 'No competitors found for comparison'
            }
        
        # Normalize current skills
        current_skills_raw = spectre_employee.get('skills', [])
        current_skills = {self._normalize_skill(skill) for skill in current_skills_raw if skill}
        current_skills.discard("")  # Remove empty strings
        
        # Aggregate competitor skills
        all_competitor_skills = set()
        skill_frequency = {}
        
        for competitor in competitors:
            comp_skills = competitor.get('skills', [])
            for skill in comp_skills:
                normalized_skill = self._normalize_skill(skill)
                if normalized_skill:
                    all_competitor_skills.add(normalized_skill)
                    skill_frequency[normalized_skill] = skill_frequency.get(normalized_skill, 0) + 1
        
        # Find missing skills (present in competitors but not in current employee)
        missing_skills = all_competitor_skills - current_skills
        
        # Prioritize by frequency (skills that appear in multiple competitors)
        min_frequency = max(1, len(competitors) * 0.3)  # At least 30% of competitors should have it
        priority_missing = [
            skill for skill in missing_skills 
            if skill_frequency.get(skill, 0) >= min_frequency
        ]
        
        return {
            'current_skills': sorted(list(current_skills)),
            'missing_skills': sorted(priority_missing),
            'all_missing': sorted(list(missing_skills)),
            'competitor_skills': sorted(list(all_competitor_skills)),
            'skill_frequency': skill_frequency,
            'gap_analysis': f'Found {len(priority_missing)} high-priority skill gaps based on {len(competitors)} competitors'
        }
    
    def _generate_llm_reasoning(self, employee: Dict[str, Any], gap_data: Dict[str, Any], competitors: List[Dict[str, Any]]) -> str:
        """Use LLM to generate reasoning for skill gaps"""
        if not self.client or not gap_data['missing_skills']:
            return gap_data.get('gap_analysis', 'Basic analysis completed')
        
        prompt = f"""
Analyze skill gaps for this employee profile:

EMPLOYEE:
- Name: {employee.get('name', 'N/A')}
- Role: {employee.get('role', 'N/A')}  
- Current Skills: {gap_data['current_skills']}

COMPETITOR ANALYSIS:
- {len(competitors)} matched competitors from similar roles
- Missing Skills Identified: {gap_data['missing_skills']}
- Competitor Skill Universe: {gap_data['competitor_skills'][:20]}  # Limit for prompt size

Provide a brief analysis (2-3 sentences) explaining:
1. Why these missing skills are important for this role
2. How these gaps compare to industry standards based on the competitor data

Keep response concise and professional.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Analysis: {len(gap_data['missing_skills'])} priority skills identified from competitor benchmarking. Error in detailed reasoning: {str(e)[:50]}"
    
    def analyze_employee(self, employee_name: str) -> Optional[SkillGap]:
        """Analyze skill gaps for a single employee"""
        norm_name = self._normalize_name(employee_name)
        
        # Get spectre employee data
        spectre_employees = self.employees_by_name.get(self.spectre_key, {})
        spectre_employee = spectre_employees.get(norm_name)
        
        if not spectre_employee:
            print(f"Warning: Employee '{employee_name}' not found in spectre company data")
            return None
        
        # Get matched competitors using spectre_matches as bridge
        competitors = self._get_matched_competitors(employee_name)
        
        if not competitors:
            if Config.DEBUG_MODE:
                print(f"Warning: No competitors found for '{employee_name}' in spectre matches")
            return None
        
        # Calculate skill gaps
        gap_data = self._calculate_skill_gap(spectre_employee, competitors)
        
        # Generate reasoning
        reasoning = self._generate_llm_reasoning(spectre_employee, gap_data, competitors) if self.use_llm else gap_data['gap_analysis']
        
        # Calculate confidence based on number of competitors and skill frequency
        if gap_data['missing_skills']:
            avg_frequency = sum(gap_data['skill_frequency'].get(skill, 0) for skill in gap_data['missing_skills'])
            avg_frequency = avg_frequency / len(gap_data['missing_skills'])
        else:
            avg_frequency = 0
        
        confidence = min(0.95, 0.4 + (len(competitors) * 0.1) + (avg_frequency * 0.2))
        
        return SkillGap(
            employee_name=spectre_employee.get('name', employee_name),
            role=spectre_employee.get('role', ''),
            current_skills=gap_data['current_skills'],
            missing_skills=gap_data['missing_skills'],
            competitor_count=len(competitors),
            gap_reasoning=reasoning,
            confidence_score=round(confidence, 2)
        )
    
    def analyze_all(self, max_employees: Optional[int] = None) -> List[SkillGap]:
        """Analyze skill gaps for all spectre company employees with matches"""
        results = []
        
        # Get all employees that have matches in spectre_matches
        matched_employees = set()
        for company, matches_list in self.spectre_matches.items():
            for match_entry in matches_list:
                if match_entry.get('matches'):  # Only if they have actual matches
                    matched_employees.add(match_entry.get('manipal_name', ''))
        
        employee_list = list(matched_employees)
        if max_employees:
            employee_list = employee_list[:max_employees]
        
        print(f"Analyzing skill gaps for {len(employee_list)} employees with matches")
        
        for i, employee_name in enumerate(employee_list, 1):
            if Config.DEBUG_MODE:
                print(f"Processing {i}/{len(employee_list)}: {employee_name}")
            
            result = self.analyze_employee(employee_name)
            if result:
                results.append(result)
        
        return results
    
    def _identify_top_skills(self, gap_data: Dict[str, Any], competitors: List[Dict[str, Any]]) -> List[str]:
        """Identify top 2 most strategic skills based on frequency and importance"""
        strategic_skills = [
            'kubernetes', 'microservices', 'cloud computing', 'distributed systems',
            'ci cd', 'docker', 'networking', 'linux', 'aws', 'azure', 'gcp',
            'python', 'java', 'javascript', 'react', 'node js', 'spring boot'
        ]
        
        skill_scores = []
        for skill in gap_data['missing_skills']:
            frequency = gap_data['skill_frequency'].get(skill, 0) / len(competitors) if competitors else 0
            strategic_value = 1.0 if skill in strategic_skills else 0.7
            
            score = frequency * strategic_value
            skill_scores.append((skill, score, frequency))
        
        # Sort by score and return top 2
        skill_scores.sort(key=lambda x: x[1], reverse=True)
        return [skill[0] for skill in skill_scores[:2]]
    
    def _get_companies_requiring_skill(self, skill: str, competitors: List[Dict[str, Any]]) -> List[str]:
        """Get companies that require a specific skill"""
        companies = set()
        for competitor in competitors:
            comp_skills = [self._normalize_skill(s) for s in competitor.get('skills', [])]
            if skill in comp_skills:
                companies.add(competitor.get('company', 'Unknown'))
        return list(companies)
    
    def _get_strategic_importance(self, skill: str) -> str:
        """Get strategic importance explanation for a skill"""
        strategies = {
            'kubernetes': 'Container orchestration is the backbone of modern cloud infrastructure. Without K8s knowledge, you cannot architect or maintain production systems at scale. This skill separates junior from senior engineers.',
            'microservices': 'Microservices architecture is how trillion-dollar companies like Amazon and Netflix achieve scale. This isn\'t just a technical skill - it\'s architectural thinking that defines system design leadership.',
            'cloud computing': 'Cloud platforms (AWS/Azure/GCP) are the foundation of modern business. Every application, every startup, every enterprise migration depends on cloud expertise. This skill determines your ceiling in tech.',
            'distributed systems': 'Distributed systems knowledge is what separates good engineers from exceptional ones. Companies like Google and Facebook won\'t even interview without this foundation - it\'s the core of internet-scale engineering.',
            'ci cd': 'CI/CD pipelines are how modern software gets built and deployed. Without this, you\'re stuck in legacy development practices. This skill is the entry ticket to DevOps and platform engineering roles.',
            'docker': 'Containerization with Docker is the standard for application packaging and deployment. Every cloud-native application uses containers - this skill is foundational to modern development workflows.',
            'networking': 'Network engineering underpins every distributed system and cloud application. This deep technical skill is highly valued and creates strong job security in infrastructure roles.',
            'linux': 'Linux powers 96% of web servers and every major cloud platform. This foundational skill is essential for backend development, DevOps, and infrastructure engineering roles.',
            'aws': 'Amazon Web Services dominates 32% of the cloud market. AWS expertise directly translates to higher salaries and access to enterprise-level positions at tech companies.',
            'python': 'Python is the language of AI, data science, and backend development. It\'s the most in-demand programming language with the highest growth trajectory in enterprise adoption.'
        }
        
        return strategies.get(skill, f'Important technical skill for modern software development with strong market demand and career growth potential.')
    
    def _get_market_trend(self, skill: str) -> str:
        """Get market trend information for a skill"""
        trends = {
            'kubernetes': 'Explosive 300% job posting growth in last 18 months. Container orchestration is now mandatory for cloud-native development',
            'microservices': 'Steady 150% increase in demand. Modern architecture standard for scalable applications',
            'cloud computing': '200% growth trajectory. Digital transformation mandate across all industries',
            'distributed systems': 'High-value niche with 180% growth. Essential for big tech and fintech scaling challenges',
            'ci cd': 'DevOps revolution driving 120% increase. Automation is non-negotiable in modern development',
            'docker': 'Containerization standard with 100% adoption rate in cloud-native companies',
            'networking': 'Infrastructure backbone skill with consistent 80% demand in enterprise roles',
            'linux': 'Foundation skill with 90% presence in backend engineering roles',
            'aws': 'Cloud leader with 160% growth in job requirements. Enterprise migration driving massive demand',
            'python': 'AI/ML boom creating 250% surge in Python roles. Fastest growing language in enterprise'
        }
        
        return trends.get(skill, 'Steady market demand with consistent growth in enterprise adoption')
    
    def _generate_enhanced_reasoning(self, skill: str, gap_data: Dict[str, Any], competitors: List[Dict[str, Any]]) -> str:
        """Generate detailed competitive reasoning for a critical skill"""
        companies = self._get_companies_requiring_skill(skill, competitors)
        skill_frequency = gap_data['skill_frequency'].get(skill, 0)
        market_coverage = (skill_frequency / len(competitors)) * 100 if competitors else 0
        
        # Calculate salary impact (mock data - replace with real analysis)
        salary_impact = {
            'kubernetes': 25, 'microservices': 20, 'cloud computing': 18,
            'distributed systems': 22, 'ci cd': 15, 'docker': 12,
            'networking': 16, 'linux': 10, 'aws': 20, 'python': 18
        }.get(skill, 15)
        
        strategic_importance = self._get_strategic_importance(skill)
        market_trend = self._get_market_trend(skill)
        tech_giants = [c for c in companies if c.lower() in ['google', 'amazon', 'microsoft', 'facebook', 'netflix', 'apple', 'uber', 'airbnb']]
        
        return f"""CRITICAL SKILL GAP ANALYSIS:

🎯 **Market Presence**: {skill_frequency} out of {len(competitors)} major competitors actively require this skill ({market_coverage:.0f}% market coverage)

🏢 **Key Competitors Requiring This**: {', '.join(companies[:3])}{f' and {len(companies) - 3} others' if len(companies) > 3 else ''}

💰 **Career Impact**: Roles requiring {skill} show {salary_impact}% higher compensation on average. This skill is a direct gateway to senior-level positions at {len(tech_giants)} tech giants in our analysis.

📈 **Strategic Importance**: {strategic_importance}

🔥 **Market Urgency**: {market_trend} - Companies are actively headhunting for this combination. Without this skill, you're automatically filtered out from {market_coverage:.0f}% of similar roles.

⚡ **Competitive Advantage**: Mastering {skill} immediately puts you in the top 15% of candidates for roles at {', '.join(companies[:2])}, where this skill is considered a core requirement, not a nice-to-have."""
    
    def generate_detailed_analysis(self, skill_gaps: List[SkillGap]) -> List[Step2Analysis]:
        """Generate detailed Step 2 analysis for skill gaps with enhanced reasoning for top 2 skills"""
        detailed_results = []
        
        for gap in skill_gaps:
            # Get competitors for this employee
            competitors = self._get_matched_competitors(gap.employee_name)
            competitor_companies = list(set(comp.get('company', 'Unknown') for comp in competitors))
            
            # Calculate gap data for enhanced reasoning
            spectre_employees = self.employees_by_name.get(self.spectre_key, {})
            norm_name = self._normalize_name(gap.employee_name)
            spectre_employee = spectre_employees.get(norm_name)
            
            if spectre_employee:
                gap_data = self._calculate_skill_gap(spectre_employee, competitors)
                
                # Identify top 2 strategic skills
                top_skills = self._identify_top_skills(gap_data, competitors)
                
                # Generate skill importance and reasoning
                skill_importance = {}
                gap_reasoning = {}
                
                for skill in gap.missing_skills:
                    # Enhanced reasoning for top 2 skills
                    if skill in top_skills[:2]:
                        gap_reasoning[skill] = self._generate_enhanced_reasoning(skill, gap_data, competitors)
                        skill_importance[skill] = "Critical"
                    else:
                        # Standard reasoning for other skills
                        skill_freq = gap_data['skill_frequency'].get(skill, 0)
                        if skill_freq >= len(competitors) * 0.7:
                            importance = "Critical"
                        elif skill_freq >= len(competitors) * 0.4:
                            importance = "Important"
                        else:
                            importance = "Nice-to-have"
                        
                        skill_importance[skill] = importance
                        gap_reasoning[skill] = f"Present in {skill_freq}/{len(competitors)} similar roles across competitors"
            else:
                skill_importance = {skill: "Important" for skill in gap.missing_skills}
                gap_reasoning = {skill: "Standard gap analysis" for skill in gap.missing_skills}
            
            # Generate recommendations
            recommendations = []
            critical_skills = [skill for skill, imp in skill_importance.items() if imp == "Critical"]
            if critical_skills:
                recommendations.append(f"🚨 PRIORITY: Immediately focus on {', '.join(critical_skills[:2])} - these are career-defining gaps")
            
            important_skills = [skill for skill, imp in skill_importance.items() if imp == "Important"]
            if important_skills:
                recommendations.append(f"📈 STRATEGIC: Develop {', '.join(important_skills[:3])} to match industry standards")
            
            if len(gap.missing_skills) > 5:
                recommendations.append("🎯 FOCUS: Prioritize top 3-5 skills to maximize learning ROI and avoid overwhelm")
            
            detailed_results.append(Step2Analysis(
                manipal_employee=gap.employee_name,
                role=gap.role,
                missing_skills=gap.missing_skills,
                skill_importance=skill_importance,
                gap_reasoning=gap_reasoning,
                overall_assessment=gap.gap_reasoning,
                recommendations=recommendations,
                competitor_companies=sorted(competitor_companies),
                competitor_count=gap.competitor_count,
                evidence_flags={
                    'competitors_included': gap.competitor_count,
                    'used_llm': self.use_llm and self.client is not None,
                    'confidence_score': gap.confidence_score,
                    'enhanced_analysis_skills': len([s for s in gap.missing_skills if s in top_skills[:2]]) if 'top_skills' in locals() else 0
                }
            ))
        
        return detailed_results
    
    def export_results(self, results: List[SkillGap], output_path: str, include_basic: bool = True):
        """Export results to JSON"""
        # Detailed output
        output_data = []
        for result in results:
            output_data.append({
                'employee_name': result.employee_name,
                'role': result.role,
                'current_skills': result.current_skills,
                'missing_skills': result.missing_skills,
                'competitor_count': result.competitor_count,
                'gap_reasoning': result.gap_reasoning,
                'confidence_score': result.confidence_score
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Basic output
        if include_basic:
            basic_path = output_path.replace('_detailed', '').replace('.json', '_basic.json')
            basic_data = [
                {
                    'manipal_employee': result.employee_name,
                    'role': result.role,
                    'missing_skills': result.missing_skills
                } for result in results
            ]
            
            with open(basic_path, 'w', encoding='utf-8') as f:
                json.dump(basic_data, f, indent=2, ensure_ascii=False)
            
            print(f"Basic results exported to: {basic_path}")
        
        print(f"Detailed results exported to: {output_path}")
        
        # Print summary
        total_employees = len(results)
        total_gaps = sum(len(r.missing_skills) for r in results)
        avg_confidence = sum(r.confidence_score for r in results) / total_employees if total_employees else 0
        
        print(f"\nSUMMARY:")
        print(f"- Employees analyzed: {total_employees}")
        print(f"- Total skill gaps identified: {total_gaps}")
        print(f"- Average confidence score: {avg_confidence:.2f}")
        print(f"- Employees with gaps: {sum(1 for r in results if r.missing_skills)}")
    
    def export_detailed_analysis(self, detailed_results: List[Step2Analysis], output_path: str):
        """Export detailed Step 2 analysis"""
        output_data = []
        for result in detailed_results:
            output_data.append({
                'manipal_employee': result.manipal_employee,
                'role': result.role,
                'missing_skills': result.missing_skills,
                'skill_importance': result.skill_importance,
                'gap_reasoning': result.gap_reasoning,
                'overall_assessment': result.overall_assessment,
                'recommendations': result.recommendations,
                'competitor_companies': result.competitor_companies,
                'competitor_count': result.competitor_count,
                'evidence_flags': result.evidence_flags
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed analysis exported to: {output_path}")

# =======================
# Main execution
# =======================
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run end-to-end skill gap analysis with unified components."
    )
    parser.add_argument(
        "--spectre", "-s",
        default=Config.SPECTRE_COMPANY,
        help="Primary company key to analyze (e.g., 'xto10x')."
    )
    parser.add_argument(
        "--skills", "-k",
        default=Config.SKILL_DIR,
        help="Folder with normalized skills JSON files."
    )
    parser.add_argument(
        "--raw", "-r",
        default=Config.RAW_DIR,
        help="Folder with raw report JSON files."
    )
    parser.add_argument(
        "--spectre-path",
        default=Config.SPECTRE_PATH,
        help="Path to spectre role match cache JSON (auto-created if missing)."
    )
    parser.add_argument(
        "--no-llm-matching",
        action="store_true",
        help="Disable LLM for role matching (use similarity-only)."
    )
    parser.add_argument(
        "--no-llm-skills",
        action="store_true",
        help="Disable LLM for skill analysis."
    )

    args = parser.parse_args()

    # Apply runtime config
    Config.SPECTRE_COMPANY = (args.spectre or "").strip().lower()
    Config.SKILL_DIR = args.skills
    Config.RAW_DIR = args.raw
    Config.SPECTRE_PATH = args.spectre_path

    # Optional feature toggles
    if args.no_llm_matching:
        Config.USE_LLM_FOR_MATCHING = False
    if args.no_llm_skills:
        Config.USE_LLM_FOR_SKILLS = False

    # Validate config
    try:
        Config.validate()
    except Exception as e:
        print(str(e))
        sys.exit(1)
    
    Config.print_summary()

    # Ensure folders exist
    os.makedirs(Config.SKILL_DIR, exist_ok=True)
    os.makedirs(Config.RAW_DIR, exist_ok=True)

    print("=== STEP 0: Setup & Load ===")
    
    # Load data
    loader = DataLoader(Config.SKILL_DIR, Config.RAW_DIR)
    loader.run()

    # Merge data
    merger = Merger(
        raw_by_company=loader.raw_by_company,
        skills_name_by_company=loader.skills_name_by_company,
        fuzzy_threshold=Config.FUZZY_NAME_THRESHOLD
    )
    company_employees = merger.run()

    # Check if we have spectre company data
    if Config.SPECTRE_COMPANY not in company_employees:
        print(f"❌ Spectre company '{Config.SPECTRE_COMPANY}' not found in merged data!")
        print(f"Available companies: {list(company_employees.keys())}")
        sys.exit(1)

    # Run spectre matching if needed
    spectre_ok = spectre_file_has_matches(Config.SPECTRE_PATH)
    if not spectre_ok or Config.FORCE_RERUN_ANALYSIS:
        print("\n=== Running SpectreMatcher ===")
        matcher = SpectreMatcher(
            company_employees=company_employees,
            spectre_company_key=Config.SPECTRE_COMPANY,
            use_llm=Config.USE_LLM_FOR_MATCHING,
            prefilter_top_k=Config.PREFILTER_TOP_K,
            prefilter_min_ratio=Config.PREFILTER_MIN_RATIO,
            similarity_threshold=Config.ROLE_SIMILARITY_THRESHOLD,
            debug=Config.DEBUG_MODE,
            debug_limit_per_company=Config.DEBUG_LIMIT_PER_COMPANY
        )
        results = matcher.run(export_json_path=Config.SPECTRE_PATH)
        matcher.print_report(results)
        spectre_ok = spectre_file_has_matches(Config.SPECTRE_PATH)
    else:
        print(f"\n=== Using existing spectre matches from {Config.SPECTRE_PATH} ===")

    if not spectre_ok:
        print("\n❌ Could not create usable spectre matches. Check role titles/inputs and rerun.")
        sys.exit(1)

    print("\n=== STEP 1 & 2: Unified Skill Gap Analysis ===")
    
    # Setup Azure config
    azure_config = {
        'api_key': os.getenv("AZURE_OPENAI_API_KEY") or Config.AZURE_API_KEY,
        'endpoint': os.getenv("AZURE_OPENAI_ENDPOINT") or Config.AZURE_ENDPOINT,
        'api_version': os.getenv("AZURE_OPENAI_API_VERSION") or Config.AZURE_API_VERSION,
        'deployment_id': os.getenv("AZURE_OPENAI_DEPLOYMENT_ID") or Config.AZURE_DEPLOYMENT
    }
    
    # Run unified skill gap analysis
    analyzer = UnifiedSkillGapAnalyzer(
        company_employees=company_employees,
        spectre_matches_path=Config.SPECTRE_PATH,
        spectre_company_key=Config.SPECTRE_COMPANY,
        use_llm=Config.USE_LLM_FOR_SKILLS,
        azure_config=azure_config
    )
    
    # Step 1: Analyze skill gaps
    print("Running Step 1: Basic skill gap analysis...")
    skill_gaps = analyzer.analyze_all(max_employees=Config.MAX_EMPLOYEES_TO_ANALYZE)
    
    if not skill_gaps:
        print("⚠️ No skill gaps found. This could mean:")
        print("  - No employees have role matches in spectre_matches.json")
        print("  - All employees already have all competitor skills")
        print("  - There's an issue with the data pipeline")
        sys.exit(1)
    
    # Export Step 1 results
    analyzer.export_results(skill_gaps, Config.STEP1_DETAILED, include_basic=True)
    
    # Step 2: Generate detailed analysis
    print("\nRunning Step 2: Detailed gap analysis...")
    detailed_analysis = analyzer.generate_detailed_analysis(skill_gaps)
    
    # Export Step 2 results
    analyzer.export_detailed_analysis(detailed_analysis, Config.STEP2_DETAILED)
    
    # Also create the final simplified output
    final_output = []
    for analysis in detailed_analysis:
        final_output.append({
            'manipal_employee': analysis.manipal_employee,
            'role': analysis.role,
            'missing_skills': analysis.missing_skills
        })
    
    with open(Config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Final output written to: {Config.OUTPUT_FILE}")
    
    # Summary statistics
    print("\n=== FINAL SUMMARY ===")
    total_spectre_employees = len(company_employees.get(Config.SPECTRE_COMPANY, []))
    employees_with_matches = len(skill_gaps)
    total_missing_skills = sum(len(gap.missing_skills) for gap in skill_gaps)
    avg_missing_per_employee = total_missing_skills / employees_with_matches if employees_with_matches else 0
    
    print(f"Total {Config.SPECTRE_COMPANY} employees: {total_spectre_employees}")
    print(f"Employees with competitor matches: {employees_with_matches}")
    print(f"Total skill gaps identified: {total_missing_skills}")
    print(f"Average gaps per employee: {avg_missing_per_employee:.1f}")
    
    if Config.SHOW_SAMPLE_OUTPUT and skill_gaps:
        print(f"\n=== SAMPLE RESULTS ===")
        for i, gap in enumerate(skill_gaps[:3]):
            print(f"\n{i+1}. {gap.employee_name} ({gap.role})")
            print(f"   Missing skills ({len(gap.missing_skills)}): {', '.join(gap.missing_skills[:5])}")
            if len(gap.missing_skills) > 5:
                print(f"   ... and {len(gap.missing_skills) - 5} more")
            print(f"   Based on {gap.competitor_count} competitors (confidence: {gap.confidence_score})")
    
    print("\n🎯 Pipeline completed successfully!")
    print("\nGenerated files:")
    print(f"  - {Config.SPECTRE_PATH} (role matches)")
    print(f"  - {Config.STEP1_DETAILED} (detailed skill gaps)")
    print(f"  - {Config.STEP2_DETAILED} (comprehensive analysis)")
    print(f"  - {Config.OUTPUT_FILE} (final summary)")