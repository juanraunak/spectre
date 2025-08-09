"""
FRACTAL ghost_3 - Complete Enhanced Version for ALL Employee Analysis
Processes all employees from Spectre company and provides comprehensive skill gap analysis
FIXED: Properly handles the JSON structure with employee_intelligence > employees
"""

import json
import os
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import logging
from pathlib import Path
import asyncio
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EmployeeProfile:
    """Normalized employee profile structure"""
    name: str
    about: str
    experience_titles: List[str]
    experience_descriptions: List[str]
    education_degrees: List[str]
    education_fields: List[str]
    current_position: str
    experience_companies: List[str]
    search_snippet: str
    activity_titles: List[str]
    company_name: str
    is_target: bool
    employee_id: str
    
@dataclass
class SkillProfile:
    """Employee skills profile"""
    employee_id: str
    name: str
    role: str
    department: str
    company: str
    is_target: bool
    core_skills: List[str]
    tools: List[str]
    soft_skills: List[str]
    certifications: List[str]
    skill_confidence: Dict[str, float]
    total_skills_count: int

@dataclass
class TalentGap:
    """Enhanced talent gap analysis result"""
    employee_id: str
    employee: str
    current_role: str
    department: str
    company: str
    current_skills: List[str]
    missing_skills: List[str]
    critical_missing_skills: List[str]
    suggested_upskilling: List[str]
    competitor_benchmarks: List[Dict[str, Any]]
    gap_explanations: List[Dict[str, str]]
    priority_score: float
    skill_gap_severity: str
# ... (rest of the code remains the same) ...

class FractalGhost3Complete:
    """Complete FRACTAL ghost_3 pipeline for processing ALL employees"""
    
    def __init__(self, azure_config: Dict[str, str] = None, max_workers: int = 5):
        if azure_config:
            # CORRECTED: Pass credentials as keyword arguments
            self.openai_client = AzureOpenAI(
                api_key=azure_config.get("api_key"),  # Correct parameter name
                api_version=azure_config.get("api_version"),
                azure_endpoint=azure_config.get("endpoint")
            )
            self.deployment_id = azure_config.get('deployment_id')
        else:
            self.openai_client = None
            self.deployment_id = None
        
        
        self.max_workers = max_workers
        self.employees = []
        self.departments = defaultdict(list)
        self.normalized_roles = defaultdict(lambda: defaultdict(list))
        self.skill_profiles = []
        self.talent_gaps = []
        self.competitor_skill_benchmarks = {}
        
        # Enhanced department classification keywords
        self.dept_keywords = {
            'finance': ['finance', 'financial', 'fintech', 'banking', 'credit', 'loan', 'investment', 'treasury', 'accounting', 'cfo', 'wealth', 'asset', 'capital', 'risk', 'audit'],
            'sales': ['sales', 'business development', 'account', 'revenue', 'channel', 'partner', 'relationship', 'client', 'customer'],
            'marketing': ['marketing', 'brand', 'content', 'digital', 'seo', 'social', 'campaign', 'growth', 'advertising'],
            'engineering': ['engineer', 'developer', 'tech', 'software', 'backend', 'frontend', 'full stack', 'devops', 'architect', 'programming'],
            'product': ['product', 'pm', 'product manager', 'roadmap', 'feature', 'user experience', 'ux'],
            'operations': ['operations', 'ops', 'logistics', 'supply chain', 'process', 'workflow', 'efficiency'],
            'hr': ['hr', 'human resources', 'people', 'talent', 'recruiting', 'recruitment', 'employee'],
            'customer_success': ['customer success', 'support', 'customer experience', 'client', 'service', 'satisfaction'],
            'data': ['data', 'analytics', 'scientist', 'analyst', 'bi', 'intelligence', 'ml', 'ai', 'machine learning'],
            'design': ['design', 'ux', 'ui', 'creative', 'visual', 'graphic', 'user interface'],
            'risk': ['risk', 'compliance', 'audit', 'governance', 'regulatory', 'control'],
            'technology': ['technology', 'it', 'infrastructure', 'security', 'cyber', 'systems'],
            'management': ['manager', 'director', 'head', 'vp', 'vice president', 'chief', 'lead', 'senior', 'executive']
        }

    # STEP 1: LOADING PROFILES
    async def load_all_employee_profiles(self, data_directory: str, target_company: str = None):
        """Load ALL employee profiles - FIXED for proper JSON structure"""
        logger.info("📂 STEP 1: Loading ALL employee profiles...")

        data_path = Path(data_directory)
        json_files = list(data_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {data_directory}")

        # Detect Spectre target company
        spectre_company_name = None
        total_profiles_found = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check for Spectre company
                if isinstance(data, dict):
                    if "Spectre_company" in data:
                        spectre_info = data["Spectre_company"]
                        spectre_company_name = spectre_info.get("company_name", "").strip()
                        logger.info(f"🎯 Found Spectre target company: {spectre_company_name}")
                    
                    # Count profiles in employee_intelligence > employees structure
                    if "employee_intelligence" in data and "employees" in data["employee_intelligence"]:
                        employees = data["employee_intelligence"]["employees"]
                        total_profiles_found += len(employees)
                        logger.info(f"📊 Found {len(employees)} employee profiles in {json_file.name}")
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to read {json_file}: {str(e)}")

        logger.info(f"📊 Total profiles found across all files: {total_profiles_found}")

        final_target_company = target_company or spectre_company_name
        if final_target_company:
            logger.info(f"🎯 Target company: {final_target_company}")

        # Load ALL profiles
        employee_counter = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                profiles = []
                if isinstance(data, dict):
                    # Primary: Look for employee_intelligence > employees structure
                    if "employee_intelligence" in data and "employees" in data["employee_intelligence"]:
                        employees_data = data["employee_intelligence"]["employees"]
                        logger.info(f"📄 Processing {len(employees_data)} profiles from {json_file.name}")
                        
                        for emp_data in employees_data:
                            # Extract the detailed_profile which contains the actual employee data
                            if "detailed_profile" in emp_data:
                                profile_data = emp_data["detailed_profile"]
                                # Also include basic_info if available
                                if "basic_info" in emp_data:
                                    basic_info = emp_data["basic_info"]
                                    profile_data["search_snippet"] = basic_info.get("search_snippet", "")
                                    if "company" not in profile_data and "company" in basic_info:
                                        profile_data["company"] = basic_info["company"]
                                profiles.append(profile_data)
                            else:
                                profiles.append(emp_data)
                        
                for profile_data in profiles:
                    if not isinstance(profile_data, dict):
                        continue

                    employee_counter += 1
                    company_name = self._extract_company_name(profile_data, json_file.stem)
                    is_target = self._is_target_company(company_name, final_target_company)

                    profile = self._normalize_profile(profile_data, company_name, is_target, f"EMP_{employee_counter:04d}")
                    if profile:
                        self.employees.append(profile)

                    if employee_counter % 25 == 0:
                        logger.info(f"✅ Processed {employee_counter} employees so far...")

            except Exception as e:
                logger.warning(f"❌ Failed to process {json_file.name}: {str(e)}")

        target_count = len([emp for emp in self.employees if emp.is_target])
        competitor_count = len([emp for emp in self.employees if not emp.is_target])
        
        logger.info(f"✅ STEP 1 COMPLETE: Loaded {len(self.employees)} total employee profiles")
        logger.info(f"🎯 Target company employees: {target_count}")
        logger.info(f"🏢 Competitor employees: {competitor_count}")

    def _extract_company_name(self, profile_data: Dict, filename: str) -> str:
        """Enhanced company name extraction"""
        # Try various company field names
        for field in ['company_name', 'company', 'organization', 'employer']:
            if field in profile_data and profile_data[field]:
                return str(profile_data[field]).strip()
        
        # Look in summary section
        if 'summary' in profile_data and isinstance(profile_data['summary'], dict):
            summary = profile_data['summary']
            for field in ['company', 'current_company']:
                if field in summary and summary[field]:
                    return str(summary[field]).strip()
        
        # Look in experience
        experience = profile_data.get('experience', [])
        if isinstance(experience, list) and experience:
            for exp in experience[:2]:
                if isinstance(exp, dict) and 'company' in exp and exp['company']:
                    return str(exp['company']).strip()
        
        # Extract from filename as fallback
        clean_name = filename.replace('_', ' ').replace('-', ' ')
        clean_name = re.sub(r'\b(report|intelligence|complete|fintech)\b', '', clean_name, flags=re.IGNORECASE)
        return clean_name.strip().title()

    def _is_target_company(self, company_name: str, target_company: str) -> bool:
        """Enhanced target company matching"""
        if not target_company:
            return False
        
        company_norm = re.sub(r'[^a-zA-Z0-9]', '', company_name.lower())
        target_norm = re.sub(r'[^a-zA-Z0-9]', '', target_company.lower())
        
        return (
            company_norm == target_norm or
            company_norm in target_norm or
            target_norm in company_norm or
            any(word in company_norm for word in target_norm.split() if len(word) > 3)
        )

    # STEP 2: NORMALIZING PROFILES
    def _normalize_profile(self, data: Dict, company_name: str, is_target: bool, employee_id: str) -> Optional[EmployeeProfile]:
        """Enhanced profile normalization"""
        try:
            name = (data.get('name') or 
                   data.get('full_name') or 
                   data.get('first_name', '') + ' ' + data.get('last_name', '') or
                   f'Employee_{employee_id}').strip()
            
            return EmployeeProfile(
                name=name,
                about=self._extract_about(data),
                experience_titles=self._extract_experience_titles(data),
                experience_descriptions=self._extract_experience_descriptions(data),
                education_degrees=self._extract_education_degrees(data),
                education_fields=self._extract_education_fields(data),
                current_position=self._extract_current_position(data),
                experience_companies=self._extract_experience_companies(data),
                search_snippet=data.get('search_snippet', ''),
                activity_titles=self._extract_activity_titles(data),
                company_name=company_name,
                is_target=is_target,
                employee_id=employee_id
            )
        except Exception as e:
            logger.warning(f"Failed to normalize profile for {employee_id}: {str(e)}")
            return None

    def _extract_about(self, data: Dict) -> str:
        for field in ['about', 'summary', 'bio', 'description', 'headline']:
            if field in data:
                value = data[field]
                if isinstance(value, dict):
                    return value.get('about', value.get('text', value.get('summary', '')))
                elif isinstance(value, str):
                    return value
        
        if 'summary' in data and isinstance(data['summary'], dict):
            summary_obj = data['summary']
            for field in ['about', 'description', 'text', 'summary']:
                if field in summary_obj and summary_obj[field]:
                    return str(summary_obj[field])
        return ''

    def _extract_current_position(self, data: Dict) -> str:
        for field in ['current_position', 'position', 'title', 'job_title', 'headline']:
            if field in data and data[field]:
                return str(data[field])
        
        if 'summary' in data and isinstance(data['summary'], dict):
            summary = data['summary']
            for field in ['current_position', 'position', 'title', 'headline']:
                if field in summary and summary[field]:
                    return str(summary[field])
        
        experience = data.get('experience', [])
        if isinstance(experience, list) and experience:
            first_exp = experience[0]
            if isinstance(first_exp, dict):
                for field in ['title', 'position', 'job_title']:
                    if field in first_exp and first_exp[field]:
                        return str(first_exp[field])
        return ''

    def _extract_experience_titles(self, data: Dict) -> List[str]:
        titles = []
        experience = data.get('experience', [])
        if isinstance(experience, list):
            for exp in experience:
                if isinstance(exp, dict):
                    title = exp.get('title') or exp.get('position') or exp.get('job_title')
                    if title:
                        titles.append(str(title))
        return titles

    def _extract_experience_descriptions(self, data: Dict) -> List[str]:
        descriptions = []
        experience = data.get('experience', [])
        if isinstance(experience, list):
            for exp in experience:
                if isinstance(exp, dict):
                    desc = exp.get('description') or exp.get('summary') or exp.get('details')
                    if desc:
                        descriptions.append(str(desc))
        return descriptions

    def _extract_experience_companies(self, data: Dict) -> List[str]:
        companies = []
        experience = data.get('experience', [])
        if isinstance(experience, list):
            for exp in experience:
                if isinstance(exp, dict):
                    company = exp.get('company') or exp.get('organization') or exp.get('employer')
                    if company:
                        companies.append(str(company))
        return companies

    def _extract_education_degrees(self, data: Dict) -> List[str]:
        degrees = []
        education = data.get('education', [])
        if isinstance(education, list):
            for edu in education:
                if isinstance(edu, dict):
                    degree = edu.get('degree') or edu.get('qualification') or edu.get('title')
                    if degree:
                        degrees.append(str(degree))
        return degrees

    def _extract_education_fields(self, data: Dict) -> List[str]:
        fields = []
        education = data.get('education', [])
        if isinstance(education, list):
            for edu in education:
                if isinstance(edu, dict):
                    field = edu.get('field_of_study') or edu.get('field') or edu.get('major') or edu.get('subject')
                    if field:
                        fields.append(str(field))
        return fields

    def _extract_activity_titles(self, data: Dict) -> List[str]:
        titles = []
        activities = data.get('activities', [])
        if isinstance(activities, list):
            for activity in activities:
                if isinstance(activity, dict):
                    title = activity.get('title') or activity.get('name')
                    if title:
                        titles.append(str(title))
        return titles

    # STEP 3: DEPARTMENT AND ROLE CLASSIFICATION
    def classify_departments_and_roles(self):
        """STEP 3: Classify employees into departments and normalize roles"""
        logger.info("🏢 STEP 3: Classifying departments and normalizing roles...")
        
        for employee in self.employees:
            department = self._classify_department(employee)
            role = self._normalize_role(employee)
            
            self.departments[department].append(employee)
            self.normalized_roles[department][role].append(employee)
        
        dept_summary = {dept: len(emps) for dept, emps in self.departments.items()}
        logger.info(f"✅ STEP 3 COMPLETE: Department distribution: {dept_summary}")

    def _classify_department(self, employee: EmployeeProfile) -> str:
        """Enhanced department classification"""
        text_to_analyze = ' '.join([
            employee.current_position,
            ' '.join(employee.experience_titles),
            employee.about,
            employee.company_name
        ]).lower()
        
        dept_scores = defaultdict(int)
        
        for dept, keywords in self.dept_keywords.items():
            for keyword in keywords:
                if keyword in text_to_analyze:
                    dept_scores[dept] += 2
                if any(word in text_to_analyze for word in keyword.split()):
                    dept_scores[dept] += 1
        
        if any(word in employee.company_name.lower() for word in ['fintech', 'bank', 'finance', 'capital']):
            dept_scores['finance'] += 3
        
        if dept_scores:
            return max(dept_scores, key=dept_scores.get)
        else:
            return 'general'

    def _normalize_role(self, employee: EmployeeProfile) -> str:
        """Enhanced role normalization"""
        current_title = employee.current_position.lower()
        
        role_patterns = {
            'CEO/Founder': ['ceo', 'chief executive', 'founder', 'co-founder', 'managing director'],
            'VP/Director': ['vp', 'vice president', 'director', 'head of', 'chief'],
            'Manager': ['manager', 'lead', 'team lead', 'senior manager'],
            'Senior Professional': ['senior', 'principal', 'sr.', 'sr ', 'lead'],
            'Professional': ['analyst', 'associate', 'executive', 'specialist', 'consultant'],
            'Junior Professional': ['junior', 'jr.', 'jr ', 'trainee', 'intern', 'assistant']
        }
        
        for normalized_role, patterns in role_patterns.items():
            for pattern in patterns:
                if pattern in current_title:
                    return normalized_role
        
        return 'Professional'

    # STEP 4: SKILL EXTRACTION USING GPT
    async def extract_skills_using_gpt(self):
        """STEP 4: Extract skills for ALL employees using GPT"""
        logger.info(f"🧠 STEP 4: Extracting skills using GPT for {len(self.employees)} employees...")
        
        if not self.openai_client:
            logger.warning("⚠️ No GPT client available, using keyword fallback")
            self._extract_skills_keyword_fallback()
            return
        
        batch_size = 20
        total_batches = (len(self.employees) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(self.employees))
            batch_employees = self.employees[start_idx:end_idx]
            
            logger.info(f"🔄 Processing batch {batch_num + 1}/{total_batches} ({len(batch_employees)} employees)")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                tasks = []
                for employee in batch_employees:
                    task = executor.submit(self._extract_skills_gpt, employee)
                    tasks.append(task)
                
                for i, task in enumerate(tasks):
                    try:
                        skill_profile = task.result()
                        if skill_profile:
                            self.skill_profiles.append(skill_profile)
                    except Exception as e:
                        logger.warning(f"GPT skill extraction failed for employee {start_idx + i + 1}: {str(e)}")
                        employee = batch_employees[i]
                        fallback_profile = self._create_fallback_skill_profile(employee)
                        if fallback_profile:
                            self.skill_profiles.append(fallback_profile)
            
            if batch_num < total_batches - 1:
                await asyncio.sleep(1)
        
        logger.info(f"✅ STEP 4 COMPLETE: Extracted skills for {len(self.skill_profiles)} employees")

    def _extract_skills_gpt(self, employee: EmployeeProfile) -> Optional[SkillProfile]:
        """Extract skills using GPT for individual employee"""
        try:
            prompt = self._build_skill_extraction_prompt(employee)
            
            response = self.openai_client.chat.completions.create(
                model=self.deployment_id,
                messages=[
                    {"role": "system", "content": "You are an expert HR analyst specializing in fintech talent analysis. Extract comprehensive skills from professional profiles and return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean JSON response
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            skills_data = json.loads(content)
            
            department = self._classify_department(employee)
            role = self._normalize_role(employee)
            
            core_skills = skills_data.get('core_skills', [])[:15]
            tools = skills_data.get('tools', [])[:10]
            soft_skills = skills_data.get('soft_skills', [])[:8]
            
            return SkillProfile(
                employee_id=employee.employee_id,
                name=employee.name,
                role=role,
                department=department,
                company=employee.company_name,
                is_target=employee.is_target,
                core_skills=core_skills,
                tools=tools,
                soft_skills=soft_skills,
                certifications=skills_data.get('certifications', [])[:5],
                skill_confidence=skills_data.get('confidence_scores', {}),
                total_skills_count=len(core_skills) + len(tools) + len(soft_skills)
            )
            
        except Exception as e:
            logger.warning(f"GPT skill extraction failed for {employee.name}: {str(e)}")
            return self._create_fallback_skill_profile(employee)

    def _build_skill_extraction_prompt(self, employee: EmployeeProfile) -> str:
        """Build comprehensive prompt for GPT skill extraction"""
        return f"""
Extract comprehensive professional skills from this fintech employee profile:

Employee: {employee.name}
Position: {employee.current_position}
Company: {employee.company_name}
About: {employee.about[:500]}
Experience: {' '.join(employee.experience_descriptions)[:600]}
Experience Titles: {', '.join(employee.experience_titles[:5])}
Education: {', '.join(employee.education_degrees + employee.education_fields)}

Return JSON in this exact format:
{{
    "core_skills": ["skill1", "skill2", "skill3"],
    "tools": ["tool1", "tool2", "tool3"],
    "soft_skills": ["skill1", "skill2"],
    "certifications": ["cert1", "cert2"],
    "confidence_scores": {{"skill_name": 0.8}}
}}

Focus on:
- Technical skills (programming, frameworks, methodologies)
- Business skills (strategy, analysis, management)
- Fintech-specific skills (payments, lending, risk, compliance)
- Tools and platforms
- Soft skills and leadership
- Professional certifications

Return ONLY the JSON, no other text.
"""

    def _create_fallback_skill_profile(self, employee: EmployeeProfile) -> SkillProfile:
        """Create fallback skill profile using keywords"""
        department = self._classify_department(employee)
        role = self._normalize_role(employee)
        
        all_text = ' '.join([
            employee.about,
            ' '.join(employee.experience_descriptions),
            employee.current_position,
            ' '.join(employee.experience_titles)
        ]).lower()
        
        skill_categories = {
            'technical': ['python', 'java', 'javascript', 'sql', 'aws', 'azure', 'api', 'machine learning', 'data science', 'ai', 'blockchain'],
            'business': ['strategy', 'planning', 'analysis', 'management', 'leadership', 'operations', 'project management'],
            'fintech': ['fintech', 'banking', 'payments', 'lending', 'credit', 'risk management', 'compliance', 'financial modeling'],
            'tools': ['salesforce', 'tableau', 'excel', 'powerpoint', 'jira', 'confluence'],
            'soft_skills': ['communication', 'teamwork', 'leadership', 'problem solving', 'creativity']
        }
        
        extracted_skills = {category: [] for category in skill_categories}
        
        for category, keywords in skill_categories.items():
            for keyword in keywords:
                if keyword in all_text:
                    extracted_skills[category].append(keyword)
        
        core_skills = extracted_skills['technical'] + extracted_skills['business'] + extracted_skills['fintech']
        
        return SkillProfile(
            employee_id=employee.employee_id,
            name=employee.name,
            role=role,
            department=department,
            company=employee.company_name,
            is_target=employee.is_target,
            core_skills=list(set(core_skills))[:15],
            tools=extracted_skills['tools'][:10],
            soft_skills=extracted_skills['soft_skills'][:8],
            certifications=[],
            skill_confidence={},
            total_skills_count=len(set(core_skills)) + len(extracted_skills['tools']) + len(extracted_skills['soft_skills'])
        )

    def _extract_skills_keyword_fallback(self):
        """Fallback keyword-based skill extraction"""
        logger.info("🔍 Using keyword-based skill extraction...")
        
        for employee in self.employees:
            skill_profile = self._create_fallback_skill_profile(employee)
            self.skill_profiles.append(skill_profile)

    # STEP 5: COMPETITOR BENCHMARKING
    async def build_competitor_benchmarks(self):
        """STEP 5: Build competitor skill benchmarks"""
        logger.info("🔍 STEP 5: Building competitor skill benchmarks...")
        
        competitor_profiles = [p for p in self.skill_profiles if not p.is_target]
        
        competitor_groups = defaultdict(lambda: defaultdict(list))
        for profile in competitor_profiles:
            competitor_groups[profile.department][profile.role].append(profile)
        
        for dept, roles in competitor_groups.items():
            if dept not in self.competitor_skill_benchmarks:
                self.competitor_skill_benchmarks[dept] = {}
                
            for role, profiles in roles.items():
                if len(profiles) >= 1:
                    all_skills = []
                    for profile in profiles:
                        all_skills.extend(profile.core_skills + profile.tools + profile.soft_skills)
                    
                    skill_frequency = Counter(all_skills)
                    total_competitors = len(profiles)
                    
                    skill_benchmarks = {}
                    for skill, count in skill_frequency.items():
                        prevalence = count / total_competitors
                        importance = "Critical" if prevalence >= 0.7 else "Important" if prevalence >= 0.4 else "Useful"
                        
                        skill_benchmarks[skill] = {
                            "prevalence": prevalence,
                            "count": count,
                            "total_competitors": total_competitors,
                            "importance": importance,
                            "competitor_companies": list(set(p.company for p in profiles))
                        }
                    
                    self.competitor_skill_benchmarks[dept][role] = {
                        "total_competitors": total_competitors,
                        "skill_benchmarks": skill_benchmarks,
                        "companies": list(set(p.company for p in profiles))
                    }
        
        logger.info(f"✅ STEP 5 COMPLETE: Built benchmarks for {len(self.competitor_skill_benchmarks)} departments")


    # STEP 6: TALENT GAP ANALYSIS
    async def analyze_talent_gaps(self):
        """STEP 6: Comprehensive talent gap analysis"""
        logger.info("📊 STEP 6: Performing talent gap analysis...")
        
        target_profiles = [p for p in self.skill_profiles if p.is_target]
        competitor_profiles = [p for p in self.skill_profiles if not p.is_target]
        
        logger.info(f"🎯 Analyzing {len(target_profiles)} target employees against {len(competitor_profiles)} competitors")
        
        for target_profile in target_profiles:
            similar_competitors = self._find_similar_competitors(target_profile, competitor_profiles)
            
            if not similar_competitors:
                similar_competitors = [p for p in competitor_profiles if p.department == target_profile.department]
            
            if not similar_competitors:
                similar_competitors = competitor_profiles[:10]
            
            gap_analysis = self._analyze_individual_skill_gaps(target_profile, similar_competitors)
            if gap_analysis:
                self.talent_gaps.append(gap_analysis)
        
        # Calculate severity distribution
        severity_counts = Counter(gap.skill_gap_severity for gap in self.talent_gaps)
        logger.info(f"✅ STEP 6 COMPLETE: Completed gap analysis for {len(self.talent_gaps)} employees")
        logger.info(f"🚨 Severity Distribution: {dict(severity_counts)}")
        return self.talent_gaps

    def _find_similar_competitors(self, target: SkillProfile, competitors: List[SkillProfile]) -> List[SkillProfile]:
        """Find similar competitor profiles based on role and skills"""
        # First filter by department and role
        similar = [c for c in competitors 
                  if c.department == target.department 
                  and c.role == target.role]
        
        if len(similar) >= 3:
            return similar
        
        # If not enough, broaden to same department
        similar = [c for c in competitors if c.department == target.department]
        return similar

    def _analyze_individual_skill_gaps(self, target: SkillProfile, competitors: List[SkillProfile]) -> TalentGap:
        """Perform detailed gap analysis for a single employee"""
        # Aggregate competitor skills
        competitor_skills = defaultdict(int)
        for comp in competitors:
            for skill in comp.core_skills + comp.tools + comp.soft_skills:
                competitor_skills[skill] += 1
        
        total_comps = len(competitors)
        
        # Identify missing skills
        target_skills = set(target.core_skills + target.tools + target.soft_skills)
        missing_skills = []
        critical_missing = []
        gap_explanations = []
        
        for skill, count in competitor_skills.items():
            prevalence = count / total_comps
            if skill not in target_skills and prevalence >= 0.25:  # At least 25% of competitors have this
                missing_skills.append(skill)
                if prevalence >= 0.6:  # Critical if >60% of competitors have it
                    critical_missing.append(skill)
                    gap_explanations.append({
                        "skill": skill,
                        "reason": f"Critical gap - {count}/{total_comps} ({prevalence:.0%}) competitors possess this",
                        "severity": "critical"
                    })
                else:
                    gap_explanations.append({
                        "skill": skill,
                        "reason": f"Significant gap - {count}/{total_comps} ({prevalence:.0%}) competitors possess this",
                        "severity": "high"
                    })
        
        # Calculate priority score (0-100)
        base_score = min(100, len(missing_skills) * 5 + len(critical_missing) * 10)
        role_factor = 1.5 if target.role in ["VP/Director", "CEO/Founder"] else 1.0
        priority_score = min(100, base_score * role_factor)
        
        # Determine severity level
        if priority_score >= 75:
            severity = "critical"
        elif priority_score >= 50:
            severity = "high"
        elif priority_score >= 25:
            severity = "medium"
        else:
            severity = "low"
        
        # Suggested upskilling (prioritize critical skills)
        upskilling = critical_missing + [s for s in missing_skills if s not in critical_missing]
        
        return TalentGap(
            employee_id=target.employee_id,
            employee=target.name,
            current_role=target.role,
            department=target.department,
            company=target.company,
            current_skills=list(target_skills),
            missing_skills=missing_skills,
            critical_missing_skills=critical_missing,
            suggested_upskilling=upskilling[:10],  # Top 10 recommendations
            competitor_benchmarks=[{
                "competitor": c.name,
                "company": c.company,
                "skills": c.core_skills + c.tools
            } for c in competitors[:3]],  # Sample 3 competitors
            gap_explanations=gap_explanations,
            priority_score=priority_score,
            skill_gap_severity=severity
        )

    # STEP 7: EXPORT RESULTS
    def export_results(self, output_dir: str = "output"):
        """Export all results to structured files"""
        logger.info(f"💾 STEP 7: Exporting results to {output_dir}")
        Path(output_dir).mkdir(exist_ok=True)
        
        # Export skill profiles
        skills_data = [asdict(p) for p in self.skill_profiles]
        with open(f"{output_dir}/skill_profiles.json", "w") as f:
            json.dump(skills_data, f, indent=2)
        
        # Export talent gaps
        gaps_data = [asdict(g) for g in self.talent_gaps]
        with open(f"{output_dir}/talent_gaps.json", "w") as f:
            json.dump(gaps_data, f, indent=2)
        
        # Export department summaries
        dept_summary = []
        for dept, profiles in self.departments.items():
            dept_gaps = [g for g in self.talent_gaps if g.department == dept]
            critical_count = sum(1 for g in dept_gaps if g.skill_gap_severity == "critical")
            
            dept_summary.append({
                "department": dept,
                "employee_count": len(profiles),
                "critical_gaps_count": critical_count,
                "most_missing_skills": self._get_top_missing_skills(dept)
            })
        
        with open(f"{output_dir}/department_summary.json", "w") as f:
            json.dump(dept_summary, f, indent=2)
        
        logger.info(f"✅ STEP 7 COMPLETE: Exported 3 result files to {output_dir}")
        return {
            "skill_profiles": len(skills_data),
            "talent_gaps": len(gaps_data),
            "departments": len(dept_summary)
        }

    def _get_top_missing_skills(self, department: str) -> List[Dict[str, Any]]:
        """Get top missing skills in a department"""
        dept_gaps = [g for g in self.talent_gaps if g.department == department]
        skill_counter = Counter()
        
        for gap in dept_gaps:
            for skill in gap.missing_skills:
                skill_counter[skill] += 1
        
        return [{"skill": skill, "count": count} 
                for skill, count in skill_counter.most_common(10)]


# Example usage
if __name__ == "__main__":
    # Hardcoded Azure OpenAI configuration
    azure_config = {
        "api_key": "2be1544b3dc14327b60a870fe8b94f35",
        "api_version": "2024-06-01",
        "endpoint": "https://notedai.openai.azure.com",
        "deployment_id": "gpt-4o"
    }
    
    logger.info("✅ Using hardcoded Azure OpenAI configuration")
    
    analyzer = FractalGhost3Complete(azure_config=azure_config)
    
    # Run full pipeline
    async def run_analysis():
        await analyzer.load_all_employee_profiles("employee_data")
        analyzer.classify_departments_and_roles()
        await analyzer.extract_skills_using_gpt()
        await analyzer.build_competitor_benchmarks()
        await analyzer.analyze_talent_gaps()
        analyzer.export_results()
    
    asyncio.run(run_analysis())