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
    """Basic employee profile structure"""
    name: str
    current_position: str
    company_name: str
    employee_id: str
    skills: List[str] = None
    
@dataclass
class CompanySkillProfile:
    """Company-wise skill profile"""
    company_name: str
    employees: List[EmployeeProfile]

class FractalSkillExtractor:
    """Extracts skills company by company"""
    
    def __init__(self, azure_config: Dict[str, str] = None, max_workers: int = 5):
        if azure_config:
            self.openai_client = AzureOpenAI(
                api_key=azure_config.get("api_key"),
                api_version=azure_config.get("api_version"),
                azure_endpoint=azure_config.get("endpoint")
            )
            self.deployment_id = azure_config.get('deployment_id')
        else:
            self.openai_client = None
            self.deployment_id = None
        
        self.max_workers = max_workers
        self.company_profiles = {}
    
    async def process_company_files(self, data_directory: str):
        """Process each company file separately"""
        logger.info("📂 Processing company files...")
        data_path = Path(data_directory)
        json_files = list(data_path.glob("*.json"))
        
        for json_file in json_files:
            company_name = self._extract_company_name(json_file)
            logger.info(f"🏢 Processing {company_name} from {json_file.name}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                employees = []
                if isinstance(data, dict):
                    # Handle different JSON structures
                    if "employee_intelligence" in data and "employees" in data["employee_intelligence"]:
                        employees_data = data["employee_intelligence"]["employees"]
                    elif "employees" in data:
                        employees_data = data["employees"]
                    else:
                        employees_data = []
                    
                    # Process each employee in the file
                    for i, emp_data in enumerate(employees_data):
                        employee = self._extract_basic_profile(emp_data, f"{company_name[:3]}_{i+1:03d}")
                        if employee:
                            employees.append(employee)
                
                # Extract skills for all employees in this company
                await self.extract_skills_for_company(company_name, employees)
                
                # Save company profile immediately
                self.save_company_profile(company_name, "output/company_skills")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {json_file.name}: {str(e)}")
    
    def _extract_company_name(self, json_file: Path) -> str:
        """Extract company name from filename"""
        filename = json_file.stem
        clean_name = re.sub(r'(_report|_intelligence|_complete|_fintech)', '', filename, flags=re.IGNORECASE)
        clean_name = clean_name.replace('_', ' ').title()
        return clean_name
    
    def _extract_basic_profile(self, emp_data: Dict, employee_id: str) -> Optional[EmployeeProfile]:
        """Extract basic employee info"""
        try:
            # Handle different profile structures
            profile_data = emp_data.get("detailed_profile", emp_data)
            
            name = (profile_data.get('name') or 
                   profile_data.get('full_name') or 
                   f"{profile_data.get('first_name', '')} {profile_data.get('last_name', '')}" or
                   f"Employee_{employee_id}").strip()
            
            position = (profile_data.get('current_position') or 
                       profile_data.get('position') or 
                       profile_data.get('title') or 
                       "")
            
            return EmployeeProfile(
                name=name,
                current_position=position,
                company_name=employee_id.split('_')[0],
                employee_id=employee_id
            )
        except Exception as e:
            logger.warning(f"Failed to extract basic profile: {str(e)}")
            return None
    
    async def extract_skills_for_company(self, company_name: str, employees: List[EmployeeProfile]):
        """Extract skills for all employees in a company"""
        logger.info(f"🧠 Extracting skills for {company_name} ({len(employees)} employees)")
        
        if not self.openai_client:
            logger.warning("⚠️ No GPT client available, using keyword fallback")
            self._extract_skills_keyword_fallback(employees)
            return
        
        # Create company profile
        company_profile = CompanySkillProfile(company_name=company_name, employees=[])
        
        batch_size = 10
        total_batches = (len(employees) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(employees))
            batch_employees = employees[start_idx:end_idx]
            
            logger.info(f"🔄 Processing batch {batch_num + 1}/{total_batches} for {company_name}")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                tasks = []
                for employee in batch_employees:
                    task = executor.submit(self._extract_skills_gpt, employee)
                    tasks.append(task)
                
                for i, task in enumerate(tasks):
                    try:
                        employee_with_skills = task.result()
                        if employee_with_skills:
                            company_profile.employees.append(employee_with_skills)
                    except Exception as e:
                        logger.warning(f"Skill extraction failed: {str(e)}")
                        employee = batch_employees[i]
                        fallback_employee = self._create_fallback_skill_profile(employee)
                        if fallback_employee:
                            company_profile.employees.append(fallback_employee)
            
            if batch_num < total_batches - 1:
                await asyncio.sleep(1)
        
        self.company_profiles[company_name] = company_profile
        logger.info(f"✅ Completed skill extraction for {company_name}")
    
    def _extract_skills_gpt(self, employee: EmployeeProfile) -> Optional[EmployeeProfile]:
        """Extract skills using GPT for individual employee"""
        try:
            # In a real implementation, you'd call GPT here
            # For this example, we'll simulate skill extraction
            simulated_skills = [
                "Financial Analysis", "Risk Management", "Python", 
                "Data Visualization", "Regulatory Compliance"
            ]
            
            employee.skills = simulated_skills
            return employee
            
        except Exception as e:
            logger.warning(f"GPT skill extraction failed for {employee.name}: {str(e)}")
            return self._create_fallback_skill_profile(employee)
    
    def _create_fallback_skill_profile(self, employee: EmployeeProfile) -> EmployeeProfile:
        """Create fallback skill profile using keywords"""
        # Simple keyword-based skill assignment
        position = employee.current_position.lower()
        
        if "data" in position:
            skills = ["SQL", "Python", "Data Analysis", "Machine Learning"]
        elif "risk" in position:
            skills = ["Risk Assessment", "Compliance", "Regulatory Frameworks"]
        elif "manager" in position:
            skills = ["Leadership", "Project Management", "Strategic Planning"]
        else:
            skills = ["Financial Analysis", "Communication", "Problem Solving"]
        
        employee.skills = skills
        return employee
    
    def save_company_profile(self, company_name: str, output_dir: str):
        """Save company skill profile to file"""
        Path(output_dir).mkdir(exist_ok=True)
        profile = self.company_profiles.get(company_name)
        
        if profile:
            filename = f"{output_dir}/{company_name.replace(' ', '_')}_skills.json"
            with open(filename, 'w') as f:
                data = {
                    "company_name": company_name,
                    "employees": [
                        {
                            "name": emp.name,
                            "position": emp.current_position,
                            "employee_id": emp.employee_id,
                            "skills": emp.skills
                        }
                        for emp in profile.employees
                    ]
                }
                json.dump(data, f, indent=2)
            logger.info(f"💾 Saved {company_name} skills to {filename}")

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
    
    # Step 1: Extract skills company by company
    extractor = FractalSkillExtractor(azure_config=azure_config)
    asyncio.run(extractor.process_company_files("employee_data"))
    
    # Step 2: After all company files are processed, we'll do normalization and gap analysis
    # (This would be implemented separately after all company skills are extracted)
    logger.info("🏁 Completed company-wise skill extraction. Proceed to normalization and gap analysis.")