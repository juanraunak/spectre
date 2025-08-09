import json
import os
import re
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DepartmentNormalizer:
    """Normalizes departments and creates individual employee JSONs"""
    
    def __init__(self):
        self.employees_by_company = defaultdict(list)
        self.department_mapping = {}
        
        # Enhanced department classification keywords
        self.dept_keywords = {
            'finance': ['finance', 'financial', 'fintech', 'banking', 'credit', 'loan', 'investment', 'treasury', 'accounting', 'cfo', 'controller'],
            'sales': ['sales', 'business development', 'account', 'revenue', 'channel', 'partner', 'bd', 'commercial'],
            'marketing': ['marketing', 'brand', 'content', 'digital', 'seo', 'social', 'campaign', 'communications', 'pr'],
            'engineering': ['engineer', 'developer', 'tech', 'software', 'backend', 'frontend', 'full stack', 'devops', 'sde', 'architect'],
            'product': ['product', 'pm', 'product manager', 'roadmap', 'feature', 'user experience', 'ux', 'ui'],
            'operations': ['operations', 'ops', 'logistics', 'supply chain', 'process', 'workflow', 'delivery'],
            'hr': ['hr', 'human resources', 'people', 'talent', 'recruiting', 'recruitment', 'chro'],
            'data': ['data', 'analytics', 'scientist', 'analyst', 'bi', 'intelligence', 'ml', 'ai', 'machine learning'],
            'risk': ['risk', 'compliance', 'audit', 'governance', 'regulatory', 'legal', 'counsel'],
            'technology': ['technology', 'it', 'infrastructure', 'security', 'cyber', 'systems', 'cto'],
            'management': ['ceo', 'president', 'founder', 'co-founder', 'managing director', 'executive'],
            'customer_success': ['customer success', 'customer service', 'support', 'client success', 'relationship'],
            'strategy': ['strategy', 'strategic', 'planning', 'consultant', 'business analyst']
        }

    def load_raw_employee_data(self, data_directory: str):
        """Load raw employee data from JSON files"""
        logger.info("📂 Loading raw employee data...")
        
        if not os.path.exists(data_directory):
            logger.error(f"Directory {data_directory} does not exist")
            return
        
        json_files = [f for f in os.listdir(data_directory) if f.endswith('.json')]
        total_employees = 0
        
        for json_file in json_files:
            try:
                file_path = os.path.join(data_directory, json_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract company name from filename
                company_name = self._extract_company_name(json_file)
                
                # Process employee data based on structure
                employees_data = []
                if isinstance(data, dict):
                    if "employee_intelligence" in data and "employees" in data["employee_intelligence"]:
                        employees_data = data["employee_intelligence"]["employees"]
                    elif "employees" in data:
                        employees_data = data["employees"]
                    elif "data" in data:
                        employees_data = data["data"]
                elif isinstance(data, list):
                    employees_data = data
                
                company_employees = []
                for idx, emp_data in enumerate(employees_data):
                    if not isinstance(emp_data, dict):
                        continue
                    
                    total_employees += 1
                    employee_id = f"{company_name.replace(' ', '')[:3].upper()}_{total_employees:04d}"
                    
                    # Extract and normalize employee info
                    normalized_emp = self._normalize_employee(emp_data, company_name, employee_id)
                    if normalized_emp:
                        company_employees.append(normalized_emp)
                
                self.employees_by_company[company_name] = company_employees
                logger.info(f"✅ Loaded {len(company_employees)} employees from {company_name}")
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to process {json_file}: {str(e)}")
        
        logger.info(f"📊 Total companies: {len(self.employees_by_company)}")
        logger.info(f"📊 Total employees processed: {total_employees}")

    def _extract_company_name(self, filename: str) -> str:
        """Extract clean company name from filename"""
        # Remove file extension
        name = filename.replace('.json', '')
        
        # Remove common suffixes
        name = re.sub(r'(_report|_intelligence|_complete|_fintech|_data|_employees)', '', name, flags=re.IGNORECASE)
        
        # Replace underscores with spaces and title case
        name = name.replace('_', ' ').title()
        
        return name

    def _normalize_employee(self, emp_data: Dict, company: str, employee_id: str) -> Optional[Dict]:
        """Normalize individual employee data"""
        try:
            # Extract name
            name = (
                emp_data.get('name') or 
                emp_data.get('full_name') or 
                f"{emp_data.get('first_name', '')} {emp_data.get('last_name', '')}" or
                f"Employee {employee_id}"
            ).strip()
            
            # Extract position/title
            position = (
                emp_data.get('current_position') or 
                emp_data.get('position') or 
                emp_data.get('title') or 
                emp_data.get('job_title') or 
                emp_data.get('role') or
                "Staff"
            )
            
            # Classify department
            department = self._classify_department(position)
            
            return {
                "employee_id": employee_id,
                "name": name,
                "position": position,
                "department": department,
                "company": company
            }
            
        except Exception as e:
            logger.warning(f"Failed to normalize employee {employee_id}: {str(e)}")
            return None

    def _classify_department(self, position: str) -> str:
        """Classify department based on position"""
        if not position:
            return "general"
        
        position_lower = position.lower()
        dept_scores = defaultdict(int)
        
        # Score each department based on keyword matches
        for dept, keywords in self.dept_keywords.items():
            for keyword in keywords:
                if keyword in position_lower:
                    # Give higher weight to exact matches
                    if keyword == position_lower:
                        dept_scores[dept] += 3
                    else:
                        dept_scores[dept] += 1
        
        if dept_scores:
            return max(dept_scores, key=dept_scores.get)
        
        return "general"

    def create_merged_company_files(self, output_directory: str = "normalized_departments", files_per_company: int = 10):
        """Create exactly 10 merged JSON files per company with distributed employees"""
        logger.info(f"📝 Creating {files_per_company} merged JSONs per company in {output_directory}")
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        total_files_created = 0
        
        for company_name, employees in self.employees_by_company.items():
            if not employees:
                continue
            
            # Create company directory
            company_dir = os.path.join(output_directory, company_name.replace(' ', '_').lower())
            if not os.path.exists(company_dir):
                os.makedirs(company_dir)
            
            # Distribute employees across 10 files
            employees_per_file = max(1, len(employees) // files_per_company)
            remainder = len(employees) % files_per_company
            
            start_idx = 0
            files_for_company = 0
            
            for file_num in range(files_per_company):
                # Calculate how many employees for this file
                current_file_size = employees_per_file
                if file_num < remainder:  # Distribute remainder across first few files
                    current_file_size += 1
                
                # Get employees for this file
                end_idx = start_idx + current_file_size
                file_employees = employees[start_idx:end_idx]
                
                if not file_employees:  # Skip empty files
                    continue
                
                # Group by department for this file
                dept_distribution = defaultdict(list)
                for emp in file_employees:
                    dept_distribution[emp["department"]].append({
                        "employee_id": emp["employee_id"],
                        "name": emp["name"],
                        "position": emp["position"],
                        "department": emp["department"]
                    })
                
                # Create filename
                filename = f"{company_name.replace(' ', '_').lower()}_batch_{file_num + 1:02d}.json"
                filepath = os.path.join(company_dir, filename)
                
                # Create merged JSON structure
                merged_json = {
                    "company_name": company_name,
                    "batch_number": file_num + 1,
                    "total_employees": len(file_employees),
                    "departments_included": list(dept_distribution.keys()),
                    "department_counts": {dept: len(emps) for dept, emps in dept_distribution.items()},
                    "employees": [emp for emp_list in dept_distribution.values() for emp in emp_list]
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(merged_json, f, indent=2, ensure_ascii=False)
                
                files_for_company += 1
                total_files_created += 1
                
                dept_summary = ", ".join([f"{dept}({len(emps)})" for dept, emps in dept_distribution.items()])
                logger.info(f"📄 Created {filename} with {len(file_employees)} employees: {dept_summary}")
                
                start_idx = end_idx
            
            # Create a company summary file
            summary_file = os.path.join(company_dir, "company_summary.json")
            
            # Overall department distribution
            all_dept_dist = defaultdict(int)
            for emp in employees:
                all_dept_dist[emp["department"]] += 1
            
            company_summary = {
                "company_name": company_name,
                "total_employees": len(employees),
                "total_files_created": files_for_company,
                "overall_department_distribution": dict(all_dept_dist),
                "files_created": [f"{company_name.replace(' ', '_').lower()}_batch_{i+1:02d}.json" 
                                for i in range(files_for_company)]
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(company_summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Created {files_for_company} batch files + 1 summary for {company_name}")
        
        logger.info(f"🏁 Total files created: {total_files_created}")
        return total_files_created

    def generate_overall_summary(self, output_directory: str = "normalized_departments"):
        """Generate overall summary of normalization"""
        logger.info("📊 Generating overall summary...")
        
        summary_data = {
            "normalization_summary": {
                "total_companies": len(self.employees_by_company),
                "total_employees": sum(len(emps) for emps in self.employees_by_company.values()),
                "companies": []
            }
        }
        
        for company_name, employees in self.employees_by_company.items():
            dept_distribution = defaultdict(int)
            for emp in employees:
                dept_distribution[emp["department"]] += 1
            
            company_info = {
                "company_name": company_name,
                "employee_count": len(employees),
                "departments": dict(dept_distribution)
            }
            summary_data["normalization_summary"]["companies"].append(company_info)
        
        # Save overall summary
        summary_file = os.path.join(output_directory, "normalization_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Overall summary saved to {summary_file}")

# Main execution function
def main():
    """Main function to run the department normalization"""
    normalizer = DepartmentNormalizer()
    
    # Load raw employee data (adjust path as needed)
    data_directory = "employee_data"  # Change this to your data directory
    normalizer.load_raw_employee_data(data_directory)
    
    # Create department-wise employee JSONs
    output_dir = "normalized_departments"
    files_created = normalizer.create_merged_company_files(output_dir, files_per_company=10)
    
    # Generate overall summary
    normalizer.generate_overall_summary(output_dir)
    
    print(f"\n🎉 Department normalization completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Total JSON files created: {files_created}")
    print(f"🏢 Companies processed: {len(normalizer.employees_by_company)}")

if __name__ == "__main__":
    main()