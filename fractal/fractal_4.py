import json
import os
import re
from typing import Dict, List, Any, Set, DefaultDict, Tuple
from collections import defaultdict
import logging
import openai
from pathlib import Path
from collections import Counter
from fractal_3 import EmployeeMatcher

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AISkillGapAnalyzer:
    """AI-powered skill gap analysis with detailed explanations"""
    
    def __init__(self, matcher):
        self.matcher = matcher
        self.employee_skills: Dict[Tuple[str, str], List[str]] = {}
        self.final_skill_gap_report: List[Dict] = []
        self.ai_explanations: Dict[Tuple[str, str, str], str] = {}
        
        # Configure OpenAI API
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            logger.warning("OpenAI API key not found. AI explanations will be simulated.")
    
    def debug_data_structure(self):
        """Debug method to inspect loaded data"""
        logger.info("🔍 DEBUG: Inspecting loaded data structures...")
        
        # Check matcher data
        detailed_matches_count = len(self.matcher.detailed_matches) if hasattr(self.matcher, 'detailed_matches') and self.matcher.detailed_matches else 0
        logger.info(f"📊 Matcher detailed_matches count: {detailed_matches_count}")
        raw_employee_data_count = len(self.matcher.raw_employee_data) if hasattr(self.matcher, 'raw_employee_data') and self.matcher.raw_employee_data else 0
        logger.info(f"📊 Raw employee data count: {raw_employee_data_count}")
        logger.info(f"📊 Employee skills loaded: {len(self.employee_skills)}")
        
        # Sample detailed matches structure
        if detailed_matches_count:
            logger.info("🔍 Sample detailed match structure:")
            sample_match = self.matcher.detailed_matches[0]
            logger.info(f"   Target employee keys: {sample_match.get('target_employee', {}).keys()}")
            if sample_match.get('target_employee'):
                target = sample_match['target_employee']
                logger.info(f"   Target employee: {target.get('name', 'N/A')} at {target.get('company', 'N/A')}")
        
        # Sample employee skills
        if self.employee_skills:
            logger.info("🔍 Sample employee skills:")
            for i, ((company, name), skills) in enumerate(list(self.employee_skills.items())[:3]):
                logger.info(f"   Employee {i+1}: {name} at {company} - {len(skills)} skills")
                logger.info(f"   Skills: {skills[:5]}{'...' if len(skills) > 5 else ''}")
    
    def load_skills_data(self, normalized_skills_dir: str):
        """Load normalized skills data from JSON files with new structure"""
        logger.info("📂 Loading normalized skills data...")

        if not os.path.exists(normalized_skills_dir):
            logger.error(f"❌ Directory {normalized_skills_dir} does not exist")
            return

        total_processed = 0
        total_files = 0
        skill_files = [f for f in os.listdir(normalized_skills_dir) 
                      if f.endswith("_skills.json")]

        logger.info(f"📁 Found {len(skill_files)} skill files in '{normalized_skills_dir}'")

        for skill_file in skill_files:
            try:
                file_path = os.path.join(normalized_skills_dir, skill_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    skill_data = json.load(f)

                total_files += 1
                logger.info(f"   📄 Processing {skill_file}...")

                # Extract company name from filename if not in data
                company_name = skill_data.get("company_name") or skill_file.replace("_skills.json", "")
                
                # Get employees list
                employees = skill_data.get('employees', [])
                logger.info(f"   👥 Found {len(employees)} employees in {company_name}")

                for emp_data in employees:
                    employee_name = emp_data.get("name")
                    skills = emp_data.get("skills", [])

                    if not employee_name or not skills:
                        logger.debug(f"⚠️ Skipping invalid skill record in {skill_file}")
                        continue

                    employee_key = (company_name.strip(), employee_name.strip())

                    # Initialize employee skills list if new entry
                    if employee_key not in self.employee_skills:
                        self.employee_skills[employee_key] = []

                    # Process and add skills
                    for single_skill in skills:
                        normalized_skill = self._normalize_skill(single_skill)
                        if normalized_skill and normalized_skill not in self.employee_skills[employee_key]:
                            self.employee_skills[employee_key].append(normalized_skill)
                            total_processed += 1

            except json.JSONDecodeError:
                logger.error(f"❌ Invalid JSON format in {skill_file}")
            except Exception as e:
                logger.error(f"❌ Failed to process {skill_file}: {str(e)}")

        logger.info(f"📊 Total skills processed: {total_processed} from {total_files} files")
        logger.info(f"📊 Total unique employees loaded: {len(self.employee_skills)}")

        # Debug preview
        if self.employee_skills:
            logger.info("🔍 Sample loaded employees:")
            for i, ((company, name), skills) in enumerate(list(self.employee_skills.items())[:5]):
                logger.info(f"   {i+1}. {name} at {company}: {len(skills)} skills")
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize skill name for consistent comparison"""
        if not skill:
            return ""
        
        # Convert to lowercase and remove extra spaces
        normalized = re.sub(r'\s+', ' ', skill.strip()).lower()
        
        # Remove non-alphanumeric characters except hyphens and slashes
        normalized = re.sub(r'[^\w\s\-/]', '', normalized)
        
        # Common skill aliases mapping
        aliases = {
            'js': 'javascript',
            'reactjs': 'react',
            'react.js': 'react',
            'nodejs': 'node.js',
            'py': 'python',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'ai': 'artificial intelligence',
            'pm': 'project management',
            'aws cloud': 'aws',
            'gcp cloud': 'gcp',
            'azure cloud': 'azure',
            'postgres': 'postgresql'
        }
        
        return aliases.get(normalized, normalized)
    
    def perform_ai_skill_gap_analysis(self):
        """Perform AI-powered skill gap analysis for top 100 employees with enhanced debugging"""
        logger.info("🧠 Performing AI-powered skill gap analysis...")
        
        # Debug checks
        if not hasattr(self.matcher, 'detailed_matches') or not self.matcher.detailed_matches:
            logger.error("❌ No detailed matches available. Matcher.detailed_matches is empty or doesn't exist.")
            logger.info("🔍 Available matcher attributes:")
            for attr in dir(self.matcher):
                if not attr.startswith('_'):
                    value = getattr(self.matcher, attr)
                    if hasattr(value, '__len__'):
                        try:
                            logger.info(f"   {attr}: length = {len(value)}")
                        except:
                            logger.info(f"   {attr}: {type(value)}")
                    else:
                        logger.info(f"   {attr}: {type(value)}")
            return
        
        if len(self.matcher.detailed_matches) < 100:
            logger.warning(f"⚠️ Only {len(self.matcher.detailed_matches)} detailed matches available (expected 100+)")
        
        if not self.employee_skills:
            logger.error("❌ No employee skills loaded. Cannot perform gap analysis.")
            return
        
        matches_to_process = min(100, len(self.matcher.detailed_matches))
        logger.info(f"📊 Processing {matches_to_process} employee matches...")
        
        # Process top employees from detailed matches
        for i, match_group in enumerate(self.matcher.detailed_matches[:matches_to_process]):
            if i < 5:  # Debug first 5
                logger.info(f"🔍 Processing employee {i+1}: {json.dumps(match_group, indent=2)[:300]}...")
            
            target_emp = match_group.get('target_employee', {})
            if not target_emp:
                logger.warning(f"⚠️ No target_employee in match group {i+1}")
                continue
                
            target_company = target_emp.get('company', '')
            target_name = target_emp.get('name', '')
            
            if not target_company or not target_name:
                logger.warning(f"⚠️ Missing target employee data: company='{target_company}', name='{target_name}'")
                continue
            
            # Create employee key for target employee
            target_key = (target_company.strip(), target_name.strip())
            
            # Get LinkedIn URL from raw data
            target_name_lower = target_name.lower()
            target_raw = {}
            if hasattr(self.matcher, 'raw_employee_data') and self.matcher.raw_employee_data:
                target_raw = self.matcher.raw_employee_data.get(target_name_lower, {})
            linkedin_target = target_raw.get('linkedin') or target_raw.get('linkedin_profile') or target_raw.get('profile_url')
            
            # Get target skills
            target_skills = set(self.employee_skills.get(target_key, []))
            
            if not target_skills:
                logger.debug(f"   ⚠️ No skills found for target employee: {target_key}")
                # Try alternative key matching
                for (comp, name), skills in self.employee_skills.items():
                    if name.lower() == target_name.lower() or comp.lower() == target_company.lower():
                        logger.debug(f"   🔍 Found potential match: ({comp}, {name}) with {len(skills)} skills")
                        break
                continue
            
            logger.debug(f"   ✅ Target employee {target_name} has {len(target_skills)} skills")
            
            # Store all skill gaps for this employee
            employee_skill_gaps = []
            
            # Get detailed matches for this employee
            detailed_matches = match_group.get('detailed_matches', [])
            if not detailed_matches:
                logger.warning(f"   ⚠️ No detailed matches for employee {target_name}")
                continue
            
            # Analyze each match
            for j, match in enumerate(detailed_matches[:3]):  # Analyze top 3 matches
                matched_emp = match.get('matched_employee', {})
                if not matched_emp:
                    continue
                    
                matched_company = matched_emp.get('company', '')
                matched_name = matched_emp.get('name', '')
                
                if not matched_company or not matched_name:
                    continue
                
                # Create employee key for matched employee
                matched_key = (matched_company.strip(), matched_name.strip())
                
                # Get LinkedIn URL for matched employee
                match_name_lower = matched_name.lower()
                match_raw = {}
                if hasattr(self.matcher, 'raw_employee_data') and self.matcher.raw_employee_data:
                    match_raw = self.matcher.raw_employee_data.get(match_name_lower, {})
                linkedin_match = match_raw.get('linkedin') or match_raw.get('linkedin_profile') or match_raw.get('profile_url')
                
                # Get matched employee skills
                matched_skills = set(self.employee_skills.get(matched_key, []))
                
                if not matched_skills:
                    logger.debug(f"     ⚠️ No skills found for matched employee: {matched_key}")
                    continue
                
                # Calculate skill gaps
                skill_gaps = matched_skills - target_skills
                
                if not skill_gaps:
                    logger.debug(f"     ℹ️ No skill gaps found between {target_name} and {matched_name}")
                    continue
                
                logger.debug(f"     ✅ Found {len(skill_gaps)} skill gaps with {matched_name}")
                
                # Analyze each skill gap with AI
                for skill in skill_gaps:
                    gap_analysis = self._analyze_skill_gap_with_ai(
                        skill, 
                        target_emp,
                        matched_emp
                    )
                    employee_skill_gaps.append({
                        'skill': skill,
                        'matched_employee': matched_emp['name'],
                        'matched_company': matched_emp['company'],
                        'matched_linkedin': linkedin_match,
                        'explanation': gap_analysis
                    })
            
            # Only include employees with identified gaps
            if employee_skill_gaps:
                self.final_skill_gap_report.append({
                    'target_employee': {
                        'company': target_company,
                        'name': target_name,
                        'position': target_emp.get('position', ''),
                        'department': target_emp.get('department', ''),
                        'linkedin': linkedin_target,
                        'skills': list(target_skills)
                    },
                    'skill_gaps': employee_skill_gaps,
                    'total_gaps': len(employee_skill_gaps),
                    'gap_importance': self._calculate_gap_importance(employee_skill_gaps)
                })
                logger.info(f"   ✅ Added {target_name} with {len(employee_skill_gaps)} skill gaps")
        
        logger.info(f"📊 Final result: Identified skill gaps for {len(self.final_skill_gap_report)} employees")
    
    def _analyze_skill_gap_with_ai(self, skill: str, target_emp: Dict, matched_emp: Dict) -> str:
        """Generate AI-powered explanation for a skill gap"""
        # Create unique key for caching explanations using company + name
        target_key = f"{target_emp.get('company', '')}_{target_emp.get('name', '')}"
        matched_key = f"{matched_emp.get('company', '')}_{matched_emp.get('name', '')}"
        explanation_key = (target_key, matched_key, skill)
        
        # Return cached explanation if available
        if explanation_key in self.ai_explanations:
            return self.ai_explanations[explanation_key]
        
        # Generate prompt for AI
        prompt = f"""
        Please analyze this skill gap in the fintech industry:
        
        Target Employee:
        - Name: {target_emp.get('name', 'N/A')}
        - Position: {target_emp.get('position', 'N/A')}
        - Department: {target_emp.get('department', 'N/A')}
        - Company: {target_emp.get('company', 'N/A')}
        
        Matched Employee:
        - Name: {matched_emp.get('name', 'N/A')}
        - Position: {matched_emp.get('position', 'N/A')}
        - Company: {matched_emp.get('company', 'N/A')}
        
        Skill Gap: {skill}
        
        Provide a detailed explanation covering:
        1. Why this skill gap exists between these employees
        2. How {matched_emp.get('company', 'the matched company')} utilizes this skill in their operations
        3. Why this skill is valuable for {target_emp.get('name', 'the target employee')}'s role at {target_emp.get('company', 'their company')}
        4. The competitive advantage this skill provides
        5. Specific fintech applications of this skill
        
        Structure your response with clear paragraphs and bullet points where appropriate.
        """
        
        # Generate AI explanation (or simulate if no API key)
        explanation = self._generate_ai_response(prompt)
        
        # Cache explanation
        self.ai_explanations[explanation_key] = explanation
        return explanation
    
    def _generate_ai_response(self, prompt: str) -> str:
        """Generate AI response using OpenAI API or simulation"""
        try:
            if openai.api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a fintech industry analyst with expertise in talent development and competitive intelligence."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message['content'].strip()
        except Exception as e:
            logger.error(f"⚠️ Failed to generate AI explanation: {str(e)}")
        
        # Fallback simulation
        return self._simulate_ai_response()
    
    def _simulate_ai_response(self) -> str:
        """Simulate AI response when API is unavailable"""
        return (
            f"This skill gap exists because the competitor invests more heavily in this specialized area. "
            f"The matched employee's company uses this skill to enhance their digital banking solutions, "
            f"providing them with a competitive edge in transaction processing efficiency. For the target employee, "
            f"acquiring this skill would enable them to develop more sophisticated risk assessment models "
            f"and improve their department's innovation capabilities."
        )
    
    def _calculate_gap_importance(self, gaps: List[Dict]) -> str:
        """Calculate overall importance of skill gaps"""
        critical_skills = {'blockchain', 'ai', 'machine learning', 'cybersecurity', 'quantitative analysis'}
        high_impact = sum(1 for gap in gaps if gap['skill'] in critical_skills)
        
        if high_impact >= 3:
            return 'Critical'
        elif high_impact >= 1:
            return 'High'
        return 'Medium'
    
    def export_final_report(self, output_dir: str = "output/final_skill_gaps"):
        """Export the final skill gap report with AI explanations"""
        logger.info(f"💾 Exporting final skill gap report to {output_dir}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Export final report
        final_report_file = os.path.join(output_dir, "top_100_skill_gaps_ai.json")
        with open(final_report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "report_date": "2023-10-15",
                "target_company": self.matcher.target_company,
                "analysis_method": "AI-powered competitive skill gap analysis",
                "employees_analyzed": len(self.final_skill_gap_report),
                "total_skill_gaps_identified": sum(emp['total_gaps'] for emp in self.final_skill_gap_report),
                "most_common_skills": self._get_most_common_skills(),
                "skill_gap_analysis": self.final_skill_gap_report
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Exported final skill gap report: {final_report_file}")
    
    def _get_most_common_skills(self) -> List[Dict]:
        """Get most common missing skills across the organization"""
        skill_counter = Counter()
        for employee in self.final_skill_gap_report:
            for gap in employee['skill_gaps']:
                skill_counter[gap['skill']] += 1
        
        return [{"skill": skill, "count": count} for skill, count in skill_counter.most_common(10)]


# Update main function with debug calls
def main():
    """Main function to run full employee matching and skill gap analysis"""
    matcher = EmployeeMatcher(target_company="Manipal Fintech")
    
    # Step 1: Load normalized data
    matcher.load_normalized_data("normalized_departments")
    
    # Step 2: Load raw data for detailed analysis
    matcher.load_raw_employee_data("employee_data")
    
    # Step 3: Perform quick matching
    matcher.perform_quick_matching()
    
    # Step 4: Perform detailed matching
    matcher.perform_detailed_matching()
    
    # Step 5: Export matching results
    matcher.export_matching_results()
    
    # Create analyzer and add debug
    analyzer = AISkillGapAnalyzer(matcher)
    
    # Debug: Check data before loading skills
    analyzer.debug_data_structure()
    
    # Load skills data
    analyzer.load_skills_data("company_skills")
    
    # Debug: Check data after loading skills
    analyzer.debug_data_structure()
    
    # Perform analysis
    analyzer.perform_ai_skill_gap_analysis()
    
    # Export results
    analyzer.export_final_report()
    
    logger.info("🏁 Ghost-3 Fractal analysis completed successfully!")

if __name__ == "__main__":
    main()