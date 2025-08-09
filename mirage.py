import json
import os
import time
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import random
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CompetitorProfile:
    """Data class for competitor information"""
    name: str
    industry: str
    similarity_score: float
    detected_via: str
    employee_count_estimate: int
    market_position: str
    key_differentiators: List[str]
    priority_level: int  # 1-10, 10 being highest priority
    
@dataclass
class GhostShadeTarget:
    """Target configuration for GHOST_SHADE agent"""
    company_name: str
    max_employees: int
    priority: int
    agent_id: str
    deployment_timestamp: str

class CompetitionDetector:
    """Detects competitors from company intelligence data"""
    
    def __init__(self):
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4o")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cx = os.getenv("GOOGLE_CSE_ID")
        
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_api_key,
        }
        
    async def analyze_for_competitors(self, intelligence_data: Dict) -> List[CompetitorProfile]:
        """
        Analyze intelligence data to detect competitors
        
        Args:
            intelligence_data: The JSON intelligence report
            
        Returns:
            List of CompetitorProfile objects
        """
        logger.info("🔍 Starting competitor detection analysis...")
        
        company_info = intelligence_data.get("company_intelligence", {})
        company_name = intelligence_data.get("report_metadata", {}).get("company_name", "Unknown")
        
        # Extract key business information
        business_context = self._extract_business_context(company_info)
        employee_insights = self._extract_employee_insights(intelligence_data.get("employee_intelligence", {}))
        
        # Use multiple detection methods
        competitors = []
        
        # Method 1: AI-powered competitor detection
        ai_competitors = await self._ai_competitor_detection(company_name, business_context)
        competitors.extend(ai_competitors)
        
        # Method 2: Industry-based competitor search
        industry_competitors = await self._industry_based_detection(business_context)
        competitors.extend(industry_competitors)
        
        # Method 3: Employee background analysis
        background_competitors = await self._employee_background_analysis(employee_insights)
        competitors.extend(background_competitors)
        
        # Deduplicate and rank competitors
        final_competitors = self._rank_and_filter_competitors(competitors, company_name)
        
        logger.info(f"✅ Detected {len(final_competitors)} high-priority competitors")
        return final_competitors[:10]  # Return top 10
    
    def _extract_business_context(self, company_info: Dict) -> Dict:
        """Extract business context from company intelligence"""
        basic_info = company_info.get("basic_info", {})
        business_analysis = company_info.get("business_analysis", {})
        
        return {
            "industry": basic_info.get("industry", ""),
            "business_model": business_analysis.get("business_model", ""),
            "description": business_analysis.get("description", ""),
            "key_products": business_analysis.get("key_products", ""),
            "market_position": business_analysis.get("market_position", ""),
            "headquarters": basic_info.get("headquarters", ""),
            "employee_estimate": basic_info.get("employee_estimate", ""),
            "founded_year": basic_info.get("founded_year", "")
        }
    
    def _extract_employee_insights(self, employee_data: Dict) -> Dict:
        """Extract insights from employee intelligence"""
        employees = employee_data.get("employees", [])
        analytics = employee_data.get("analytics", {}).get("employee_analytics", {})
        
        # Extract previous companies from employee backgrounds
        previous_companies = []
        skills = []
        positions = []
        
        for emp in employees:
            if emp.get("detailed_profile"):
                profile = emp["detailed_profile"]
                
                # Extract experience companies
                experiences = profile.get("experience", [])
                if isinstance(experiences, list):
                    for exp in experiences:
                        company = exp.get("company", "").strip()
                        if company and company not in previous_companies:
                            previous_companies.append(company)
                
                # Extract skills
                profile_skills = profile.get("skills", [])
                if isinstance(profile_skills, list):
                    for skill in profile_skills:
                        if isinstance(skill, dict):
                            skill_name = skill.get("name", "")
                            if skill_name:
                                skills.append(skill_name)
        
        return {
            "previous_companies": previous_companies[:20],  # Top 20 most common
            "top_skills": analytics.get("top_skills", [])[:15],
            "top_positions": analytics.get("top_positions", [])[:10],
            "total_employees_analyzed": len(employees)
        }
    
    async def _ai_competitor_detection(self, company_name: str, business_context: Dict) -> List[CompetitorProfile]:
        """Use AI to detect competitors based on business context"""
        if not self.azure_api_key or not self.azure_endpoint:
            logger.warning("Azure OpenAI not configured, skipping AI competitor detection")
            return []
        
        prompt = f"""
        Analyze the following company information and identify the top 15 direct competitors in the same industry/market space.
        Focus on companies that:
        1. Operate in the same industry
        2. Have similar business models
        3. Target similar customer segments
        4. Offer competing products/services
        
        Company: {company_name}
        Industry: {business_context.get('industry', 'Unknown')}
        Business Model: {business_context.get('business_model', 'Unknown')}
        Description: {business_context.get('description', 'Unknown')[:1000]}
        Market Position: {business_context.get('market_position', 'Unknown')}
        
        Return your response as a JSON array with this exact format:
        [
          {{
            "name": "Competitor Name",
            "industry": "Industry",
            "similarity_score": 8.5,
            "market_position": "Market leader/Challenger/Niche player",
            "key_differentiators": ["differentiator1", "differentiator2"],
            "employee_count_estimate": 500
          }}
        ]
        
        Only return valid JSON, no additional text.
        """
        
        try:
            response = await self._azure_chat_completion([
                {"role": "system", "content": "You are a business intelligence analyst specializing in competitive analysis. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ])
            
            if response:
                competitors_data = json.loads(response)
                competitors = []
                
                for comp_data in competitors_data:
                    competitor = CompetitorProfile(
                        name=comp_data.get("name", ""),
                        industry=comp_data.get("industry", ""),
                        similarity_score=float(comp_data.get("similarity_score", 0)),
                        detected_via="AI Analysis",
                        employee_count_estimate=int(comp_data.get("employee_count_estimate", 0)),
                        market_position=comp_data.get("market_position", ""),
                        key_differentiators=comp_data.get("key_differentiators", []),
                        priority_level=min(10, max(1, int(comp_data.get("similarity_score", 0))))
                    )
                    competitors.append(competitor)
                
                logger.info(f"🤖 AI detected {len(competitors)} competitors")
                return competitors
                
        except Exception as e:
            logger.error(f"AI competitor detection failed: {e}")
            return []
    
    async def _industry_based_detection(self, business_context: Dict) -> List[CompetitorProfile]:
        """Search for competitors based on industry keywords"""
        if not self.google_api_key or not self.google_cx:
            logger.warning("Google Search not configured, skipping industry-based detection")
            return []
        
        industry = business_context.get("industry", "")
        business_model = business_context.get("business_model", "")
        
        if not industry:
            return []
        
        # Generate search queries for competitor discovery
        search_queries = [
            f'"{industry}" companies leaders market',
            f'"{industry}" competitors comparison',
            f'top companies {industry} sector',
            f'{business_model} companies {industry}',
            f'"{industry}" market players analysis'
        ]
        
        competitors = []
        
        for query in search_queries[:3]:  # Limit to avoid rate limits
            try:
                company_names = await self._search_for_companies(query)
                for name in company_names:
                    competitor = CompetitorProfile(
                        name=name,
                        industry=industry,
                        similarity_score=6.0,  # Medium confidence
                        detected_via="Industry Search",
                        employee_count_estimate=0,  # Unknown
                        market_position="Unknown",
                        key_differentiators=[],
                        priority_level=6
                    )
                    competitors.append(competitor)
                    
                await asyncio.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Industry search failed for query '{query}': {e}")
        
        logger.info(f"🔍 Industry search detected {len(competitors)} potential competitors")
        return competitors
    
    async def _employee_background_analysis(self, employee_insights: Dict) -> List[CompetitorProfile]:
        """Analyze employee backgrounds to find previous employers (potential competitors)"""
        previous_companies = employee_insights.get("previous_companies", [])
        
        if not previous_companies:
            return []
        
        competitors = []
        
        # Filter out obvious non-competitors (consulting firms, universities, etc.)
        exclude_keywords = [
            "university", "college", "consulting", "accenture", "deloitte", 
            "pwc", "kpmg", "ey", "mckinsey", "bain", "bcg", "freelance",
            "self-employed", "startup", "stealth"
        ]
        
        for company in previous_companies[:15]:  # Top 15 previous companies
            if any(keyword in company.lower() for keyword in exclude_keywords):
                continue
                
            competitor = CompetitorProfile(
                name=company,
                industry="Unknown",
                similarity_score=7.0,  # High confidence - employees moved between companies
                detected_via="Employee Background Analysis",
                employee_count_estimate=0,
                market_position="Unknown",
                key_differentiators=["Talent overlap"],
                priority_level=7
            )
            competitors.append(competitor)
        
        logger.info(f"👥 Employee analysis detected {len(competitors)} competitors")
        return competitors
    
    async def _search_for_companies(self, query: str) -> List[str]:
        """Search for companies using Google Custom Search"""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'key': self.google_api_key,
            'cx': self.google_cx,
            'num': 5
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        companies = []
                        
                        for item in data.get('items', []):
                            title = item.get('title', '')
                            snippet = item.get('snippet', '')
                            
                            # Extract company names from title and snippet
                            company_names = self._extract_company_names(title + " " + snippet)
                            companies.extend(company_names)
                        
                        return list(set(companies))  # Remove duplicates
                    else:
                        return []
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []
    
    def _extract_company_names(self, text: str) -> List[str]:
        """Extract potential company names from text"""
        # Simple regex patterns for company names
        patterns = [
            r'\b[A-Z][a-zA-Z0-9\s]+(?:Inc|Corp|LLC|Ltd|Company|Co|Technologies|Tech|Solutions|Systems|Group|Holdings)\b',
            r'\b[A-Z][a-zA-Z]+[A-Z][a-zA-Z]+\b',  # CamelCase words
        ]
        
        companies = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            companies.extend(matches)
        
        # Filter out common false positives
        exclude = ['Google', 'Microsoft', 'Amazon', 'Apple', 'Facebook', 'LinkedIn', 'Twitter']
        companies = [comp.strip() for comp in companies if comp.strip() not in exclude and len(comp.strip()) > 3]
        
        return companies[:3]  # Return top 3 per text
    
    def _rank_and_filter_competitors(self, competitors: List[CompetitorProfile], original_company: str) -> List[CompetitorProfile]:
        """Rank competitors by priority and filter duplicates"""
        # Remove the original company and obvious duplicates
        filtered = []
        seen_names = set()
        
        for comp in competitors:
            comp_name_lower = comp.name.lower().strip()
            original_lower = original_company.lower().strip()
            
            # Skip if it's the original company
            if comp_name_lower == original_lower:
                continue
            
            # Skip if we've already seen this company
            if comp_name_lower in seen_names:
                continue
                
            # Skip if name is too short or generic
            if len(comp.name.strip()) < 3:
                continue
                
            seen_names.add(comp_name_lower)
            filtered.append(comp)
        
        # Sort by similarity score and priority level
        ranked = sorted(filtered, key=lambda x: (x.similarity_score, x.priority_level), reverse=True)
        
        logger.info(f"📊 Ranked and filtered to {len(ranked)} unique competitors")
        return ranked
    
    async def _azure_chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        """Make Azure OpenAI API call"""
        url = f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment_id}/chat/completions?api-version={self.azure_api_version}"
        
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 3000
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        logger.error(f"Azure API error: {response.status}")
                        return ""
        except Exception as e:
            logger.error(f"Azure API call failed: {e}")
            return ""

class GhostShadeAgentManager:
    """Manages deployment and coordination of GHOST_SHADE agents"""
    
    def __init__(self):
        self.active_agents = {}
        self.completed_missions = []
        self.data_dir = "ghost_shade_intelligence"
        os.makedirs(self.data_dir, exist_ok=True)
        
    async def deploy_ghost_shade_agents(self, competitors: List[CompetitorProfile]) -> List[GhostShadeTarget]:
        """
        Deploy GHOST_SHADE agents to gather employee intelligence on competitors
        
        Args:
            competitors: List of competitor profiles
            
        Returns:
            List of deployment targets
        """
        logger.info(f"👻 Deploying GHOST_SHADE agents for {len(competitors)} competitors")
        
        targets = []
        
        for i, competitor in enumerate(competitors[:10]):  # Max 10 agents
            agent_id = f"GHOST_SHADE_{i+1:02d}"
            
            # Determine employee target count based on priority and estimated size
            if competitor.employee_count_estimate > 0:
                max_employees = min(100, max(20, competitor.employee_count_estimate // 10))
            else:
                # Default based on priority
                max_employees = 50 if competitor.priority_level >= 7 else 30
            
            target = GhostShadeTarget(
                company_name=competitor.name,
                max_employees=max_employees,
                priority=competitor.priority_level,
                agent_id=agent_id,
                deployment_timestamp=datetime.utcnow().isoformat()
            )
            
            targets.append(target)
            self.active_agents[agent_id] = {
                "target": target,
                "status": "DEPLOYED",
                "start_time": time.time(),
                "competitor_profile": competitor
            }
            
            logger.info(f"👻 {agent_id} deployed -> {competitor.name} | Target: {max_employees} employees | Priority: {competitor.priority_level}")
        
        # Launch agents in parallel
        await self._execute_parallel_ghost_shade_missions(targets)
        
        return targets
    
    async def _execute_parallel_ghost_shade_missions(self, targets: List[GhostShadeTarget]):
        """Execute GHOST_SHADE missions in parallel"""
        logger.info(f"🚀 Launching {len(targets)} parallel GHOST_SHADE missions")
        
        # Import the original GoogleCSEEmployeeFinder and BrightDataScraper
        from dataclasses import dataclass
        
        @dataclass
        class RawEmployee:
            name: str
            linkedin_url: str
            snippet: str
            company: str
        
        # Create tasks for parallel execution
        tasks = []
        for target in targets:
            task = asyncio.create_task(self._ghost_shade_mission(target))
            tasks.append(task)
            
            # Stagger deployments to avoid rate limits
            await asyncio.sleep(2)
        
        # Wait for all missions to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_missions = 0
        for i, result in enumerate(results):
            agent_id = targets[i].agent_id
            if isinstance(result, Exception):
                logger.error(f"❌ {agent_id} mission failed: {result}")
                self.active_agents[agent_id]["status"] = "FAILED"
            else:
                if result:
                    successful_missions += 1
                    self.active_agents[agent_id]["status"] = "COMPLETED"
                    self.completed_missions.append(result)
                    logger.info(f"✅ {agent_id} mission completed successfully")
                else:
                    self.active_agents[agent_id]["status"] = "NO_DATA"
                    logger.warning(f"⚠️ {agent_id} completed but found no data")
        
        logger.info(f"🎯 GHOST_SHADE missions completed: {successful_missions}/{len(targets)} successful")
    
    async def _ghost_shade_mission(self, target: GhostShadeTarget) -> Optional[Dict]:
        """Execute a single GHOST_SHADE mission"""
        logger.info(f"👻 {target.agent_id} starting mission: {target.company_name}")
        
        try:
            # Initialize the employee finder (from original code)
            from shade import GoogleCSEEmployeeFinder, BrightDataScraper
            
            employee_finder = GoogleCSEEmployeeFinder()
            
            # Find employees for the competitor
            employees = await asyncio.to_thread(
                employee_finder.find_employees, 
                target.company_name, 
                target.max_employees
            )
            
            if not employees:
                logger.warning(f"👻 {target.agent_id} found no employees for {target.company_name}")
                return None
            
            logger.info(f"👻 {target.agent_id} found {len(employees)} employees for {target.company_name}")
            
            # Scrape detailed profiles
            scraper = BrightDataScraper()
            linkedin_urls = [emp.linkedin_url for emp in employees]
            detailed_profiles = scraper.scrape_profiles_in_batches(linkedin_urls, batch_size=10)
            
            # Create competitor intelligence report
            competitor_data = {
                "mission_metadata": {
                    "agent_id": target.agent_id,
                    "target_company": target.company_name,
                    "mission_timestamp": target.deployment_timestamp,
                    "completion_timestamp": datetime.utcnow().isoformat(),
                    "mission_status": "COMPLETED"
                },
                "employee_intelligence": {
                    "summary": {
                        "total_employees_found": len(employees),
                        "detailed_profiles_scraped": len(detailed_profiles) if detailed_profiles else 0,
                        "scraping_success_rate": (len(detailed_profiles) / len(employees) * 100) if detailed_profiles and employees else 0
                    },
                    "employees": []
                }
            }
            
            # Process employee data (similar to original code)
            detailed_profiles_map = {}
            if detailed_profiles:
                for profile in detailed_profiles:
                    url = profile.get('url', '')
                    if url:
                        detailed_profiles_map[url] = profile
            
            for raw_emp in employees:
                employee_data = {
                    "basic_info": {
                        "name": raw_emp.name,
                        "linkedin_url": raw_emp.linkedin_url,
                        "company": raw_emp.company,
                        "search_snippet": raw_emp.snippet
                    },
                    "detailed_profile": detailed_profiles_map.get(raw_emp.linkedin_url),
                    "data_status": {
                        "found_in_search": True,
                        "detailed_scraped": raw_emp.linkedin_url in detailed_profiles_map
                    }
                }
                
                competitor_data["employee_intelligence"]["employees"].append(employee_data)
            
            # Save competitor intelligence
            safe_company_name = re.sub(r'[^a-zA-Z0-9_-]', '_', target.company_name.lower())
            filename = f"{safe_company_name}_report.json"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(competitor_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"👻 {target.agent_id} saved intelligence to: {filepath}")
            
            return competitor_data
            
        except Exception as e:
            logger.error(f"👻 {target.agent_id} mission failed: {e}")
            return None
    
    def generate_mission_summary(self) -> Dict:
        """Generate summary of all GHOST_SHADE missions"""
        total_agents = len(self.active_agents)
        completed = len([a for a in self.active_agents.values() if a["status"] == "COMPLETED"])
        failed = len([a for a in self.active_agents.values() if a["status"] == "FAILED"])
        no_data = len([a for a in self.active_agents.values() if a["status"] == "NO_DATA"])
        
        summary = {
            "mission_overview": {
                "total_agents_deployed": total_agents,
                "successful_missions": completed,
                "failed_missions": failed,
                "no_data_missions": no_data,
                "success_rate": (completed / total_agents * 100) if total_agents > 0 else 0
            },
            "agent_details": [],
            "intelligence_gathered": {
                "total_competitor_companies": len(self.completed_missions),
                "total_employees_found": sum(mission["employee_intelligence"]["summary"]["total_employees_found"] for mission in self.completed_missions),
                "total_detailed_profiles": sum(mission["employee_intelligence"]["summary"]["detailed_profiles_scraped"] for mission in self.completed_missions)
            }
        }
        
        for agent_id, agent_data in self.active_agents.items():
            target = agent_data["target"]
            competitor = agent_data["competitor_profile"]
            
            agent_detail = {
                "agent_id": agent_id,
                "target_company": target.company_name,
                "status": agent_data["status"],
                "priority": target.priority,
                "competitor_similarity": competitor.similarity_score,
                "detection_method": competitor.detected_via
            }
            
            summary["agent_details"].append(agent_detail)
        
        return summary

class GhostMirage:
    """Main GHOST_MIRAGE system - Competition analyzer and agent coordinator"""
    
    def __init__(self):
        self.detector = CompetitionDetector()
        self.agent_manager = GhostShadeAgentManager()
        self.output_dir = "ghost_mirage_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
    async def analyze_and_deploy(self, intelligence_report_path: str) -> Dict:
        """
        Main GHOST_MIRAGE operation:
        1. Analyze intelligence report for competitors
        2. Deploy GHOST_SHADE agents to gather competitor employee data
        
        Args:
            intelligence_report_path: Path to the JSON intelligence report
            
        Returns:
            Complete GHOST_MIRAGE operation summary
        """
        logger.info("🌟 GHOST_MIRAGE ACTIVATED 🌟")
        logger.info("🔍 Competition Analysis & Ghost Agent Deployment System")
        
        # Load intelligence report
        try:
            with open(intelligence_report_path, 'r', encoding='utf-8') as f:
                intelligence_data = json.load(f)
                
            company_name = intelligence_data.get("report_metadata", {}).get("company_name", "Unknown")
            logger.info(f"📊 Analyzing intelligence report for: {company_name}")
            
        except Exception as e:
            logger.error(f"Failed to load intelligence report: {e}")
            return {"error": "Failed to load intelligence report"}
        
        # Phase 1: Competitor Detection
        logger.info("\n🔍 PHASE 1: COMPETITOR DETECTION")
        logger.info("=" * 50)
        
        competitors = await self.detector.analyze_for_competitors(intelligence_data)
        
        if not competitors:
            logger.warning("No competitors detected. Operation terminated.")
            return {"error": "No competitors detected"}
        
        logger.info(f"🎯 Detected {len(competitors)} priority competitors:")
        for i, comp in enumerate(competitors, 1):
            logger.info(f"   {i:2d}. {comp.name} | Score: {comp.similarity_score:.1f} | via {comp.detected_via}")
        
        # Phase 2: GHOST_SHADE Agent Deployment
        logger.info(f"\n👻 PHASE 2: GHOST_SHADE AGENT DEPLOYMENT")
        logger.info("=" * 50)
        
        targets = await self.agent_manager.deploy_ghost_shade_agents(competitors)
        
        # Phase 3: Generate Final Report
        logger.info(f"\n📋 PHASE 3: FINAL INTELLIGENCE SYNTHESIS")
        logger.info("=" * 50)
        
        mission_summary = self.agent_manager.generate_mission_summary()
        
        # Create comprehensive GHOST_MIRAGE report
        ghost_mirage_report = {
            "ghost_mirage_metadata": {
                "operation_id": f"MIRAGE_{int(time.time())}",
                "target_company": company_name,
                "operation_timestamp": datetime.utcnow().isoformat(),
                "analysis_source": intelligence_report_path,
                "total_competitors_detected": len(competitors),
                "ghost_shade_agents_deployed": len(targets)
            },
            "competitor_analysis": {
                "detection_summary": {
                    "total_competitors_found": len(competitors),
                    "ai_detected": len([c for c in competitors if c.detected_via == "AI Analysis"]),
                    "industry_detected": len([c for c in competitors if c.detected_via == "Industry Search"]),
                    "employee_background_detected": len([c for c in competitors if c.detected_via == "Employee Background Analysis"])
                },
                "competitor_profiles": [
                    {
                        "name": comp.name,
                        "industry": comp.industry,
                        "similarity_score": comp.similarity_score,
                        "priority_level": comp.priority_level,
                        "detection_method": comp.detected_via,
                        "market_position": comp.market_position,
                        "key_differentiators": comp.key_differentiators,
                        "employee_count_estimate": comp.employee_count_estimate
                    }
                    for comp in competitors
                ]
            },
            "ghost_shade_operations": mission_summary,
            "executive_summary": self._generate_executive_summary(company_name, competitors, mission_summary)
        }
        
        # Save GHOST_MIRAGE report
        report_filename = f"GHOST_MIRAGE_Report_{company_name.replace(' ', '_')}_{int(time.time())}.json"
        report_path = os.path.join(self.output_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(ghost_mirage_report, f, indent=2, ensure_ascii=False)
        
        # Final Summary
        logger.info(f"\n🌟 GHOST_MIRAGE OPERATION COMPLETED 🌟")
        logger.info(f"📊 Operation Summary:")
        logger.info(f"   • Target Company: {company_name}")
        logger.info(f"   • Competitors Detected: {len(competitors)}")
        logger.info(f"   • GHOST_SHADE Agents Deployed: {len(targets)}")
        logger.info(f"   • Successful Intelligence Missions: {mission_summary['mission_overview']['successful_missions']}")
        logger.info(f"   • Total Employee Profiles Gathered: {mission_summary['intelligence_gathered']['total_employees_found']}")
        logger.info(f"   • Success Rate: {mission_summary['mission_overview']['success_rate']:.1f}%")
        logger.info(f"📁 Complete GHOST_MIRAGE Report: {report_path}")
        
        return ghost_mirage_report
    
    def _generate_executive_summary(self, company_name: str, competitors: List[CompetitorProfile], mission_summary: Dict) -> str:
        """Generate executive summary of GHOST_MIRAGE operation"""
        summary_parts = [
            f"GHOST_MIRAGE competitive intelligence operation executed for {company_name}.",
            f"Detected {len(competitors)} priority competitors using multi-method analysis:",
            f"AI-powered detection, industry research, and employee background analysis."
        ]
        
        if competitors:
            top_competitor = max(competitors, key=lambda x: x.similarity_score)
            summary_parts.append(f"Primary threat identified: {top_competitor.name} (similarity score: {top_competitor.similarity_score:.1f}).")
        
        successful_missions = mission_summary['mission_overview']['successful_missions']
        total_employees = mission_summary['intelligence_gathered']['total_employees_found']
        
        summary_parts.extend([
            f"Deployed {len(competitors)} GHOST_SHADE agents for employee intelligence gathering.",
            f"Successfully completed {successful_missions} intelligence missions.",
            f"Gathered intelligence on {total_employees} competitor employees across all targets.",
            f"Operation provides comprehensive competitive landscape analysis and employee movement patterns."
        ])
        
        return " ".join(summary_parts)

# Additional utility functions for GHOST_MIRAGE

class GhostMirageAnalytics:
    """Advanced analytics for GHOST_MIRAGE operations"""
    
    def __init__(self):
        pass
    
    def analyze_competitive_landscape(self, ghost_mirage_report: Dict) -> Dict:
        """Analyze the competitive landscape from GHOST_MIRAGE report"""
        competitors = ghost_mirage_report.get("competitor_analysis", {}).get("competitor_profiles", [])
        
        if not competitors:
            return {"error": "No competitor data available"}
        
        # Market positioning analysis
        position_distribution = {}
        for comp in competitors:
            position = comp.get("market_position", "Unknown")
            position_distribution[position] = position_distribution.get(position, 0) + 1
        
        # Similarity score analysis
        high_threat = [c for c in competitors if c.get("similarity_score", 0) >= 8.0]
        medium_threat = [c for c in competitors if 6.0 <= c.get("similarity_score", 0) < 8.0]
        low_threat = [c for c in competitors if c.get("similarity_score", 0) < 6.0]
        
        # Detection method effectiveness
        detection_methods = {}
        for comp in competitors:
            method = comp.get("detection_method", "Unknown")
            detection_methods[method] = detection_methods.get(method, 0) + 1
        
        analytics = {
            "threat_level_distribution": {
                "high_threat_competitors": len(high_threat),
                "medium_threat_competitors": len(medium_threat),
                "low_threat_competitors": len(low_threat)
            },
            "market_position_analysis": position_distribution,
            "detection_method_effectiveness": detection_methods,
            "top_threats": [
                {
                    "name": comp["name"],
                    "similarity_score": comp["similarity_score"],
                    "threat_level": "HIGH" if comp["similarity_score"] >= 8.0 else "MEDIUM" if comp["similarity_score"] >= 6.0 else "LOW"
                }
                for comp in sorted(competitors, key=lambda x: x.get("similarity_score", 0), reverse=True)[:5]
            ]
        }
        
        return analytics
    
    def analyze_employee_movement_patterns(self, ghost_shade_data: List[Dict]) -> Dict:
        """Analyze employee movement patterns across competitors"""
        if not ghost_shade_data:
            return {"error": "No GHOST_SHADE data available"}
        
        # Aggregate employee data from all competitors
        all_employees = []
        company_employee_counts = {}
        
        for mission_data in ghost_shade_data:
            company_name = mission_data.get("mission_metadata", {}).get("target_company", "Unknown")
            employees = mission_data.get("employee_intelligence", {}).get("employees", [])
            
            company_employee_counts[company_name] = len(employees)
            
            for emp in employees:
                if emp.get("detailed_profile"):
                    all_employees.append({
                        "company": company_name,
                        "profile": emp["detailed_profile"]
                    })
        
        # Analyze common skills across competitors
        skill_frequency = {}
        position_frequency = {}
        previous_companies = {}
        
        for emp_data in all_employees:
            profile = emp_data["profile"]
            company = emp_data["company"]
            
            # Skills analysis
            skills = profile.get("skills", [])
            if isinstance(skills, list):
                for skill in skills:
                    if isinstance(skill, dict):
                        skill_name = skill.get("name", "")
                        if skill_name:
                            skill_frequency[skill_name] = skill_frequency.get(skill_name, 0) + 1
            
            # Position analysis
            experiences = profile.get("experience", [])
            if isinstance(experiences, list) and experiences:
                current_position = experiences[0].get("title", "")
                if current_position:
                    position_frequency[current_position] = position_frequency.get(current_position, 0) + 1
                
                # Previous companies analysis
                for exp in experiences[1:]:  # Skip current company
                    prev_company = exp.get("company", "")
                    if prev_company:
                        if prev_company not in previous_companies:
                            previous_companies[prev_company] = []
                        previous_companies[prev_company].append(company)
        
        # Find talent overlap patterns
        talent_overlap = {}
        for prev_company, current_companies in previous_companies.items():
            if len(set(current_companies)) > 1:  # Multiple competitors hired from same company
                talent_overlap[prev_company] = {
                    "competitor_count": len(set(current_companies)),
                    "total_hires": len(current_companies),
                    "competitors": list(set(current_companies))
                }
        
        analytics = {
            "employee_distribution": company_employee_counts,
            "top_skills_across_competitors": [
                {"skill": skill, "frequency": freq}
                for skill, freq in sorted(skill_frequency.items(), key=lambda x: x[1], reverse=True)[:20]
            ],
            "common_positions": [
                {"position": pos, "frequency": freq}
                for pos, freq in sorted(position_frequency.items(), key=lambda x: x[1], reverse=True)[:15]
            ],
            "talent_overlap_sources": dict(sorted(talent_overlap.items(), key=lambda x: x[1]["total_hires"], reverse=True)[:10]),
            "insights": {
                "total_competitor_employees_analyzed": len(all_employees),
                "companies_with_talent_overlap": len(talent_overlap),
                "most_common_skill": max(skill_frequency.items(), key=lambda x: x[1])[0] if skill_frequency else "Unknown",
                "talent_concentration_risk": len([company for company, data in talent_overlap.items() if data["total_hires"] > 3])
            }
        }
        
        return analytics

# Main execution function
async def execute_ghost_mirage_operation(intelligence_report_path: str):
    """
    Main function to execute complete GHOST_MIRAGE operation
    
    Args:
        intelligence_report_path: Path to the company intelligence JSON report
    """
    
    print("🌟" * 25)
    print("🌟  GHOST_MIRAGE SYSTEM ACTIVATED  🌟")
    print("🌟     Competition Intelligence     🌟") 
    print("🌟    & Agent Deployment System    🌟")
    print("🌟" * 25)
    print()
    
    # Check if intelligence report exists
    if not os.path.exists(intelligence_report_path):
        print(f"❌ Intelligence report not found: {intelligence_report_path}")
        return
    
    # Initialize GHOST_MIRAGE
    ghost_mirage = GhostMirage()
    
    try:
        # Execute main operation
        operation_result = await ghost_mirage.analyze_and_deploy(intelligence_report_path)
        
        if "error" in operation_result:
            print(f"❌ Operation failed: {operation_result['error']}")
            return
        
        # Generate advanced analytics
        print(f"\n🔬 GENERATING ADVANCED ANALYTICS...")
        analytics_engine = GhostMirageAnalytics()
        
        # Competitive landscape analysis
        landscape_analysis = analytics_engine.analyze_competitive_landscape(operation_result)
        
        # Employee movement pattern analysis (if GHOST_SHADE data available)
        employee_analysis = {}
        if ghost_mirage.agent_manager.completed_missions:
            employee_analysis = analytics_engine.analyze_employee_movement_patterns(
                ghost_mirage.agent_manager.completed_missions
            )
        
        # Save enhanced analytics
        enhanced_report = {
            **operation_result,
            "advanced_analytics": {
                "competitive_landscape": landscape_analysis,
                "employee_movement_patterns": employee_analysis,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
        }
        
        # Save enhanced report
        company_name = operation_result["ghost_mirage_metadata"]["target_company"]
        enhanced_filename = f"GHOST_MIRAGE_Enhanced_{company_name.replace(' ', '_')}_{int(time.time())}.json"
        enhanced_path = os.path.join(ghost_mirage.output_dir, enhanced_filename)
        
        with open(enhanced_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_report, f, indent=2, ensure_ascii=False)
        
        # Final summary with analytics
        print(f"\n📊 ADVANCED ANALYTICS SUMMARY:")
        print(f"   • Competitive Landscape Analyzed: ✅")
        if landscape_analysis.get("top_threats"):
            print(f"   • Top Threat: {landscape_analysis['top_threats'][0]['name']}")
        
        if employee_analysis.get("insights"):
            insights = employee_analysis["insights"]
            print(f"   • Competitor Employees Analyzed: {insights.get('total_competitor_employees_analyzed', 0)}")
            print(f"   • Companies with Talent Overlap: {insights.get('companies_with_talent_overlap', 0)}")
        
        print(f"📁 Enhanced Report: {enhanced_path}")
        print(f"\n🎯 GHOST_MIRAGE OPERATION STATUS: MISSION ACCOMPLISHED ✅")
        
        return enhanced_report
        
    except Exception as e:
        logger.error(f"GHOST_MIRAGE operation failed: {e}")
        print(f"❌ GHOST_MIRAGE operation failed: {e}")
        return None

# CLI Interface
def main():
    """Command line interface for GHOST_MIRAGE"""
    
    print("🌟 GHOST_MIRAGE - Competition Intelligence System 🌟")
    print("=" * 60)
    
    # Check environment variables
    required_env_vars = [
        ("GOOGLE_API_KEY", "AIzaSyDSzs48YYQsaJKQItxjXlHFe6TV3fnwEDc"),
        ("GOOGLE_CSE_ID", "64ea1a8c4f8334d71"),
    ]
    
    optional_env_vars = [
        ("AZURE_OPENAI_API_KEY", "Azure OpenAI API key for AI analysis"),
        ("AZURE_OPENAI_ENDPOINT", "Azure OpenAI endpoint URL"),
        ("BRIGHT_DATA_API_KEY", "Bright Data API key for LinkedIn scraping")
    ]
    
    missing_required = []
    for var_name, description in required_env_vars:
        if not os.getenv(var_name):
            missing_required.append((var_name, description))
    
    if missing_required:
        print("❌ Missing required environment variables:")
        for var_name, description in missing_required:
            print(f"   {var_name}: {description}")
        print("\nPlease set these environment variables and try again.")
        return
    
    print("📋 Configuration Status:")
    for var_name, description in required_env_vars + optional_env_vars:
        status = "✅ Configured" if os.getenv(var_name) else "⚠️ Not configured"
        print(f"   {var_name}: {status}")
    print()
    
    # Get intelligence report path
    try:
        report_path = input("Enter path to intelligence report (JSON file): ").strip()
        if not report_path:
            raise ValueError("Intelligence report path is required")
        
        if not os.path.exists(report_path):
            raise ValueError(f"File not found: {report_path}")
        
        print(f"\n🚀 Starting GHOST_MIRAGE operation...")
        print(f"   Intelligence Report: {report_path}")
        
        # Run the operation
        asyncio.run(execute_ghost_mirage_operation(report_path))
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    except KeyboardInterrupt:
        print(f"\n👋 Operation cancelled by user")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return

if __name__ == "__main__":
    main()