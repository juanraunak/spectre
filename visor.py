import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import time
from urllib.parse import urljoin, urlparse
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class CompanyData:
    name: str
    website: str = ""
    description: str = ""
    industry: str = ""
    headquarters: str = ""
    founded_year: str = ""
    employee_estimate: str = ""
    revenue_estimate: str = ""
    funding_info: str = ""
    tech_stack: List[str] = None
    social_links: Dict[str, str] = None
    recent_news: List[Dict[str, str]] = None
    financial_data: Dict[str, Any] = None
    business_model: str = ""
    key_products: str = ""
    market_position: str = ""
    competitive_analysis: str = ""
    growth_metrics: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if self.tech_stack is None:
            self.tech_stack = []
        if self.social_links is None:
            self.social_links = {}
        if self.recent_news is None:
            self.recent_news = []
        if self.financial_data is None:
            self.financial_data = {}

class CompanyReportGenerator:
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        self.max_concurrent_requests = 5
        self.delay_between_requests = 1.0
        self.request_timeout = 15
        self.max_links_per_query = 5  # Reduced from 8 to 5

    async def azure_chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4000) -> str:
        """Make Azure OpenAI API call with error handling"""
        url = f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment_id}/chat/completions?api-version={self.azure_api_version}"
        
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
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

    def generate_search_queries(self, company_name: str) -> List[str]:
        """Generate strategic search queries for comprehensive company data"""
        queries = [
            f'"{company_name}" official website about company',
            f'"{company_name}" company information revenue financial results',
            f'"{company_name}" funding investment crunchbase',
            f'"{company_name}" news latest updates press releases',
            f'"{company_name}" business model products services',
            f'"{company_name}" employees team size headquarters location',
            f'"{company_name}" technology stack engineering',
            f'"{company_name}" industry analysis market position'
        ]
        return queries

    async def google_search(self, query: str) -> List[str]:
        """Perform Google Custom Search API call - limited to 5 results"""
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'key': self.google_api_key,
            'cx': self.google_cx,
            'num': self.max_links_per_query  # Only get 5 links per query
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=self.request_timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [item['link'] for item in data.get('items', [])[:self.max_links_per_query]]
                    else:
                        logger.error(f"Google Search API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Google search failed for query '{query}': {e}")
            return []

    async def fetch_and_clean_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch and clean webpage content with better error handling"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={'User-Agent': self.headers['User-Agent']}, 
                                     timeout=self.request_timeout) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove unwanted elements
                        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                            element.decompose()
                        
                        # Extract metadata
                        title = soup.title.string.strip() if soup.title else ""
                        meta_desc = soup.find("meta", attrs={"name": "description"})
                        description = meta_desc.get('content', '').strip() if meta_desc else ""
                        
                        # Extract main content
                        content_selectors = ['main', 'article', '.content', '.post-content', '#content', '.main-content', 'body']
                        content_text = ""
                        
                        for selector in content_selectors:
                            element = soup.select_one(selector)
                            if element:
                                content_text = element.get_text(strip=True)
                                break
                        
                        if not content_text:
                            content_text = soup.get_text(strip=True)
                        
                        # Clean text
                        content_text = re.sub(r'\s+', ' ', content_text)
                        content_text = re.sub(r'[^\w\s.,;:!?()-]', '', content_text)
                        
                        return {
                            'url': url,
                            'title': title,
                            'description': description,
                            'content': content_text[:15000],  # Reduced content size
                            'social_links': self.extract_social_links(soup),
                            'company_info': self.extract_company_info(soup, content_text)
                        }
                    else:
                        logger.warning(f"Failed to fetch {url}: Status {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def extract_social_links(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract social media links from webpage"""
        social_links = {
            'linkedin': '',
            'twitter': '',
            'facebook': '',
            'instagram': '',
            'youtube': '',
            'crunchbase': '',
            'github': ''
        }
        
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            if 'linkedin.com' in href:
                social_links['linkedin'] = link['href']
            elif 'twitter.com' in href or 'x.com' in href:
                social_links['twitter'] = link['href']
            elif 'facebook.com' in href:
                social_links['facebook'] = link['href']
            elif 'instagram.com' in href:
                social_links['instagram'] = link['href']
            elif 'youtube.com' in href:
                social_links['youtube'] = link['href']
            elif 'crunchbase.com' in href:
                social_links['crunchbase'] = link['href']
            elif 'github.com' in href:
                social_links['github'] = link['href']
        
        return {k: v for k, v in social_links.items() if v}

    def extract_company_info(self, soup: BeautifulSoup, content: str) -> Dict[str, str]:
        """Extract company information using regex patterns"""
        info = {}
        
        # Extract revenue figures
        revenue_patterns = [
            r'\$(\d+(?:\.\d+)?)\s*(?:billion|million|B|M)\s*(?:in\s*)?(?:revenue|sales|ARR|MRR)',
            r'revenue\s*of\s*\$(\d+(?:\.\d+)?)\s*(?:billion|million|B|M)',
            r'(\d+(?:\.\d+)?)\s*(?:billion|million|B|M)\s*(?:in\s*)?revenue'
        ]
        
        for pattern in revenue_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                info['revenue_estimate'] = match.group(0)
                break
        
        # Extract employee count
        employee_patterns = [
            r'(\d+(?:,\d+)?)\s*employees',
            r'team\s*of\s*(\d+(?:,\d+)?)',
            r'(\d+(?:,\d+)?)\s*people\s*(?:work|employed)',
            r'workforce\s*of\s*(\d+(?:,\d+)?)'
        ]
        
        for pattern in employee_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                info['employee_estimate'] = match.group(0)
                break
        
        # Extract founding year
        year_pattern = r'(?:founded|established|started|launched)(?:\s+in)?\s*(\d{4})'
        match = re.search(year_pattern, content, re.IGNORECASE)
        if match:
            info['founded_year'] = match.group(1)
        
        # Extract headquarters
        hq_patterns = [
            r'headquarters\s*(?:in|at|located)?\s*([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)',
            r'based\s*(?:in|at)\s*([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)',
            r'located\s*(?:in|at)\s*([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)'
        ]
        
        for pattern in hq_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                info['headquarters'] = match.group(1)
                break
        
        return info

    async def reasoning_analysis(self, content: str, company_name: str) -> Dict[str, str]:
        """Use GPT reasoning to analyze and extract key information"""
        reasoning_prompt = f"""
        You are an expert business analyst with advanced reasoning capabilities. Analyze the following content about {company_name} and extract key information using step-by-step reasoning.

        REASONING PROCESS:
        1. First, identify if this content is actually about {company_name} or a different company
        2. Extract and validate key facts using context clues
        3. Determine reliability of information based on source type
        4. Cross-reference information for consistency
        5. Provide confidence scores for each extracted fact

        EXTRACT THE FOLLOWING WITH REASONING:
        - Company Name Verification: Is this definitely about {company_name}? (High/Medium/Low confidence)
        - Business Model: What does the company do? How do they make money?
        - Industry: What industry/sector does the company operate in?
        - Revenue Information: Any financial figures mentioned?
        - Employee Count: Team size or workforce information?
        - Founding Information: When was it founded? By whom?
        - Location: Where is the company based?
        - Products/Services: What are their main offerings?
        - Recent News: Any recent developments or announcements?
        - Technology Stack: What technologies do they use?

        Please provide your analysis in the following JSON format:
        {{
            "company_verification": {{"confidence": "High/Medium/Low", "reasoning": "explanation"}},
            "business_model": {{"info": "extracted info", "confidence": "High/Medium/Low", "reasoning": "explanation"}},
            "industry": {{"info": "extracted info", "confidence": "High/Medium/Low", "reasoning": "explanation"}},
            "revenue": {{"info": "extracted info", "confidence": "High/Medium/Low", "reasoning": "explanation"}},
            "employees": {{"info": "extracted info", "confidence": "High/Medium/Low", "reasoning": "explanation"}},
            "founded": {{"info": "extracted info", "confidence": "High/Medium/Low", "reasoning": "explanation"}},
            "location": {{"info": "extracted info", "confidence": "High/Medium/Low", "reasoning": "explanation"}},
            "products": {{"info": "extracted info", "confidence": "High/Medium/Low", "reasoning": "explanation"}},
            "recent_news": {{"info": "extracted info", "confidence": "High/Medium/Low", "reasoning": "explanation"}},
            "technology": {{"info": "extracted info", "confidence": "High/Medium/Low", "reasoning": "explanation"}}
        }}

        Content to analyze:
        {content[:10000]}
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert business analyst with advanced reasoning capabilities. Always provide structured JSON responses with confidence levels and reasoning."
            },
            {
                "role": "user",
                "content": reasoning_prompt
            }
        ]
        
        return await self.azure_chat_completion(messages, temperature=0.2, max_tokens=3000)

    async def gather_insights(self, analyzed_data: List[Dict], company_name: str) -> str:
        """Gather and synthesize insights from all analyzed data"""
        gather_prompt = f"""
        You are a senior business intelligence analyst. You have received multiple analyses of {company_name} from different sources. 
        Your task is to synthesize this information into a comprehensive, accurate report.

        SYNTHESIS PROCESS:
        1. Evaluate confidence levels and prioritize high-confidence information
        2. Identify consistent information across sources
        3. Resolve conflicts by considering source reliability
        4. Fill gaps using logical inference
        5. Structure findings into a coherent narrative

        ANALYSIS DATA:
        {json.dumps(analyzed_data, indent=2)}

        Create a comprehensive report with these sections:

        1. EXECUTIVE SUMMARY (3-4 sentences)
        2. BUSINESS OVERVIEW
           - Core business model and value proposition
           - Primary products/services
           - Target market and industry position
        3. COMPANY DETAILS
           - Founding information
           - Location and headquarters
           - Team size and structure
        4. FINANCIAL OVERVIEW
           - Revenue information (if available)
           - Funding status
           - Growth indicators
        5. TECHNOLOGY & OPERATIONS
           - Technology stack
           - Operational capabilities
           - Innovation focus
        6. MARKET POSITION
           - Industry context
           - Competitive landscape
           - Recent developments
        7. CONFIDENCE ASSESSMENT
           - Overall confidence in findings
           - Data quality assessment
           - Information gaps identified

        Focus on accuracy and cite confidence levels where appropriate. If information is uncertain or conflicting, clearly indicate this.
        """

        messages = [
            {
                "role": "system",
                "content": "You are a senior business intelligence analyst specializing in synthesizing information from multiple sources to create accurate, comprehensive reports."
            },
            {
                "role": "user",
                "content": gather_prompt
            }
        ]
        
        return await self.azure_chat_completion(messages, temperature=0.3, max_tokens=4000)

    async def execute_parallel_searches(self, company_name: str) -> List[str]:
        """Execute multiple search queries in parallel - optimized for 5 links per query"""
        queries = self.generate_search_queries(company_name)
        logger.info(f"🔍 Executing {len(queries)} parallel searches for {company_name} (5 links per query)")
        
        tasks = []
        for i, query in enumerate(queries):
            delay = i * (self.delay_between_requests / len(queries))
            tasks.append(asyncio.create_task(self.delayed_search(query, delay)))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_urls = []
        
        for result in results:
            if isinstance(result, list):
                all_urls.extend(result)
            else:
                logger.error(f"Search failed: {result}")
        
        unique_urls = list(dict.fromkeys(all_urls))
        logger.info(f"📊 Collected {len(unique_urls)} unique URLs from parallel searches")
        
        return unique_urls

    async def delayed_search(self, query: str, delay: float) -> List[str]:
        """Execute search with delay"""
        await asyncio.sleep(delay)
        return await self.google_search(query)

    async def process_urls_parallel(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Process URLs in parallel with rate limiting"""
        logger.info(f"🔄 Processing {len(urls)} URLs in parallel")
        
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def process_with_semaphore(url):
            async with semaphore:
                result = await self.fetch_and_clean_page(url)
                await asyncio.sleep(self.delay_between_requests)
                return result
        
        tasks = [process_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, dict) and result:
                valid_results.append(result)
        
        logger.info(f"✅ Successfully processed {len(valid_results)}/{len(urls)} URLs")
        return valid_results

    async def generate_comprehensive_report(self, company_name: str) -> CompanyData:
        """Generate comprehensive company report with GPT reasoning and gathering"""
        print(f"\n🏢 Generating comprehensive report for: {company_name}")
        print("=" * 60)
        start_time = time.time()
        
        # Phase 1: Parallel web scraping (optimized for 5 links per query)
        print("🔍 Phase 1: Executing parallel web searches...")
        urls = await self.execute_parallel_searches(company_name)
        if not urls:
            print("❌ No URLs found from search queries")
            return CompanyData(name=company_name, timestamp=datetime.utcnow().isoformat())
        
        # Phase 2: Parallel content processing
        print("🔄 Phase 2: Processing web content...")
        page_data = await self.process_urls_parallel(urls)
        if not page_data:
            print("❌ No valid page data extracted")
            return CompanyData(name=company_name, timestamp=datetime.utcnow().isoformat())
        
        # Phase 3: GPT Reasoning Analysis
        print("🧠 Phase 3: Analyzing content with GPT reasoning...")
        reasoning_tasks = []
        for page in page_data:
            page_content = f"Title: {page['title']}\nDescription: {page['description']}\nContent: {page['content']}"
            reasoning_tasks.append(self.reasoning_analysis(page_content, company_name))
        
        reasoning_results = await asyncio.gather(*reasoning_tasks, return_exceptions=True)
        
        # Filter valid reasoning results
        valid_reasoning = []
        for result in reasoning_results:
            if isinstance(result, str) and result:
                try:
                    # Try to parse as JSON to validate
                    parsed = json.loads(result)
                    valid_reasoning.append(parsed)
                except:
                    # If not valid JSON, skip
                    continue
        
        # Phase 4: Gather insights using GPT
        print("📊 Phase 4: Gathering insights with GPT synthesis...")
        comprehensive_analysis = await self.gather_insights(valid_reasoning, company_name)
        
        # Phase 5: Aggregate data
        print("🔗 Phase 5: Aggregating company data...")
        company_data = CompanyData(
            name=company_name,
            website=page_data[0]['url'] if page_data else "",
            description=comprehensive_analysis,
            business_model=comprehensive_analysis,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Aggregate extracted information from reasoning
        for reasoning in valid_reasoning:
            if reasoning.get('company_verification', {}).get('confidence') == 'High':
                # Extract high-confidence information
                if reasoning.get('revenue', {}).get('confidence') in ['High', 'Medium']:
                    company_data.revenue_estimate = reasoning['revenue']['info']
                if reasoning.get('employees', {}).get('confidence') in ['High', 'Medium']:
                    company_data.employee_estimate = reasoning['employees']['info']
                if reasoning.get('founded', {}).get('confidence') in ['High', 'Medium']:
                    company_data.founded_year = reasoning['founded']['info']
                if reasoning.get('location', {}).get('confidence') in ['High', 'Medium']:
                    company_data.headquarters = reasoning['location']['info']
                if reasoning.get('industry', {}).get('confidence') in ['High', 'Medium']:
                    company_data.industry = reasoning['industry']['info']
        
        # Aggregate social links
        for page in page_data:
            company_data.social_links.update(page['social_links'])
        
        company_data.financial_data = {
            'reasoning_analysis': valid_reasoning,
            'synthesis_report': comprehensive_analysis
        }
        
        processing_time = time.time() - start_time
        print(f"✅ Report generated in {processing_time:.2f} seconds")
        
        return company_data

    def print_terminal_report(self, company_data: CompanyData):
        """Print comprehensive report to terminal"""
        print("\n" + "=" * 80)
        print(f"🏢 COMPREHENSIVE COMPANY INTELLIGENCE REPORT")
        print("=" * 80)
        print(f"📊 Company: {company_data.name}")
        print(f"🌐 Website: {company_data.website}")
        print(f"📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Comprehensive Analysis (from GPT synthesis)
        print("\n📋 COMPREHENSIVE ANALYSIS")
        print("-" * 40)
        if company_data.description:
            print(company_data.description)
        else:
            print("No comprehensive analysis available")
        
        # Company Details
        print("\n📈 COMPANY DETAILS")
        print("-" * 40)
        print(f"🏢 Industry: {company_data.industry or 'Not specified'}")
        print(f"🏢 Headquarters: {company_data.headquarters or 'Not specified'}")
        print(f"📅 Founded: {company_data.founded_year or 'Not specified'}")
        print(f"👥 Employees: {company_data.employee_estimate or 'Not specified'}")
        print(f"💰 Revenue: {company_data.revenue_estimate or 'Not specified'}")
        
        # Digital Presence
        print("\n🌐 DIGITAL PRESENCE")
        print("-" * 40)
        if company_data.social_links:
            for platform, url in company_data.social_links.items():
                print(f"{platform.title()}: {url}")
        else:
            print("No social media links found")
        
        # Data Quality Assessment
        print("\n📊 DATA QUALITY ASSESSMENT")
        print("-" * 40)
        if company_data.financial_data and company_data.financial_data.get('reasoning_analysis'):
            reasoning_data = company_data.financial_data['reasoning_analysis']
            high_confidence_count = sum(1 for item in reasoning_data 
                                      if item.get('company_verification', {}).get('confidence') == 'High')
            print(f"Sources analyzed: {len(reasoning_data)}")
            print(f"High-confidence sources: {high_confidence_count}")
            print(f"Confidence ratio: {high_confidence_count/len(reasoning_data)*100:.1f}%")
        else:
            print("No data quality assessment available")
        
        print("\n" + "=" * 80)
        print("📊 Report Complete")
        print("=" * 80)

# Main execution
async def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <company_name>")
        print("Example: python script.py 'Tesla'")
        sys.exit(1)
    
    company_name = sys.argv[1]
    generator = CompanyReportGenerator()
    
    try:
        report = await generator.generate_comprehensive_report(company_name)
        generator.print_terminal_report(report)
    except Exception as e:
        print(f"❌ Failed to generate report for {company_name}: {e}")
        logger.error(f"Report generation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())

    print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))
print("GOOGLE_CX:", os.getenv("GOOGLE_CX"))
