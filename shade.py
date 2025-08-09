import json
import os
import time
import requests
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import random
import re
from urllib.parse import quote
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

    # Sanity check
print((None or "").strip())  # Won’t crash, outputs ''
print(("   something  " or "").strip())  # Outputs 'something'

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RawEmployee:
    """Data class for raw employee information from Google Search"""
    name: str
    linkedin_url: str
    snippet: str
    company: str

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
    financial_data: Dict[str, any] = None
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

class GoogleCSEEmployeeFinder:
    """Google Custom Search Engine integration for finding LinkedIn employee profiles"""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        if not self.cse_id:
            raise ValueError("GOOGLE_CSE_ID environment variable is required")
        
        # Session for making requests
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

    def find_employees(self, company_name: str, max_results: int = 50) -> List[RawEmployee]:
        """
        Find employees using Google Custom Search Engine
        
        Args:
            company_name: Name of the company to search for
            max_results: Maximum number of employees to find (up to 100 due to API limits)
            
        Returns:
            List of RawEmployee objects
        """
        logger.info(f"🔍 Starting Google CSE search for employees of: {company_name}")
        logger.info(f"📊 Target: {max_results} employee profiles")
        
        # Construct search query
        search_query = f'site:linkedin.com/in "{company_name}"'
        logger.info(f"🔎 Search query: {search_query}")
        
        employees = []
        start_index = 1
        results_per_page = 10  # Google CSE returns max 10 results per request
        
        # Calculate how many pages we need
        max_pages = min(10, (max_results + results_per_page - 1) // results_per_page)  # Max 100 results from Google CSE
        
        for page in range(max_pages):
            try:
                logger.info(f"📄 Fetching page {page + 1}/{max_pages} (results {start_index}-{min(start_index + 9, max_results)})")
                
                # Make API request
                page_results = self._search_page(search_query, start_index)
                
                if not page_results:
                    logger.warning(f"No results returned for page {page + 1}")
                    break
                
                # Process results
                for result in page_results:
                    if len(employees) >= max_results:
                        break
                        
                    employee = self._extract_employee_from_result(result, company_name)
                    if employee:
                        employees.append(employee)
                        logger.info(f"✅ Found: {employee.name} - {employee.snippet}")
                
                # Check if we have enough results
                if len(employees) >= max_results:
                    break
                
                # Rate limiting - be respectful to Google's API
                delay = random.uniform(2, 5)
                logger.info(f"⏱️ Rate limiting: waiting {delay:.1f} seconds...")
                time.sleep(delay)
                
                start_index += results_per_page
                
            except Exception as e:
                logger.error(f"Error fetching page {page + 1}: {e}")
                continue
        
        logger.info(f"🎯 Successfully found {len(employees)} employees for {company_name}")
        return employees

    def _search_page(self, query: str, start_index: int) -> List[Dict]:
        """
        Search a single page using Google CSE API
        
        Args:
            query: Search query
            start_index: Starting result index (1-based)
            
        Returns:
            List of search result dictionaries
        """
        params = {
            'key': self.api_key,
            'cx': self.cse_id,
            'q': query,
            'start': start_index,
            'num': 10,  # Maximum results per request
            'fields': 'items(title,link,snippet)'  # Only get fields we need
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'items' not in data:
                logger.warning("No 'items' field in API response")
                return []
            
            return data['items']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return []

    def _extract_employee_from_result(self, result: Dict, company_name: str) -> Optional[RawEmployee]:
        """
        Extract employee information from a Google search result
        
        Args:
            result: Google search result dictionary
            company_name: Company name for context
            
        Returns:
            RawEmployee object or None if extraction fails
        """
        try:
            # Extract LinkedIn URL
            linkedin_url = result.get('link', '')
            if not linkedin_url or 'linkedin.com/in/' not in linkedin_url:
                logger.warning(f"Invalid LinkedIn URL: {linkedin_url}")
                return None
            
            # Extract name from title
            title = result.get('title', '')
            name = self._extract_name_from_title(title)
            
            if not name:
                logger.warning(f"Could not extract name from title: {title}")
                return None
            
            # Get snippet
            snippet = result.get('snippet', '')
            
            # Clean up the snippet
            snippet = self._clean_snippet(snippet)
            
            return RawEmployee(
                name=name,
                linkedin_url=linkedin_url,
                snippet=snippet,
                company=company_name
            )
            
        except Exception as e:
            logger.warning(f"Error extracting employee from result: {e}")
            return None

    def _extract_name_from_title(self, title: str) -> str:
        """
        Extract name from LinkedIn page title
        
        Args:
            title: Page title from Google search result
            
        Returns:
            Extracted name or empty string
        """
        if not title:
            return ""
        
        # LinkedIn titles usually follow pattern "Name | Title | LinkedIn" or "Name - Title - LinkedIn"
        separators = ['|', '-', '–', '—']
        
        for separator in separators:
            if separator in title:
                parts = title.split(separator)
                name_part = parts[0].strip()
                
                # Basic validation - name should have at least 2 characters and no suspicious patterns
                if len(name_part) >= 2 and not any(word in name_part.lower() for word in ['linkedin', 'profile', 'www.']):
                    return name_part
        
        # Fallback: use the whole title if no separators found, but clean it up
        clean_title = title.replace('LinkedIn', '').replace('- LinkedIn', '').strip()
        if len(clean_title) >= 2:
            return clean_title
        
        return ""

    def _clean_snippet(self, snippet: str) -> str:
        """
        Clean up the snippet text
        
        Args:
            snippet: Raw snippet from Google search
            
        Returns:
            Cleaned snippet
        """
        if not snippet:
            return ""
        
        # Remove common unwanted phrases
        unwanted_phrases = [
            'View the profiles of professionals named',
            'View the profiles of people named',
            'There are ',
            ' professionals named',
            'on LinkedIn.',
            'LinkedIn is the world\'s largest professional network',
        ]
        
        cleaned = snippet
        for phrase in unwanted_phrases:
            cleaned = cleaned.replace(phrase, '')
        
        # Clean up extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()

class BrightDataScraper:
    """Bright Data integration for scraping LinkedIn profiles"""
    
    def __init__(self):
        self.api_key = os.getenv("BRIGHT_DATA_API_KEY", "8bda8a8ccf119c9ee2bf9d16591fb28cf591c7d3d7e382aec56ff567e7743da4")
        self.dataset_id = os.getenv("BRIGHT_DATA_DATASET_ID", "gd_l1viktl72bvl7bjuj0")
        
        self.trigger_url = f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={self.dataset_id}&include_errors=true"
        self.status_url = "https://api.brightdata.com/datasets/v3/progress/"
        self.result_url = "https://api.brightdata.com/datasets/v3/snapshot/"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def scrape_profiles_in_batches(self, urls: List[str], batch_size: int = 10) -> List[Dict]:
        """
        Scrape LinkedIn profiles in batches using Bright Data
        
        Args:
            urls: List of LinkedIn URLs to scrape
            batch_size: Number of profiles to scrape per batch
            
        Returns:
            List of scraped profile data
        """
        logger.info(f"🚀 Starting batch scraping of {len(urls)} profiles")
        logger.info(f"📦 Batch size: {batch_size}")
        
        all_profiles = []
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            logger.info(f"\n📦 Processing Batch {batch_num}: {len(batch)} profiles")
            
            # Trigger scrape for this batch
            snapshot_id = self._trigger_scrape(batch)
            if not snapshot_id:
                logger.error(f"❌ Failed to trigger batch {batch_num}")
                continue
            
            # Wait for completion
            if self._wait_until_ready(snapshot_id):
                batch_data = self._fetch_results(snapshot_id, batch_num)
                if batch_data:
                    all_profiles.extend(batch_data)
            else:
                logger.error(f"❌ Batch {batch_num} failed or timed out")
        
        logger.info(f"✅ Scraping completed. Total profiles scraped: {len(all_profiles)}")
        return all_profiles

    def _trigger_scrape(self, urls: List[str]) -> Optional[str]:
        """Trigger a scrape job for a batch of URLs"""
        payload = [{"url": url} for url in urls]
        
        try:
            response = requests.post(self.trigger_url, headers=self.headers, json=payload)
            logger.info(f"🚀 Trigger response: {response.status_code}")
            
            if response.ok:
                return response.json().get("snapshot_id")
            else:
                logger.error(f"Failed to trigger scrape: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error triggering scrape: {e}")
            return None

    def _wait_until_ready(self, snapshot_id: str, timeout: int = 600, interval: int = 10) -> bool:
        """Wait until snapshot is ready"""
        logger.info(f"⏳ Waiting for snapshot {snapshot_id} to complete...")
        
        for elapsed in range(0, timeout, interval):
            try:
                response = requests.get(self.status_url + snapshot_id, headers=self.headers)
                if response.ok:
                    status = response.json().get("status")
                    logger.info(f"⏳ {elapsed}s - Status: {status}")
                    
                    if status == "ready":
                        logger.info("✅ Snapshot ready!")
                        return True
                    elif status == "error":
                        logger.error("❌ Snapshot failed")
                        return False
                        
                time.sleep(interval)
                
            except Exception as e:
                logger.warning(f"Error checking status: {e}")
                time.sleep(interval)
        
        logger.error("❌ Timeout waiting for snapshot")
        return False

    def _fetch_results(self, snapshot_id: str, batch_num: int) -> List[Dict]:
        """Fetch results from completed snapshot"""
        result_url = self.result_url + snapshot_id
        
        try:
            response = requests.get(result_url, headers=self.headers, timeout=120)
            
            if response.ok:
                # Handle NDJSON (newline-delimited JSON)
                data = [json.loads(line) for line in response.text.strip().splitlines()]
                logger.info(f"✅ Fetched {len(data)} profiles from batch {batch_num}")
                return data
            else:
                logger.error(f"❌ Failed to fetch results for batch {batch_num}: {response.status_code}")
                logger.error(response.text)
                return []
                
        except Exception as e:
            logger.error(f"Error fetching results: {e}")
            return []

class CompanyReportGenerator:
    """Company research and analysis using web scraping and AI"""
    
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
        self.max_links_per_query = 5

    async def azure_chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4000) -> str:
        """Make Azure OpenAI API call with error handling"""
        if not self.azure_api_key or not self.azure_endpoint:
            logger.warning("Azure OpenAI not configured, skipping AI analysis")
            return "AI analysis not available - Azure OpenAI credentials not configured"
        
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
        """Perform Google Custom Search API call"""
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'key': self.google_api_key,
            'cx': self.google_cx,
            'num': self.max_links_per_query
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

    async def fetch_and_clean_page(self, url: str) -> Optional[Dict[str, any]]:
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
                            'content': content_text[:15000],
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

    async def execute_parallel_searches(self, company_name: str) -> List[str]:
        """Execute multiple search queries in parallel"""
        queries = self.generate_search_queries(company_name)
        logger.info(f"🔍 Executing {len(queries)} parallel searches for {company_name}")
        
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

    async def process_urls_parallel(self, urls: List[str]) -> List[Dict[str, any]]:
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

    async def generate_company_report(self, company_name: str) -> CompanyData:
        """Generate comprehensive company report"""
        logger.info(f"🏢 Generating company report for: {company_name}")
        
        # Execute parallel web scraping
        urls = await self.execute_parallel_searches(company_name)
        if not urls:
            logger.warning("No URLs found from search queries")
            return CompanyData(name=company_name, timestamp=datetime.utcnow().isoformat())
        
        # Process content
        page_data = await self.process_urls_parallel(urls)
        if not page_data:
            logger.warning("No valid page data extracted")
            return CompanyData(name=company_name, timestamp=datetime.utcnow().isoformat())
        
        # Create company data
        company_data = CompanyData(
            name=company_name,
            website=page_data[0]['url'] if page_data else "",
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Aggregate information
        all_content = []
        for page in page_data:
            all_content.append(f"Title: {page['title']}\nDescription: {page['description']}\nContent: {page['content'][:5000]}")
            company_data.social_links.update(page['social_links'])
            
            # Extract company info
            info = page['company_info']
            if info.get('revenue_estimate') and not company_data.revenue_estimate:
                company_data.revenue_estimate = info['revenue_estimate']
            if info.get('employee_estimate') and not company_data.employee_estimate:
                company_data.employee_estimate = info['employee_estimate']
            if info.get('founded_year') and not company_data.founded_year:
                company_data.founded_year = info['founded_year']
            if info.get('headquarters') and not company_data.headquarters:
                company_data.headquarters = info['headquarters']
        
        # Create summary description
        combined_content = '\n\n'.join(all_content[:3])  # Use first 3 pages
        company_data.description = f"Company research based on web data analysis. Found {len(page_data)} relevant sources."
        
        # Use AI for analysis if available
        if self.azure_api_key and self.azure_endpoint:
            try:
                analysis_prompt = f"""
                Analyze the following content about {company_name} and provide a comprehensive business summary.
                Include: business model, industry, key products/services, market position, and any financial information.
                Be concise but informative.
                
                Content:
                {combined_content[:8000]}
                """
                
                messages = [
                    {"role": "system", "content": "You are a business analyst. Provide clear, factual analysis."},
                    {"role": "user", "content": analysis_prompt}
                ]
                
                ai_analysis = await self.azure_chat_completion(messages, temperature=0.3, max_tokens=2000)
                if ai_analysis:
                    company_data.description = ai_analysis
                    company_data.business_model = ai_analysis
                    
            except Exception as e:
                logger.error(f"AI analysis failed: {e}")
        
        logger.info(f"✅ Company report generated for {company_name}")
        return company_data

class ComprehensiveDataManager:
    """Class to handle all data (company + employees) in a single comprehensive format"""
    
    def __init__(self):
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def save_comprehensive_data(self, company_name: str, raw_employees: List[RawEmployee], 
                              detailed_profiles: List[Dict] = None, company_data: CompanyData = None) -> str:
        """
        Save all data (company + employees) to a single comprehensive JSON file
        
        Args:
            company_name: Name of the company
            raw_employees: List of RawEmployee objects from Google search
            detailed_profiles: List of detailed profile data from Bright Data (optional)
            company_data: CompanyData object with company research (optional)
            
        Returns:
            Path to the saved file
        """
        # Create safe filename
        safe_company_name = re.sub(r'[^a-zA-Z0-9_-]', '_', company_name.lower())
        filename = f"{safe_company_name}_complete_intelligence_report.json"
        filepath = os.path.join(self.data_dir, filename)
        
        # Create comprehensive data structure
        comprehensive_data = {
            "Spectre_company": {
                "company_name": company_name,
                "report_type": "Complete Intelligence Report",
                "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "data_sources": ["Google Custom Search Engine"],
                "report_sections": ["company_intelligence", "employee_intelligence", "analytics"]
            },
            "company_intelligence": {},
            "employee_intelligence": {
                "summary": {
                    "total_employees_found": len(raw_employees),
                    "detailed_profiles_scraped": len(detailed_profiles) if detailed_profiles else 0,
                    "scraping_success_rate": 0,
                    "scraping_completed": bool(detailed_profiles)
                },
                "employees": []
            },
            "analytics": {},
            "executive_summary": ""
        }
        
        # Add data sources
        if detailed_profiles:
            comprehensive_data["report_metadata"]["data_sources"].append("Bright Data LinkedIn Scraper")
        if company_data:
            comprehensive_data["report_metadata"]["data_sources"].append("Company Web Research")
            if company_data.financial_data:
                comprehensive_data["report_metadata"]["data_sources"].append("AI Analysis")
        
        # Calculate success rate
        if detailed_profiles and raw_employees:
            success_rate = len(detailed_profiles) / len(raw_employees) * 100
            comprehensive_data["employee_intelligence"]["summary"]["scraping_success_rate"] = round(success_rate, 2)
        
        # Add company intelligence
        if company_data:
            comprehensive_data["company_intelligence"] = {
                "basic_info": {
                    "name": company_data.name,
                    "website": company_data.website,
                    "industry": company_data.industry,
                    "headquarters": company_data.headquarters,
                    "founded_year": company_data.founded_year,
                    "employee_estimate": company_data.employee_estimate,
                    "revenue_estimate": company_data.revenue_estimate
                },
                "business_analysis": {
                    "description": company_data.description,
                    "business_model": company_data.business_model,
                    "key_products": company_data.key_products,
                    "market_position": company_data.market_position
                },
                "digital_presence": {
                    "social_links": company_data.social_links,
                    "tech_stack": company_data.tech_stack
                },
                "financial_data": company_data.financial_data,
                "recent_news": company_data.recent_news
            }
        else:
            comprehensive_data["company_intelligence"] = {
                "basic_info": {
                    "name": company_name,
                    "note": "Company research not available - missing Azure OpenAI or Google CX configuration"
                }
            }
        
        # Create URL to detailed profile mapping for faster lookup
        detailed_profiles_map = {}
        if detailed_profiles:
            for profile in detailed_profiles:
                url = profile.get('url', '')
                if url:
                    detailed_profiles_map[url] = profile
        
        # Combine raw and detailed data for each employee
        for raw_emp in raw_employees:
            employee_data = {
                "basic_info": {
                    "name": raw_emp.name,
                    "linkedin_url": raw_emp.linkedin_url,
                    "company": raw_emp.company,
                    "search_snippet": raw_emp.snippet
                },
                "detailed_profile": None,
                "data_status": {
                    "found_in_search": True,
                    "detailed_scraped": False,
                    "scraping_error": None
                },
                "summary": {}
            }
            
            # Add detailed profile data if available
            if raw_emp.linkedin_url in detailed_profiles_map:
                detailed_profile = detailed_profiles_map[raw_emp.linkedin_url]
                employee_data["detailed_profile"] = detailed_profile
                employee_data["data_status"]["detailed_scraped"] = True
                
                # Extract key information for easy access
                employee_data["summary"] = self._extract_profile_summary(detailed_profile)
            else:
                employee_data["summary"] = {
                    "full_name": raw_emp.name,
                    "current_position": "Not available",
                    "location": "Not available",
                    "experience_years": "Not available",
                    "skills_count": 0,
                    "education_count": 0,
                    "connections": "Not available"
                }
                
                if detailed_profiles:  # Only mark as error if scraping was attempted
                    employee_data["data_status"]["scraping_error"] = "Profile not found in scraped data"
            
            comprehensive_data["employee_intelligence"]["employees"].append(employee_data)
        
        # Add analytics/summary section
        comprehensive_data["analytics"] = self._generate_comprehensive_analytics(
            comprehensive_data["employee_intelligence"]["employees"], 
            company_data
        )
        
        # Generate executive summary
        comprehensive_data["executive_summary"] = self._generate_executive_summary(
            company_name, 
            comprehensive_data["company_intelligence"], 
            comprehensive_data["employee_intelligence"]["summary"],
            comprehensive_data["analytics"]
        )
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Comprehensive intelligence report saved to: {filepath}")
        return filepath
    
    def _extract_profile_summary(self, profile: Dict) -> Dict:
        """Extract key summary information from detailed profile"""
        summary = {
            "full_name": str(profile.get("name") or "").strip(),
            "current_position": "",
            "location": str(profile.get("location") or "").strip(),
            "experience_years": 0,
            "skills_count": 0,
            "education_count": 0,
            "connections": str(profile.get("connections") or "").strip()
        }
        
        # Extract current position from experience
        experiences = profile.get("experience", [])
        if experiences and isinstance(experiences, list):
            current_exp = experiences[0] if experiences else {}
            summary["current_position"] = str(current_exp.get("title", "")).strip()
            
            # Calculate total experience years (rough estimate)
            total_months = 0
            for exp in experiences:
                duration = str(exp.get("duration", ""))
                # Basic parsing of duration strings like "2 yrs 3 mos"
                years = re.findall(r'(\d+)\s*yr', duration)
                months = re.findall(r'(\d+)\s*mo', duration)
                
                exp_months = 0
                if years:
                    exp_months += int(years[0]) * 12
                if months:
                    exp_months += int(months[0])
                
                total_months += exp_months
            
            summary["experience_years"] = round(total_months / 12, 1) if total_months > 0 else 0
        
        # Count skills and education
        skills = profile.get("skills", [])
        summary["skills_count"] = len(skills) if isinstance(skills, list) else 0
        
        education = profile.get("education", [])
        summary["education_count"] = len(education) if isinstance(education, list) else 0
        
        return summary
    
    def _generate_comprehensive_analytics(self, employees: List[Dict], company_data: CompanyData = None) -> Dict:
        """Generate comprehensive analytics from all data"""
        total_employees = len(employees)
        scraped_count = sum(1 for emp in employees if emp["data_status"]["detailed_scraped"])
        
        analytics = {
            "employee_analytics": {
                "totals": {
                    "employees_found": total_employees,
                    "profiles_scraped": scraped_count,
                    "scraping_success_rate": round((scraped_count / total_employees * 100), 2) if total_employees > 0 else 0
                },
                "top_positions": [],
                "top_skills": [],
                "top_locations": [],
                "experience_distribution": {
                    "0-2_years": 0,
                    "3-5_years": 0,
                    "6-10_years": 0,
                    "11+_years": 0
                }
            },
            "company_analytics": {},
            "data_quality": {
                "sources_analyzed": 0,
                "confidence_score": 0,
                "completeness_score": 0
            }
        }
        
        # Collect data for employee analytics
        positions = []
        skills = []
        locations = []
        
        for emp in employees:
            if emp["data_status"]["detailed_scraped"]:
                summary = emp.get("summary", {})
                
                # Collect positions
                position = summary.get("current_position", "").strip()
                if position:
                    positions.append(position)
                
                # Collect locations
                location = summary.get("location", "").strip()
                if location:
                    locations.append(location)
                
                # Experience distribution
                exp_years = summary.get("experience_years", 0)
                if isinstance(exp_years, (int, float)):
                    if exp_years <= 2:
                        analytics["employee_analytics"]["experience_distribution"]["0-2_years"] += 1
                    elif exp_years <= 5:
                        analytics["employee_analytics"]["experience_distribution"]["3-5_years"] += 1
                    elif exp_years <= 10:
                        analytics["employee_analytics"]["experience_distribution"]["6-10_years"] += 1
                    else:
                        analytics["employee_analytics"]["experience_distribution"]["11+_years"] += 1
                
                # Collect skills from detailed profile
                if emp["detailed_profile"]:
                    profile_skills = emp["detailed_profile"].get("skills", [])
                    if isinstance(profile_skills, list):
                        skills.extend([skill.get("name", "") for skill in profile_skills if isinstance(skill, dict)])
        
        # Generate top lists using Counter
        from collections import Counter
        
        analytics["employee_analytics"]["top_positions"] = [{"position": pos, "count": count} 
                                                          for pos, count in Counter(positions).most_common(10)]
        
        analytics["employee_analytics"]["top_skills"] = [{"skill": skill, "count": count} 
                                                       for skill, count in Counter(skills).most_common(20)]
        
        analytics["employee_analytics"]["top_locations"] = [{"location": loc, "count": count} 
                                                          for loc, count in Counter(locations).most_common(10)]
        
        # Company analytics
        if company_data:
            analytics["company_analytics"] = {
                "web_presence": {
                    "social_platforms": len(company_data.social_links),
                    "platforms": list(company_data.social_links.keys())
                },
                "data_richness": {
                    "has_revenue_data": bool(company_data.revenue_estimate),
                    "has_employee_estimate": bool(company_data.employee_estimate),
                    "has_founding_info": bool(company_data.founded_year),
                    "has_location_info": bool(company_data.headquarters),
                    "has_business_description": bool(company_data.description)
                }
            }
        
        # Data quality assessment
        analytics["data_quality"]["completeness_score"] = self._calculate_completeness_score(employees, company_data)
        analytics["data_quality"]["confidence_score"] = analytics["employee_analytics"]["totals"]["scraping_success_rate"]
        
        return analytics
    
    def _calculate_completeness_score(self, employees: List[Dict], company_data: CompanyData = None) -> float:
        """Calculate overall data completeness score"""
        total_score = 0
        max_score = 0
        
        # Employee data completeness (60% of total score)
        if employees:
            scraped_employees = [emp for emp in employees if emp["data_status"]["detailed_scraped"]]
            employee_completeness = len(scraped_employees) / len(employees) * 60
            total_score += employee_completeness
        max_score += 60
        
        # Company data completeness (40% of total score)
        if company_data:
            company_fields = [
                company_data.description,
                company_data.industry,
                company_data.headquarters,
                company_data.founded_year,
                company_data.employee_estimate,
                company_data.revenue_estimate,
                company_data.social_links,
                company_data.business_model
            ]
            filled_fields = sum(1 for field in company_fields if field)
            company_completeness = (filled_fields / len(company_fields)) * 40
            total_score += company_completeness
        max_score += 40
        
        return round((total_score / max_score) * 100, 2) if max_score > 0 else 0
    
    def _generate_executive_summary(self, company_name: str, company_intel: Dict, 
                                  employee_summary: Dict, analytics: Dict) -> str:
        """Generate executive summary of the intelligence report"""
        summary_parts = []
        
        # Company overview
        if company_intel.get("basic_info", {}).get("name"):
            company_info = company_intel["basic_info"]
            summary_parts.append(f"Company Overview: {company_name}")
            
            if company_info.get("industry"):
                summary_parts.append(f"operates in the {company_info['industry']} industry")
            
            if company_info.get("headquarters"):
                summary_parts.append(f"headquartered in {company_info['headquarters']}")
            
            if company_info.get("founded_year"):
                summary_parts.append(f"founded in {company_info['founded_year']}")
        
        # Employee intelligence summary
        emp_found = employee_summary.get("total_employees_found", 0)
        emp_scraped = employee_summary.get("detailed_profiles_scraped", 0)
        success_rate = employee_summary.get("scraping_success_rate", 0)
        
        summary_parts.append(f"Employee Intelligence: Found {emp_found} employees on LinkedIn, successfully scraped detailed profiles for {emp_scraped} ({success_rate}% success rate)")
        
        # Top insights
        top_positions = analytics.get("employee_analytics", {}).get("top_positions", [])
        if top_positions:
            top_pos = top_positions[0]["position"]
            summary_parts.append(f"Most common position: {top_pos}")
        
        # Data quality
        completeness = analytics.get("data_quality", {}).get("completeness_score", 0)
        summary_parts.append(f"Overall data completeness: {completeness}%")
        
        return ". ".join(summary_parts) + "."

# Parallel execution function
async def run_parallel_intelligence_gathering(company_name: str, max_employees: int = 50):
    """Run company research and employee finding in parallel"""
    logger.info(f"🚀 Starting parallel intelligence gathering for {company_name}")
    
    # Initialize all components
    employee_finder = GoogleCSEEmployeeFinder()
    company_researcher = CompanyReportGenerator()
    
    # Create tasks for parallel execution
    tasks = []
    
    # Task 1: Find employees
    logger.info("📋 Task 1: Finding employees...")
    employee_task = asyncio.create_task(
        asyncio.to_thread(employee_finder.find_employees, company_name, max_employees)
    )
    tasks.append(("employees", employee_task))
    
    # Task 2: Research company (only if properly configured)
    if (os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID")):
        logger.info("📋 Task 2: Researching company...")
        company_task = asyncio.create_task(
            company_researcher.generate_company_report(company_name)
        )
        tasks.append(("company", company_task))
    else:
        logger.warning("⚠️ Skipping company research - missing Google CX configuration")
        tasks.append(("company", None))
    
    # Execute tasks in parallel
    results = {}
    for task_name, task in tasks:
        if task:
            try:
                result = await task
                results[task_name] = result
                logger.info(f"✅ Completed: {task_name}")
            except Exception as e:
                logger.error(f"❌ Failed: {task_name} - {e}")
                results[task_name] = None
        else:
            results[task_name] = None
    
    return results["employees"], results["company"]

def main():
    """Main function to run the complete intelligence gathering pipeline"""
    
    print("🚀 Complete Company Intelligence Pipeline")
    print("📊 Company Research + Employee Intelligence")
    print("=" * 70)
    
    # Check environment variables
    required_env_vars = [
        ("GOOGLE_API_KEY", "Get your API key from: https://console.developers.google.com/"),
        ("GOOGLE_CSE_ID", "Create a Custom Search Engine at: https://cse.google.com/")
    ]
    
    missing_vars = []
    for var_name, help_text in required_env_vars:
        if not os.getenv(var_name):
            missing_vars.append((var_name, help_text))
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var_name, help_text in missing_vars:
            print(f"   {var_name}: {help_text}")
        return
    
    # Optional environment variables
    optional_vars = [
        ("GOOGLE_CSE_ID", "Custom Search Engine ID for company research"),
        ("AZURE_OPENAI_API_KEY", "Azure OpenAI API key for AI analysis"),
        ("AZURE_OPENAI_ENDPOINT", "Azure OpenAI endpoint URL"),
        ("BRIGHT_DATA_API_KEY", "Bright Data API key for LinkedIn scraping")
    ]
    
    print("📋 Configuration Status:")
    for var_name, description in optional_vars:
        status = "✅ Configured" if os.getenv(var_name) else "⚠️ Not configured"
        print(f"   {var_name}: {status}")
    print()
    
    # Get user input
    try:
        company_name = input("Enter company name: ").strip()
        if not company_name:
            raise ValueError("Company name is required")
        
        max_results_input = input("Enter maximum number of employees to find (default 50, max 100): ").strip()
        max_results = int(max_results_input) if max_results_input else 50
        max_results = min(max_results, 100)  # Google CSE API limit
        
        print(f"\n🔄 Starting comprehensive intelligence gathering...")
        print(f"   Company: {company_name}")
        print(f"   Max employees: {max_results}")
        print(f"   Will run company research and employee finding in parallel")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
        return
    
    async def run_pipeline():
        try:
            start_time = time.time()
            
            # Phase 1: Parallel intelligence gathering
            print(f"\n🔍 Phase 1: Parallel Intelligence Gathering...")
            employees, company_data = await run_parallel_intelligence_gathering(company_name, max_results)
            
            if not employees:
                print("❌ No employees found. Try adjusting your search query or company name.")
                return
            
            print(f"✅ Found {len(employees)} employees")
            if company_data:
                print(f"✅ Company research completed")
            else:
                print(f"⚠️ Company research skipped or failed")
            
            # Phase 2: Scrape LinkedIn profiles
            print(f"\n🔧 Phase 2: Scraping LinkedIn profiles...")
            scraper = BrightDataScraper()
            linkedin_urls = [emp.linkedin_url for emp in employees]
            detailed_profiles = scraper.scrape_profiles_in_batches(linkedin_urls)
            
            # Phase 3: Create comprehensive report
            print(f"\n📋 Phase 3: Creating comprehensive intelligence report...")
            data_manager = ComprehensiveDataManager()
            output_file = data_manager.save_comprehensive_data(
                company_name, employees, detailed_profiles, company_data
            )
            
            # Final summary
            processing_time = time.time() - start_time
            print(f"\n🎉 Intelligence gathering completed in {processing_time:.1f} seconds!")
            print(f"📊 Final Results:")
            print(f"   • Company: {company_name}")
            print(f"   • Company research: {'✅ Complete' if company_data else '⚠️ Skipped'}")
            print(f"   • Employees found: {len(employees)}")
            print(f"   • Detailed profiles scraped: {len(detailed_profiles) if detailed_profiles else 0}")
            if detailed_profiles and employees:
                success_rate = len(detailed_profiles) / len(employees) * 100
                print(f"   • Scraping success rate: {success_rate:.1f}%")
            print(f"📁 Complete intelligence report: {output_file}")
            print(f"\n💡 The report contains:")
            print(f"   • Executive summary and analytics")
            print(f"   • Complete company intelligence")
            print(f"   • Employee profiles and insights")
            print(f"   • Data quality assessment")
            
        except Exception as e:
            logger.error(f"Error in pipeline execution: {e}")
            print(f"❌ Error: {e}")
    
    # Run the async pipeline
    asyncio.run(run_pipeline())

if __name__ == "__main__":
    main()

