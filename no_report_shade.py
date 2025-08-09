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
        batch_files = []
        
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
                    # Save batch file for backup
                    batch_file = f"batch_{batch_num}.json"
                    batch_files.append(batch_file)
                    self._save_batch_file(batch_data, batch_file)
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

    def _save_batch_file(self, data: List[Dict], filename: str):
        """Save batch data to file as backup"""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Batch saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving batch file: {e}")

class EmployeeDataSaver:
    """Class to handle saving employee data to JSON files"""
    
    def __init__(self):
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def save_raw_employees(self, company_name: str, employees: List[RawEmployee]) -> str:
        """
        Save raw employee data to JSON file
        
        Args:
            company_name: Name of the company
            employees: List of RawEmployee objects
            
        Returns:
            Path to the saved file
        """
        # Create safe filename
        safe_company_name = re.sub(r'[^a-zA-Z0-9_-]', '_', company_name.lower())
        filename = f"{safe_company_name}_employees_raw.json"
        filepath = os.path.join(self.data_dir, filename)
        
        # Prepare data for JSON serialization
        data = {
            "company": company_name,
            "total_employees": len(employees),
            "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": "Google Custom Search Engine",
            "employees": []
        }
        
        for emp in employees:
            employee_data = {
                "name": emp.name,
                "linkedin_url": emp.linkedin_url,
                "snippet": emp.snippet,
                "company": emp.company
            }
            data["employees"].append(employee_data)
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Raw employee data saved to: {filepath}")
        return filepath

    def save_scraped_employees(self, company_name: str, employees: List[Dict]) -> str:
        """
        Save scraped employee data to JSON file
        
        Args:
            company_name: Name of the company
            employees: List of scraped employee data dictionaries
            
        Returns:
            Path to the saved file
        """
        # Create safe filename
        safe_company_name = re.sub(r'[^a-zA-Z0-9_-]', '_', company_name.lower())
        filename = f"{safe_company_name}_employees_detailed.json"
        filepath = os.path.join(self.data_dir, filename)
        
        # Prepare data for JSON serialization
        data = {
            "company": company_name,
            "total_employees": len(employees),
            "scraping_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": "Bright Data LinkedIn Scraper",
            "employees": employees
        }
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Detailed employee data saved to: {filepath}")
        return filepath

    def save_urls_to_file(self, urls: List[str], filename: str = "profiles.txt") -> str:
        """Save LinkedIn URLs to text file for manual review or processing"""
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for url in urls:
                f.write(url + '\n')
        
        logger.info(f"📝 LinkedIn URLs saved to: {filepath}")
        return filepath

def main():
    """Main function to run the complete LinkedIn employee discovery and scraping pipeline"""
    
    print("🚀 LinkedIn Employee Discovery & Scraping Pipeline")
    print("=" * 60)
    
    # Check environment variables
    required_env_vars = [
        ("GOOGLE_API_KEY", "Get your API key from: https://console.developers.google.com/"),
        ("GOOGLE_CSE_ID", "Create a Custom Search Engine at: https://cse.google.com/")
    ]
    
    for var_name, help_text in required_env_vars:
        if not os.getenv(var_name):
            print(f"❌ Error: {var_name} environment variable is required")
            print(f"   {help_text}")
            return
    
    # Get user input
    try:
        company_name = input("Enter company name: ").strip()
        if not company_name:
            raise ValueError("Company name is required")
        
        max_results_input = input("Enter maximum number of employees to find (default 50, max 100): ").strip()
        max_results = int(max_results_input) if max_results_input else 50
        max_results = min(max_results, 100)  # Google CSE API limit
        
        scrape_profiles = input("Do you want to scrape detailed profiles using Bright Data? (y/n, default n): ").strip().lower()
        scrape_profiles = scrape_profiles in ['y', 'yes', 'true', '1']
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
        return
    
    try:
        # Initialize components
        finder = GoogleCSEEmployeeFinder()
        saver = EmployeeDataSaver()
        
        # Stage 1: Find employees using Google CSE
        print(f"\n🔍 Stage 1: Finding LinkedIn profiles for {company_name}...")
        employees = finder.find_employees(company_name, max_results)
        
        if not employees:
            print("❌ No employees found. Try adjusting your search query or company name.")
            return
        
        # Save raw results
        raw_output_file = saver.save_raw_employees(company_name, employees)
        
        # Extract LinkedIn URLs
        linkedin_urls = [emp.linkedin_url for emp in employees]
        urls_file = saver.save_urls_to_file(linkedin_urls)
        
        # Display Stage 1 summary
        print(f"\n✅ Stage 1 Completed!")
        print(f"📊 Found {len(employees)} employees for {company_name}")
        print(f"📁 Raw data saved to: {raw_output_file}")
        print(f"📝 LinkedIn URLs saved to: {urls_file}")
        
        # Show sample results
        print("\n📋 Sample results:")
        for i, emp in enumerate(employees[:3]):
            print(f"  {i+1}. {emp.name}")
            print(f"     URL: {emp.linkedin_url}")
            print(f"     Snippet: {emp.snippet[:100]}...")
            print()
        
        if len(employees) > 3:
            print(f"  ... and {len(employees) - 3} more employees")
        
        # Stage 2: Scrape detailed profiles (optional)
        if scrape_profiles:
            print(f"\n🔧 Stage 2: Scraping detailed profiles...")
            
            scraper = BrightDataScraper()
            detailed_profiles = scraper.scrape_profiles_in_batches(linkedin_urls)
            
            if detailed_profiles:
                # Save detailed results
                detailed_output_file = saver.save_scraped_employees(company_name, detailed_profiles)
                
                print(f"\n✅ Stage 2 Completed!")
                print(f"📊 Scraped {len(detailed_profiles)} detailed profiles")
                print(f"📁 Detailed data saved to: {detailed_output_file}")
            else:
                print("\n❌ Stage 2 failed - no profiles were successfully scraped")
        else:
            print(f"\n⏭️ Skipping Stage 2 (profile scraping)")
            print(f"   You can run profile scraping later using the saved URLs in: {urls_file}")
        
        print(f"\n🎉 Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()