import time
import json
import os
import random
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from dotenv import load_dotenv

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)

# ========== LOAD ENV ==========
load_dotenv()
LINKEDIN_EMAIL = os.getenv("LINKEDIN_EMAIL")
LINKEDIN_PASSWORD = os.getenv("LINKEDIN_PASSWORD")

PROFILE_DIR = os.path.abspath("linkedin_browser_profile")
OUTPUT_DIR = "profiles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def human_delay(min_sec=1, max_sec=3):
    delay = random.uniform(min_sec, max_sec)
    if random.random() < 0.1:
        delay += random.uniform(2, 8)
    time.sleep(delay)

def micro_delay():
    time.sleep(random.uniform(0.1, 0.5))

def reading_delay():
    time.sleep(random.uniform(3, 12))

def overnight_batch_delay():
    delay = random.randint(3600, 5400)  # 1–1.5 hours
    print(f"\n🌙 Night break: {delay // 60} minutes...")
    time.sleep(delay)

# ========== SELENIUM SETUP ==========
def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument(f"--user-data-dir={PROFILE_DIR}")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    viewports = [
        "--window-size=1366,768", "--window-size=1920,1080",
        "--window-size=1536,864", "--window-size=1440,900"
    ]
    options.add_argument(random.choice(viewports))

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ]
    options.add_argument(f"--user-agent={random.choice(user_agents)}")

    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def human_like_typing(element, text):
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(0.02, 0.08))
        if random.random() < 0.02:
            element.send_keys(Keys.BACKSPACE)
            time.sleep(0.2)
            element.send_keys(char)

def login_if_needed(driver):
    driver.get("https://www.linkedin.com/login")
    human_delay(2, 4)
    if "login" in driver.current_url:
        print("🔐 Logging in...")
        email_input = driver.find_element(By.ID, "username")
        ActionChains(driver).move_to_element(email_input).click().perform()
        micro_delay()
        human_like_typing(email_input, LINKEDIN_EMAIL)

        human_delay(1, 3)
        password_input = driver.find_element(By.ID, "password")
        ActionChains(driver).move_to_element(password_input).click().perform()
        micro_delay()
        human_like_typing(password_input, LINKEDIN_PASSWORD)

        human_delay(2, 4)
        login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
        ActionChains(driver).move_to_element(login_button).pause(random.uniform(0.5, 1.5)).click().perform()
        human_delay(8, 15)

def smart_scroll_and_load(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(10):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(0.8, 1.5))
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(random.uniform(0.5, 1.0))

def expand_sections(driver):
    selectors = [
        '//button[contains(text(), "See more")]',
        '//button[contains(text(), "Show more")]',
        '//button[@aria-label="See more"]'
    ]
    for selector in selectors:
        try:
            buttons = driver.find_elements(By.XPATH, selector)
            for btn in buttons[:2]:
                if btn.is_displayed():
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
                    time.sleep(random.uniform(0.3, 0.8))
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(random.uniform(0.5, 1.2))
        except:
            continue

def get_profile_content(driver, url, is_skills=False):
    driver.get(url)
    time.sleep(random.uniform(3, 5) if not is_skills else random.uniform(2, 4))
    smart_scroll_and_load(driver)
    if not is_skills:
        expand_sections(driver)
        time.sleep(random.uniform(2, 3))
    else:
        time.sleep(random.uniform(1, 2))
    return driver.find_element(By.TAG_NAME, "body").text

def scrape_profile(driver, url):
    print(f"🔄 Visiting: {url}")
    try:
        if random.random() < 0.08:
            time.sleep(random.randint(15, 45))
        main = get_profile_content(driver, url, False)
        time.sleep(random.uniform(1, 2))
        skills_url = url.rstrip("/") + "/details/skills"
        print("  📋 Getting skills...")
        skills = get_profile_content(driver, skills_url, True)
        return {
            "url": url,
            "raw_text": main,
            "skills_text": skills,
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "has_skills": bool(skills.strip())
        }
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return None

def save_profile(data, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {filename}")

def read_profiles(file_path="profiles.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    random.shuffle(urls)
    return urls

def run_batch(driver, urls, start_idx, batch_size):
    end_idx = min(start_idx + batch_size, len(urls))
    for idx in range(start_idx, end_idx):
        url = urls[idx]
        profile_id = sanitize_filename(url.rstrip("/").split("/")[-1])
        filename = f"{idx+1:03d}_{profile_id}.json"
        if os.path.exists(os.path.join(OUTPUT_DIR, filename)):
            print(f"📁 Skipping: {filename}")
            continue
        data = scrape_profile(driver, url)
        if data:
            save_profile(data, filename)
        else:
            print(f"⚠️  Failed to scrape profile {idx+1}")
        if idx < end_idx - 1:
            human_delay(2, 6)
    return end_idx

# ========== MAIN ==========
if __name__ == "__main__":
    print("🌙 Starting safe LinkedIn scraper (no proxy)...")
    urls = read_profiles()
    total = len(urls)
    batch_size = random.randint(10, 15)
    index = 0
    print(f"📊 Total profiles: {total} | 📦 Batch size: {batch_size}")
    driver = setup_driver()
    try:
        login_if_needed(driver)
        while index < total:
            batch_end = run_batch(driver, urls, index, batch_size)
            index = batch_end
            if index < total:
                print(f"✅ Batch done. {total - index} profiles left.")
                overnight_batch_delay()
                if random.random() < 0.3:
                    batch_size = random.randint(10, 15)
                    print(f"🔄 New batch size: {batch_size}")
    except KeyboardInterrupt:
        print("\n⏹️ Scraper manually stopped.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        driver.quit()
        print("\n🌅 Done. Check 'profiles' folder for data.")
