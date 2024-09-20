from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import os
import time
import requests

# Set Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Provide the path to the chromedriver executable
chrome_driver_path = r"C:\Users\jusin\Documents\GitHub\metascrape\figures\download\chromedriver-win64\chromedriver.exe"

# Create a Service object
chrome_service = Service(chrome_driver_path)

# Initialize the ChromeDriver with options and service
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

# Search queries for Google Images
search_queries = [
    'bar plot', 'area chart', 'box plot', 'bubble chart', 'flow chart',
    'line chart', 'map plot', 'network diagram', 'pareto chart', 'pie chart',
    'radar plot', 'scatter plot', 'tree diagram', 'venn diagram', 'pattern bar graph'
]

# Download function to fetch images using Selenium
def download_images(query):
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    driver.get(search_url)
    
    # Scroll to load more images (adjust range for more images)
    for _ in range(2):
        driver.execute_script("window.scrollBy(0,document.body.scrollHeight)")
        time.sleep(2)

    # Find image elements
    image_elements = driver.find_elements_by_css_selector("img.Q4LuWd")

    # Directory to save the images
    output_dir = os.path.join("download", "images", query.replace(" ", "_"))
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for img in image_elements:
        src = img.get_attribute("src")
        if src and "http" in src:
            try:
                img_data = requests.get(src).content
                with open(os.path.join(output_dir, f"{query.replace(' ', '_')}_{count}.jpg"), 'wb') as img_file:
                    img_file.write(img_data)
                count += 1
                if count >= 10:  # Limit the number of downloads per query (you can change this)
                    break
            except Exception as e:
                print(f"Could not download image {count} for query '{query}': {str(e)}")

for query in search_queries:
    download_images(query)
    print(f"Finished downloading images for query: {query}")

# Close the driver after downloading
driver.quit()
