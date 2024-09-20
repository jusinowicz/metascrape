"""
Download training images from google. 
Because we want to download so many images, the chromedriver needs to be installed.
For instructions see https://google-images-download.readthedocs.io/en/latest/troubleshooting.html#installing-the-chromedriver-with-selenium


"""

from google_images_download import google_images_download
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Set Chrome options
chrome_options = Options()

# Provide the path to the chromedriver executable
chrome_driver_path = r"C:\Users\jusin\Documents\GitHub\metascrape\figures\download\chromedriver-win64\chromedriver.exe"

# Create a Service object to avoid deprecation warnings
chrome_service = Service(chrome_driver_path)

# Initialize the ChromeDriver with options and service
driver = webdriver.Chrome(service=chrome_service, options=chrome_options) 

response = google_images_download.googleimagesdownload() 
search_queries = ['bar plot', 'area chart', 'box plot', 'bubble chart', 'flow chart',
'line chart', 'map plot', 'network diagram', 'pareto chart', 'pie chart', 'radar plot',
'scatter plot', 'tree diagram', 'venn diagram', 'pattern bar graph'] 

def downloadimages(query): 
        chromedriver = chrome_driver_path 
        output_dir = "./download/images"
        arguments = {"keywords": query, 
                    "limit": 2000, 
                    "print_urls": True,
                    "size": "medium",
                    "output_directory": output_dir,
                    "chromedriver": chromedriver}
        try:
            response.download(arguments)
        except:
            pass

for query in search_queries:
    downloadimages(query)
    print()


from google_images_download import google_images_download 

# Provide the path to the chromedriver executable
chrome_driver_path = r"C:\Users\jusin\Documents\GitHub\metascrape\figures\download\chromedriver-win64\chromedriver.exe"

response = google_images_download.googleimagesdownload() 
search_queries = ['bar plot', 'area chart', 'box plot', 'bubble chart', 'flow chart',
'line chart', 'map plot', 'network diagram', 'pareto chart', 'pie chart', 'radar plot',
'scatter plot', 'tree diagram', 'venn diagram', 'pattern bar graph'] 

def downloadimages(query): 
        chromedriver = r"C:\ProgramData\chocolatey\lib\chromedriver\tools\chromedriver.exe"
        output_dir = r"D:\Summer2020\PSUProject\download\images"
        arguments = {"keywords": query, 
                     "limit": 2000, 
                     "print_urls": True, 
                     "size": "medium",
                     "output_directory": output_dir
                     }
        try:
            response.download(arguments, --chromedriver = chrome_driver_path) 
        except:
            pass

for query in search_queries: 
    downloadimages(query) 
    print()