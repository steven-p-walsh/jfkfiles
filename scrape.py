import requests
from bs4 import BeautifulSoup
import os
import urllib.parse

# Define headers with a user-agent to mimic a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Define the base URL and download directory
base_url = "https://www.archives.gov/research/jfk/release-2025"
download_dir = "jfk_pdfs"

# Create the download directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Fetch the HTML content of the webpage
response = requests.get(base_url, headers=headers)
response.raise_for_status()  # Raise an exception for HTTP errors
html_content = response.text

# Parse the HTML content with BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Find all <a> tags with href attributes ending in ".pdf"
pdf_links = soup.find_all('a', href=lambda x: x and x.endswith('.pdf'))

# Counters to track progress
downloaded = 0
skipped = 0

# Function to download a file from a URL to a destination path
def download_file(url, dest_path):
    response = requests.get(url, stream=True, timeout=10, headers=headers)
    response.raise_for_status()  # Raise an exception if the download fails
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# Process each PDF link
for link in pdf_links:
    # Get the relative URL from the href attribute and convert to absolute URL
    relative_url = link['href']
    absolute_url = urllib.parse.urljoin(base_url, relative_url)
    
    # Extract the file name from the absolute URL
    file_name = absolute_url.split('/')[-1]
    file_path = os.path.join(download_dir, file_name)
    
    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"File {file_name} already exists, skipping.")
        skipped += 1
    else:
        print(f"Downloading {file_name}...")
        try:
            download_file(absolute_url, file_path)
            downloaded += 1
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")

# Print a summary of the operation
print(f"Downloaded {downloaded} files, skipped {skipped} files.")