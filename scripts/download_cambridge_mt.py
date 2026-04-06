import os
import re
import zipfile
import urllib.request
from bs4 import BeautifulSoup

TARGET_DIR = os.path.expanduser("~/Downloads/Agent_Training_Assets/Audio")
os.makedirs(TARGET_DIR, exist_ok=True)

url = "https://cambridge-mt.com/ms/mtk/"
print(f"Fetching {url}...")
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8')
except Exception as e:
    print(f"Failed to fetch page: {e}")
    exit(1)

soup = BeautifulSoup(html, "html.parser")
links = soup.find_all("a", href=True)

zip_links = []
for a in links:
    href = a['href']
    if href.endswith('.zip') and '_Full' in href:
        # Resolve relative URLs
        if not href.startswith('http'):
            href = urllib.parse.urljoin(url, href)
        zip_links.append(href)

# Deduplicate
zip_links = list(set(zip_links))

print(f"Found {len(zip_links)} full multitrack ZIP links. Taking first 10 for demonstration.")
zip_links = zip_links[:10]

for idx, link in enumerate(zip_links):
    filename = os.path.basename(link)
    zip_path = os.path.join(TARGET_DIR, filename)
    extract_path = os.path.join(TARGET_DIR, filename.replace('.zip', ''))
    
    if os.path.exists(extract_path):
        print(f"[{idx+1}/10] Already extracted: {extract_path}")
        continue
        
    print(f"[{idx+1}/10] Downloading {link}...")
    try:
        urllib.request.urlretrieve(link, zip_path)
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        os.remove(zip_path) # Clean up zip to save space
        print(f"Done extracting {extract_path}")
    except Exception as e:
        print(f"Error processing {link}: {e}")

print("Multitracks download complete.")
