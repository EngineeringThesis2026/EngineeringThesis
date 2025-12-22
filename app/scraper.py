"""
Scraper that downloads court ruling PDFs from orzeczenia.ms.gov.pl website.
"""

import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
from urllib.parse import urljoin


# CONFIGURATION
BASE_URL = "https://orzeczenia.ms.gov.pl"

# HTTP Headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "DNT": "1",
}

# Delay between requests (seconds)
DELAY = 0.5


# FUNCTION 1: Getting page with results


def get_page(search_url, page_number=1):
    """
    Gets HTML of search results page.

    Returns HTML as text or None if error.
    """
    # Change page number in URL
    url_without_number = search_url.rsplit("/", 1)[0]
    full_url = f"{url_without_number}/{page_number}"

    print(f"Getting page {page_number}: {full_url}")

    try:
        # Send GET request
        response = requests.get(full_url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        # Wait
        time.sleep(DELAY)

        print(f"Got page ({len(response.text)} characters)")
        return response.text

    except requests.exceptions.Timeout:
        print("Timeout - page is not responding")
        return None

    except Exception as error:
        print(f"Error: {error}")
        return None


# FUNCTION 2: Getting links to rulings


def get_links(html):
    """
    Gets links to rulings from HTML.

    Returns list of URLs.
    """
    if not html:
        return []

    print("Looking for links to rulings...")

    try:
        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")
        found_links = []

        # Find all links
        for link in soup.find_all("a", href=True):
            href = link["href"]

            # Only links to rulings
            if "/details/" in href or "/content/" in href:
                full_url = urljoin(BASE_URL, href)

                # Avoid duplicates
                if full_url not in found_links:
                    found_links.append(full_url)

        print(f"Found {len(found_links)} links")
        return found_links

    except Exception as error:
        print(f"Parsing error: {error}")
        return []


# FUNCTION 3: Downloading single PDF


def download_pdf(ruling_url, output_folder):
    """
    Downloads PDF from single ruling.

    Returns file path or None if error.
    """
    try:
        # Change /details/ to /content/
        if "/details/" in ruling_url:
            ruling_url = ruling_url.replace("/details/", "/content/")

        # Check if file already exists
        parts = ruling_url.rstrip("/").split("/")
        if parts:
            case_id = parts[-1].replace("$N/", "")
            file_path = Path(output_folder) / f"{case_id}.pdf"
            if file_path.exists():
                print(f"{case_id}.pdf already exists - skipping")
                return str(file_path)

        print("Getting ruling...")

        # Get ruling page
        response = requests.get(ruling_url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        # Find PDF link
        soup = BeautifulSoup(response.text, "html.parser")
        pdf_link = None

        # Look for download button
        button = soup.find("li", class_="download_btn")
        if button:
            a_tag = button.find("a", href=True)
            if a_tag:
                pdf_link = urljoin(BASE_URL, a_tag["href"])
                print("Found PDF link")

        # Plan B: search everywhere
        if not pdf_link:
            for link in soup.find_all("a", href=True):
                if "/content.pdffile/" in link["href"]:
                    pdf_link = urljoin(BASE_URL, link["href"])
                    print("Found PDF link (plan B)")
                    break

        if not pdf_link:
            print("PDF not found")
            return None

        # Download PDF
        print("Downloading PDF...")
        time.sleep(DELAY)

        pdf_response = requests.get(pdf_link, headers=HEADERS, timeout=60)
        pdf_response.raise_for_status()

        # Check if it is PDF
        if not pdf_response.content.startswith(b"%PDF"):
            print("This is not a PDF!")
            return None

        # Create file name
        file_name = None
        if "/content/" in ruling_url:
            parts = ruling_url.rstrip("/").split("/")
            if parts:
                case_id = parts[-1].replace("$N/", "")
                file_name = f"{case_id}.pdf"

        if not file_name:
            file_name = f"ruling_{int(time.time())}.pdf"

        # Save PDF
        file_path = Path(output_folder) / file_name

        with open(file_path, "wb") as file:
            file.write(pdf_response.content)

        print(f"Saved: {file_name}")

        return str(file_path)

    except Exception as error:
        print(f"Error: {error}")
        return None


# FUNCTION 4: Download legal codes from ISAP


def download_codes():
    """Downloads Kodeks cywilny and Kodeks pracy from ISAP."""
    Path("app/data/civil_code").mkdir(parents=True, exist_ok=True)
    civil_code_path = Path("app/data/civil_code/kodeks_cywilny.pdf")
    if civil_code_path.exists():
        print("kodeks_cywilny.pdf already exists - skipping")
    else:
        print("Downloading kodeks_cywilny.pdf...")
        r = requests.get(
            "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19640160093/U/D19640093Lj.pdf",
            timeout=60,
        )
        civil_code_path.write_bytes(r.content)

    Path("app/data/labor_code").mkdir(parents=True, exist_ok=True)
    labor_code_path = Path("app/data/labor_code/kodeks_pracy.pdf")
    if labor_code_path.exists():
        print("kodeks_pracy.pdf already exists - skipping")
    else:
        print("Downloading kodeks_pracy.pdf...")
        r = requests.get(
            "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19740240141/U/D19740141Lj.pdf",
            timeout=60,
        )
        labor_code_path.write_bytes(r.content)


# FUNCTION 5: Main scraping function


def scrape(search_url, page_count=1, output_folder="app/data/scraped_judgments"):
    """
    MAIN FUNCTION - Downloads PDFs from many pages.

    Returns number of downloaded PDFs.
    """
    print("=" * 70)
    print("START SCRAPING")
    print("=" * 70)

    # Create folder
    folder = Path(output_folder)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Folder: {folder}\n")
    except Exception as error:
        print(f"Cannot create folder: {error}")
        return 0

    downloaded_count = 0

    # Loop through pages
    for page_number in range(1, page_count + 1):
        print(f"\n{'=' * 70}")
        print(f"PAGE {page_number}/{page_count}")
        print(f"{'=' * 70}")

        # 1. Get page
        html = get_page(search_url, page_number)

        if not html:
            print(f"Skipping page {page_number}")
            continue

        # 2. Get links
        links = get_links(html)

        if not links:
            print(f"No links on page {page_number}")
            continue

        # 3. Download PDFs
        for i, link in enumerate(links, 1):
            print(f"\n   Ruling {i}/{len(links)}")

            file_path = download_pdf(link, output_folder)

            if file_path:
                downloaded_count += 1

            # Delay
            if i < len(links):
                time.sleep(DELAY)

    # Summary
    print(f"\n{'=' * 70}")
    print("END")
    print(f"{'=' * 70}")
    print(f"Downloaded PDFs: {downloaded_count}")
    print(f"Folder: {folder}")
    print(f"{'=' * 70}\n")

    return downloaded_count


if __name__ == "__main__":
    import sys

    # Default values
    URL = "https://orzeczenia.ms.gov.pl/search/advanced/$N/$N/$N/$N/$N/153510/$N/$N/$N/$N/$N/$N/$N/$N/$N/score/descending/1"
    OUTPUT_FOLDER = "app/data/rulings"

    # Download legal codes first
    download_codes()

    # Default: 10 pages
    pages = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    scrape(search_url=URL, page_count=pages, output_folder=OUTPUT_FOLDER)
