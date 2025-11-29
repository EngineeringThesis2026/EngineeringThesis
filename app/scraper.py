"""
Scraper pobierajacy orzeczenia pdf ze strony orzeczenia.ms.gov.pl.
"""

import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
from urllib.parse import urljoin


# KONFIGURACJA
STRONA_BAZOWA = "https://orzeczenia.ms.gov.pl"

# Nagłówki HTTP
NAGLOWKI = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "DNT": "1",
}

# Opóźnienie między requestami (sekundy)
OPOZNIENIE = 0.5


# FUNKCJA 1: Pobieranie strony z wynikami


def pobierz_strone(url_wyszukiwania, numer_strony=1):
    """
    Pobiera HTML strony z wynikami wyszukiwania.

    Zwraca HTML jako tekst lub None jeśli błąd.
    """
    # Zamień numer strony w URL
    url_bez_numeru = url_wyszukiwania.rsplit("/", 1)[0]
    url_pelny = f"{url_bez_numeru}/{numer_strony}"

    print(f"Pobieram stronę {numer_strony}: {url_pelny}")

    try:
        # Wyślij GET
        odpowiedz = requests.get(url_pelny, headers=NAGLOWKI, timeout=30)
        odpowiedz.raise_for_status()

        # Poczekaj
        time.sleep(OPOZNIENIE)

        print(f"Pobrano ({len(odpowiedz.text)} znaków)")
        return odpowiedz.text

    except requests.exceptions.Timeout:
        print(f"Timeout - strona nie odpowiada")
        return None

    except Exception as blad:
        print(f"Błąd: {blad}")
        return None


# FUNKCJA 2: Wyciąganie linków do orzeczeń


def wyciagnij_linki(html):
    """
    Wyciąga linki do orzeczeń z HTML-a.

    Zwraca listę URLi.
    """
    if not html:
        return []

    print("Szukam linków do orzeczeń...")

    try:
        # Parsuj HTML
        soup = BeautifulSoup(html, "html.parser")
        znalezione_linki = []

        # Znajdź wszystkie linki
        for link in soup.find_all("a", href=True):
            href = link["href"]

            # Tylko linki do orzeczeń
            if "/details/" in href or "/content/" in href:
                pelny_url = urljoin(STRONA_BAZOWA, href)

                # Unikanie duplikatów
                if pelny_url not in znalezione_linki:
                    znalezione_linki.append(pelny_url)

        print(f"Znaleziono {len(znalezione_linki)} linków")
        return znalezione_linki

    except Exception as blad:
        print(f"Błąd parsowania: {blad}")
        return []


# FUNKCJA 3: Pobieranie pojedynczego PDF


def pobierz_pdf(url_orzeczenia, folder_output):
    """
    Pobiera PDF z pojedynczego orzeczenia.

    Zwraca ścieżkę do pliku lub None jeśli błąd.
    """
    try:
        # Zamień /details/ na /content/
        if "/details/" in url_orzeczenia:
            url_orzeczenia = url_orzeczenia.replace("/details/", "/content/")
            print(f"Zmieniono na /content/")

        print(f"Pobieram orzeczenie...")

        # Pobierz stronę orzeczenia
        odpowiedz = requests.get(url_orzeczenia, headers=NAGLOWKI, timeout=30)
        odpowiedz.raise_for_status()

        # Znajdź link do PDF
        soup = BeautifulSoup(odpowiedz.text, "html.parser")
        link_do_pdf = None

        # Szukaj przycisku pobierania
        przycisk = soup.find("li", class_="download_btn")
        if przycisk:
            tag_a = przycisk.find("a", href=True)
            if tag_a:
                link_do_pdf = urljoin(STRONA_BAZOWA, tag_a["href"])
                print(f"Znaleziono link PDF")

        # Plan B: szukaj wszędzie
        if not link_do_pdf:
            for link in soup.find_all("a", href=True):
                if "/content.pdffile/" in link["href"]:
                    link_do_pdf = urljoin(STRONA_BAZOWA, link["href"])
                    print(f"Znaleziono link PDF (plan B)")
                    break

        if not link_do_pdf:
            print(f"Nie znaleziono PDF")
            return None

        # Pobierz PDF
        print(f"Pobieram PDF...")
        time.sleep(OPOZNIENIE)

        odpowiedz_pdf = requests.get(link_do_pdf, headers=NAGLOWKI, timeout=60)
        odpowiedz_pdf.raise_for_status()

        # Sprawdź czy to PDF
        if not odpowiedz_pdf.content.startswith(b"%PDF"):
            print(f"To nie jest PDF!")
            return None

        # Wygeneruj nazwę pliku
        nazwa_pliku = None
        if "/content/" in url_orzeczenia:
            czesci = url_orzeczenia.rstrip("/").split("/")
            if czesci:
                id_sprawy = czesci[-1].replace("$N/", "")
                nazwa_pliku = f"{id_sprawy}.pdf"

        if not nazwa_pliku:
            nazwa_pliku = f"orzeczenie_{int(time.time())}.pdf"

        # Zapisz PDF
        sciezka = Path(folder_output) / nazwa_pliku

        with open(sciezka, "wb") as plik:
            plik.write(odpowiedz_pdf.content)

        print(f"Zapisano: {nazwa_pliku}")

        return str(sciezka)

    except Exception as blad:
        print(f"Błąd: {blad}")
        return None


# FUNKCJA 4: Główna funkcja scrapingu


def scrapuj(url_wyszukiwania, ile_stron=1, folder_output="app/data/scraped_judgments"):
    """
    GŁÓWNA FUNKCJA - Pobiera PDFy z wielu stron.

    Zwraca liczbę pobranych PDFów.
    """
    print("=" * 70)
    print("START SCRAPINGU")
    print("=" * 70)

    # Stwórz folder
    folder = Path(folder_output)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Folder: {folder}\n")
    except Exception as blad:
        print(f"Nie można stworzyć folderu: {blad}")
        return 0

    licznik_pobranych = 0

    # Pętla przez strony
    for numer_strony in range(1, ile_stron + 1):
        print(f"\n{'=' * 70}")
        print(f"STRONA {numer_strony}/{ile_stron}")
        print(f"{'=' * 70}")

        # 1. Pobierz stronę
        html = pobierz_strone(url_wyszukiwania, numer_strony)

        if not html:
            print(f"Pomijam stronę {numer_strony}")
            continue

        # 2. Wyciągnij linki
        linki = wyciagnij_linki(html)

        if not linki:
            print(f"Brak linków na stronie {numer_strony}")
            continue

        # 3. Pobierz PDFy
        for i, link in enumerate(linki, 1):
            print(f"\n   Orzeczenie {i}/{len(linki)}")

            sciezka = pobierz_pdf(link, folder_output)

            if sciezka:
                licznik_pobranych += 1

            # Opóźnienie
            if i < len(linki):
                time.sleep(OPOZNIENIE)

    # Podsumowanie
    print(f"\n{'=' * 70}")
    print(f"KONIEC")
    print(f"{'=' * 70}")
    print(f"Pobrano PDFów: {licznik_pobranych}")
    print(f"Folder: {folder}")
    print(f"{'=' * 70}\n")

    return licznik_pobranych


if __name__ == "__main__":
    # URL wyszukiwania
    URL = "https://orzeczenia.ms.gov.pl/search/advanced/$N/$N/$N/$N/$N/153510/$N/$N/$N/$N/$N/$N/$N/$N/$N/score/descending/1"

    scrapuj(url_wyszukiwania=URL, ile_stron=2, folder_output="app/data/scrape")
