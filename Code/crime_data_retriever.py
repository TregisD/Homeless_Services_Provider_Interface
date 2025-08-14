import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

page_url = "https://opendataphilly.org/datasets/crime-incidents/"

response = requests.get(page_url)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

csv_link = None
for a_tag in soup.find_all("a"):
    span = a_tag.find("span", {"property": "dct:title"})
    if span and span.text.strip() == "Crime Incidents from 2025 (CSV)":
        csv_link = a_tag.get("href")
        break

if not csv_link:
    raise Exception("CSV link for 2025 not found on the page.")

print("Found CSV link:", csv_link)

csv_response = requests.get(csv_link)
csv_response.raise_for_status()

df = pd.read_csv(StringIO(csv_response.text))

df.to_csv("crime_incidents_2025.csv", index=False)

print("CSV downloaded and saved as 'crime_incidents_2025.csv'.")