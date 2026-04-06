import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from datetime import datetime, timedelta
import time
import ssl
import certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())

page_url = "https://opendataphilly.org/datasets/crime-incidents/"

current_year = datetime.now().year
dataset_title = f"Crime Incidents from {current_year} (CSV)"

response = requests.get(page_url)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

csv_link = None
for a_tag in soup.find_all("a"):
    span = a_tag.find("span", {"property": "dct:title"})
    if span and span.text.strip() == dataset_title:
        csv_link = a_tag.get("href")
        break

if not csv_link:
    raise Exception("CSV link for 2025 not found on the page.")

print("Found CSV link:", csv_link)

csv_response = requests.get(csv_link)
csv_response.raise_for_status()

df = pd.read_csv(StringIO(csv_response.text))

df.to_csv("input_crime_data.csv", index=False)

print("CSV downloaded and saved as 'input_crime_data.csv'.")

# ----- Configuration ----
input_csv = "input_crime_data.csv"      
output_csv = "crime_past_3_days.csv"   
columns_kept = ["text_general_code", "dispatch_date_time", "lat", "lng"]
batch_size = 50
delay = 1
retries = 3
error_wait_seconds = 5
# -------------------------

df = pd.read_csv(input_csv)

df = df.dropna(subset=["lat", "lng"])
df = df[pd.to_numeric(df["lat"], errors="coerce").notna()]
df = df[pd.to_numeric(df["lng"], errors="coerce").notna()]

df = df.sort_values("dispatch_date_time", ascending=False)
df["dispatch_date_time"] = pd.to_datetime(df["dispatch_date_time"], errors="coerce")

now = pd.Timestamp.utcnow()
three_days_ago = now - pd.Timedelta(days=3)
df = df[df["dispatch_date_time"] >= three_days_ago]

missing_cols = [col for col in columns_kept if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")
df = df[columns_kept]

print(f"Processing {len(df)} rows from the last day...")

geolocator = Nominatim(user_agent="zipcode_finder", ssl_context=ssl_context)
reverse = RateLimiter(
    geolocator.reverse,
    min_delay_seconds=delay,   
    max_retries=retries,        
    error_wait_seconds=error_wait_seconds, 
)

def get_zip(lat, lng):
    try:
        location = reverse((lat, lng), language="en", addressdetails=True)
        if location and "postcode" in location.raw["address"]:
            return location.raw["address"]["postcode"]
    except Exception as e:
        print(f"Error at ({lat}, {lng}): {e}")
    return None

if "zipcode" not in df.columns:
    df["zipcode"] = None

for start in range(0, len(df), batch_size):
    end = start + batch_size
    batch = df.iloc[start:end]
    print(f"Processing rows {start} to {end-1}...")

    for idx, row in batch.iterrows():
        lat, lng = row["lat"], row["lng"]
        if pd.isna(row["zipcode"]) and pd.notna(lat) and pd.notna(lng):
            zipcode = get_zip(lat, lng)
            df.at[idx, "zipcode"] = zipcode

    df.to_csv(output_csv, index=False)
    print(f"Saved progress to {output_csv}")
    time.sleep(2)

print("All batches processed.")