import requests
import pandas as pd
import time
import numpy as np
import snowflake.connector
from typing import Tuple, Optional
import google.generativeai as genai
import re
import json
import os
from dotenv import load_dotenv
load_dotenv()
# Google Places API Configuration
PLACES_API_KEY = os.getenv("PLACES_API")

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API")  # Replace with actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# API Endpoints
NEARBY_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

# Expanded list of food-related place types
PLACE_TYPES = [
    "restaurant", "cafe", "bakery", "bar", "meal_takeaway", "meal_delivery",
    "bistro", "fast_food", "food_court", "dessert_shop", "ice_cream_shop",
    "juice_bar", "coffee_shop", "food_truck", "diner", "confectionery",
    "pub", "brewery", "winery"
]


def get_user_input() -> Tuple[str, int]:
    """Get location and radius from user input."""
    location = input("Enter location coordinates (latitude,longitude, e.g., 21.1094572,79.0674375): ").strip()
    while not re.match(r'^-?\d+\.\d+,-?\d+\.\d+$', location):
        print("Invalid format. Please use latitude,longitude (e.g., 21.1094572,79.0674375)")
        location = input("Enter location coordinates: ").strip()
    
    try:
        radius = int(input("Enter search radius in meters (e.g., 1000): ").strip())
        if radius <= 0:
            raise ValueError
    except ValueError:
        print("Invalid radius. Using default 1000 meters.")
        radius = 1000
    
    return location, radius

def get_reviews(place_id: str, api_key: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetch reviews from Google Place Details API and filter top positive/negative reviews."""
    params = {
        "place_id": place_id,
        "fields": "reviews",
        "key": api_key
    }
    
    for attempt in range(3):
        try:
            response = requests.get(PLACE_DETAILS_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            break
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Attempt {attempt + 1} failed for reviews of {place_id}: {e}")
            time.sleep(2)
            if attempt == 2:
                return None, None
    
    if "result" in data and "reviews" in data["result"] and data["result"]["reviews"]:
        reviews = data["result"]["reviews"]
        sorted_reviews = sorted(reviews, key=lambda r: r.get("rating", 0))
        
        positive_reviews = [r for r in sorted_reviews if r.get("rating", 0) >= 4]
        negative_reviews = [r for r in sorted_reviews if r.get("rating", 0) <= 2]
        
        top_positive = positive_reviews[-3:] if positive_reviews else []
        top_negative = negative_reviews[:3] if negative_reviews else []
        
        pos_reviews = " | ".join([f"[{r.get('author_name', 'Anonymous')}] {r.get('rating', 'N/A')}/5: {r.get('text', 'No text')[:200]}" 
                                 for r in top_positive]) if top_positive else None
        neg_reviews = " | ".join([f"[{r.get('author_name', 'Anonymous')}] {r.get('rating', 'N/A')}/5: {r.get('text', 'No text')[:200]}" 
                                 for r in top_negative]) if top_negative else None
        
        return pos_reviews, neg_reviews
    return None, None

def get_batch_attributes(restaurants: list) -> Tuple[dict, dict]:
    """Use Gemini API to determine cuisine types and price levels for all restaurants in one request with retry logic."""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            prompt = """
            Analyze the following diferenças data and determine:
            1. Cuisine types (comma-separated string, e.g., "Indian,Chinese")
            2. Price level (integer 1-4, where 1=Inexpensive, 2=Moderate, 3=Expensive, 4=Very expensive)
            for each restaurant based on its name and reviews.
            Return a JSON object with two keys:
            - "cuisines": a dictionary with place_id as key and cuisine types as value
            - "price_levels": a dictionary with place_id as key and price level (1-4) as value
            Use "General" for cuisine if undetermined, and 2 for price level if undetermined.
            Do not include any explanation, only the JSON output.

            Example output:
            {
                "cuisines": {
                    "place_id_1": "Indian,Chinese",
                    "place_id_2": "Italian",
                    "place_id_3": "General"
                },
                "price_levels": {
                    "place_id_1": 2,
                    "place_id_2": 3,
                    "place_id_3": 1
                }
            }

            Restaurant Data:
            """
            for restaurant in restaurants:
                prompt += f"""
                Place ID: {restaurant['place_id']}
                Name: {restaurant['name']}
                Positive Reviews: {restaurant['positive_reviews'][:500] if restaurant['positive_reviews'] else 'None'}
                Negative Reviews: {restaurant['negative_reviews'][:500] if restaurant['negative_reviews'] else 'None'}
                ---
                """
            
            print(f"Sending Gemini request for {len(restaurants)} restaurants (Attempt {attempt + 1}/{max_attempts})")
            response = model.generate_content(prompt)
            raw_response = response.text.strip()
            print(f"Raw Gemini response: {raw_response}")
            if raw_response.startswith('```json'):
                raw_response = raw_response[7:-3].strip()
            data = json.loads(raw_response)
            if not isinstance(data, dict) or "cuisines" not in data or "price_levels" not in data:
                raise ValueError("Response is not a valid JSON object with required keys")
            return data["cuisines"], data["price_levels"]
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if "429" in str(e) or "quota" in str(e).lower():
                wait_time = 4 * (2 ** attempt)  # Exponential backoff: 4s, 8s, 16s
                print(f"Quota error detected, waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            if attempt == max_attempts - 1:
                print("All attempts failed, returning default values")
                return (
                    {r['place_id']: "General" for r in restaurants},
                    {r['place_id']: 2 for r in restaurants}
                )
            continue

def get_places_and_reviews(api_key: str, location: str, radius: int = 1000) -> pd.DataFrame:
    """Fetch food-related places and reviews within a radius, avoiding duplicates."""
    places = []
    seen_place_ids = set()
    
    for keyword in ["food", "restaurant", "cafe"] + PLACE_TYPES:
        params = {
            "location": location,
            "radius": radius,
            "keyword": keyword,
            "key": api_key
        }
        
        url = NEARBY_SEARCH_URL
        page_count = 0
        while url and page_count < 3:
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
            except (requests.exceptions.RequestException, ValueError) as e:
                print(f"Error fetching data for {keyword}: {e}")
                break
            
            for place in data.get("results", []):
                place_id = place.get("place_id")
                if place_id and place_id not in seen_place_ids:
                    place_types = place.get("types", [])
                    if any(pt in PLACE_TYPES for pt in place_types) or "food" in place_types:
                        lat = place["geometry"]["location"]["lat"]
                        lng = place["geometry"]["location"]["lng"]
                        pos_reviews, neg_reviews = get_reviews(place_id, api_key)
                        
                        places.append({
                            "place_id": place_id,
                            "name": place.get("name", "Unknown"),
                            "address": place.get("vicinity", "N/A"),
                            "lat": lat,
                            "lng": lng,
                            "rating": place.get("rating", 0.0),
                            "user_ratings_total": place.get("user_ratings_total", 0),
                            "price_level": 0,  # Will be filled later
                            "business_status": place.get("business_status", "OPERATIONAL"),
                            "types": ", ".join(place_types) if place_types else "restaurant",
                            "positive_reviews": pos_reviews if pos_reviews else "No positive reviews",
                            "negative_reviews": neg_reviews if neg_reviews else "No negative reviews",
                            "cuisine_type": ""  # Will be filled later
                        })
                        seen_place_ids.add(place_id)
            
            next_page_token = data.get("next_page_token")
            if next_page_token:
                time.sleep(2)
                url = NEARBY_SEARCH_URL
                params = {"pagetoken": next_page_token, "key": api_key}
                page_count += 1
            else:
                url = None
    
    # Get cuisine types and price levels in one batch
    if places:
        cuisine_types, price_levels = get_batch_attributes(places)
        for place in places:
            place["cuisine_type"] = cuisine_types.get(place["place_id"], "General")
            place["price_level"] = price_levels.get(place["place_id"], 2)
    
    return pd.DataFrame(places)

def store_data_in_snowflake(df: pd.DataFrame) -> None:
    """Store data in Snowflake, ensuring no null entries and avoiding duplicates."""
    try:
        conn = snowflake.connector.connect(**json.loads(os.getenv("SNOWFLAKE_CONFIG")))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE OR REPLACE TABLE RESTAURANTDB.RESTAURANT_SCHEMA.CLIFTON_N (
                PLACE_ID VARCHAR(16777216),
                NAME VARCHAR(16777216),
                ADDRESS VARCHAR(16777216),
                LAT NUMBER(38,7),
                LNG NUMBER(38,14),
                RATING NUMBER(38,1),
                USER_RATINGS_TOTAL NUMBER(38,0),
                PRICE_LEVEL NUMBER(38,0),
                BUSINESS_STATUS VARCHAR(16777216),
                TYPES VARCHAR(16777216),
                POSITIVE_REVIEWS VARCHAR(16777216),
                NEGATIVE_REVIEWS VARCHAR(16777216),
                CUISINETYPE VARCHAR(50)
            )
        """)

        for _, row in df.iterrows():
            # Ensure no null values
            place_id = row["place_id"] or "Unknown"
            name = row["name"] or "Unknown"
            address = row["address"] or "N/A"
            lat = row["lat"] if pd.notnull(row["lat"]) else 0.0
            lng = row["lng"] if pd.notnull(row["lng"]) else 0.0
            rating = row["rating"] if pd.notnull(row["rating"]) else 0.0
            user_ratings_total = int(row["user_ratings_total"]) if pd.notnull(row["user_ratings_total"]) else 0
            price_level = int(row["price_level"]) if pd.notnull(row["price_level"]) else 2
            business_status = row["business_status"] or "OPERATIONAL"
            types = row["types"] or "restaurant"
            positive_reviews = row["positive_reviews"] or "No positive reviews"
            negative_reviews = row["negative_reviews"] or "No negative reviews"
            cuisine_type = row["cuisine_type"] or "General"

            cursor.execute("""
                MERGE INTO CLIFTON_N AS target
                USING (SELECT %s AS PLACE_ID) AS source
                ON target.PLACE_ID = source.PLACE_ID
                WHEN NOT MATCHED THEN INSERT (
                    PLACE_ID, NAME, ADDRESS, LAT, LNG, RATING, USER_RATINGS_TOTAL, 
                    PRICE_LEVEL, BUSINESS_STATUS, TYPES, POSITIVE_REVIEWS, NEGATIVE_REVIEWS, CUISINETYPE
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                place_id,
                place_id, name, address, lat, lng, rating, user_ratings_total,
                price_level, business_status, types, positive_reviews, negative_reviews, cuisine_type
            ))
        
        conn.commit()
        print(f"Stored {len(df)} rows in Snowflake without duplicates.")
    
    except Exception as e:
        print(f"Error storing data in Snowflake: {e}")
    
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    # Get user input
    location, radius = get_user_input()
    
    # Fetch live data from Google Places API
    df = get_places_and_reviews(PLACES_API_KEY, location, radius)
    df = df.replace({np.nan: None})
    
    # Save to CSV for backup
    csv_filename = f"restaurant_data___{location.replace(',', '_')}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

    df1 = pd.read_csv(csv_filename)
    df1 = df1.replace({np.nan: None})
    print(df1[['name', 'rating', 'price_level', 'positive_reviews', 'negative_reviews', 'cuisine_type']].head())
    
    # Store in Snowflake
    store_data_in_snowflake(df1)