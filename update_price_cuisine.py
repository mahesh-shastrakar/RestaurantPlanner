import pandas as pd
import snowflake.connector
from typing import Dict
import google.generativeai as genai
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()
# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API")  # Replace with actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# Snowflake Connection Parameters


def get_batch_cuisine_types(restaurants: list) -> Dict[str, str]:
    """Use Gemini API to determine cuisine types for all restaurants in one request."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        prompt = """
        Analyze the following restaurant data and determine the cuisine types for each restaurant based on its name and reviews. 
        Return a JSON object where the key is the place_id and the value is a comma-separated string of cuisine types (e.g., "Indian,Chinese").
        If no specific cuisine can be determined, use "General".
        Do not include any explanation, only the JSON output.

        Example output:
        {
            "place_id_1": "Indian,Chinese",
            "place_id_2": "Italian",
            "place_id_3": "General"
        }

        Restaurant Data:
        """
        for restaurant in restaurants:
            prompt += f"""
            Place ID: {restaurant['PLACE_ID']}
            Name: {restaurant['NAME']}
            Positive Reviews: {restaurant['POSITIVE_REVIEWS'][:500] if restaurant['POSITIVE_REVIEWS'] else 'None'}
            Negative Reviews: {restaurant['NEGATIVE_REVIEWS'][:500] if restaurant['NEGATIVE_REVIEWS'] else 'None'}
            ---
            """
        
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                raw_response = response.text.strip()
                if raw_response.startswith('```json'):
                    raw_response = raw_response[7:-3].strip()
                cuisine_data = json.loads(raw_response)
                if not isinstance(cuisine_data, dict):
                    raise ValueError("Response is not a valid JSON object")
                return cuisine_data
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Attempt {attempt + 1} failed to parse Gemini cuisine response: {e}")
                if attempt == 2:
                    print("All attempts failed for cuisine, returning default values")
                    return {r['PLACE_ID']: "General" for r in restaurants}
                time.sleep(2)
    except Exception as e:
        print(f"Error getting cuisines: {e}")
        return {r['PLACE_ID']: "General" for r in restaurants}

def get_batch_price_levels(restaurants: list) -> Dict[str, int]:
    """Use Gemini API to infer price levels (1-4) for all restaurants based on reviews in one request."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        prompt = """
        Analyze the following restaurant data and infer the price level (1-4) for each restaurant based on its name and reviews.
        Price levels are:
        1: Inexpensive (budget, fast food, street food)
        2: Moderate (casual dining, mid-range)
        3: Expensive (upscale, fine dining)
        4: Very expensive (luxury, high-end dining)
        Return a JSON object where the key is the place_id and the value is an integer (1, 2, 3, or 4).
        If no specific price level can be determined, use 2 (Moderate).
        Do not include any explanation, only the JSON output.

        Example output:
        {
            "place_id_1": 2,
            "place_id_2": 3,
            "place_id_3": 1
        }

        Restaurant Data:
        """
        for restaurant in restaurants:
            prompt += f"""
            Place ID: {restaurant['PLACE_ID']}
            Name: {restaurant['NAME']}
            Positive Reviews: {restaurant['POSITIVE_REVIEWS'][:500] if restaurant['POSITIVE_REVIEWS'] else 'None'}
            Negative Reviews: {restaurant['NEGATIVE_REVIEWS'][:500] if restaurant['NEGATIVE_REVIEWS'] else 'None'}
            ---
            """
        
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                raw_response = response.text.strip()
                if raw_response.startswith('```json'):
                    raw_response = raw_response[7:-3].strip()
                price_data = json.loads(raw_response)
                if not isinstance(price_data, dict):
                    raise ValueError("Response is not a valid JSON object")
                return price_data
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Attempt {attempt + 1} failed to parse Gemini price level response: {e}")
                if attempt == 2:
                    print("All attempts failed for price levels, returning default values")
                    return {r['PLACE_ID']: 2 for r in restaurants}
                time.sleep(2)
    except Exception as e:
        print(f"Error getting price levels: {e}")
        return {r['PLACE_ID']: 2 for r in restaurants}

def update_price_and_cuisine(table_name: str) -> None:
    """Update only PRICE_LEVEL and CUISINETYPE in the specified Snowflake table."""
    try:
        # Connect to Snowflake
        conn = snowflake.connector.connect(**json.loads(os.getenv("SNOWFLAKE_CONFIG")))
        cursor = conn.cursor()

        # Fetch required columns from the table
        query = f"""
            SELECT PLACE_ID, NAME, POSITIVE_REVIEWS, NEGATIVE_REVIEWS
            FROM {table_name}
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)

        # Convert DataFrame to list of dictionaries for processing
        restaurants = df.to_dict('records')

        # Get updated cuisine types and price levels using Gemini API
        cuisine_types = get_batch_cuisine_types(restaurants)
        price_levels = get_batch_price_levels(restaurants)

        # Update the table with new values
        for restaurant in restaurants:
            place_id = restaurant['PLACE_ID']
            new_cuisine = cuisine_types.get(place_id, "General")
            new_price_level = price_levels.get(place_id, 2)

            update_query = f"""
                UPDATE {table_name}
                SET 
                    CUISINETYPE = %s,
                    PRICE_LEVEL = %s
                WHERE PLACE_ID = %s
            """
            cursor.execute(update_query, (new_cuisine, new_price_level, place_id))

        conn.commit()
        print(f"Successfully updated PRICE_LEVEL and CUISINETYPE for {len(restaurants)} records in {table_name}.")

    except Exception as e:
        print(f"Error updating data in Snowflake: {e}")
    
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    # Get table name from user
    table_name = input("Enter the Snowflake table name (e.g., RESTAURANTDB.RESTAURANT_SCHEMA.RESTAURANT_DATA_NAAMEE): ").strip()
    
    # Update price levels and cuisine types
    update_price_and_cuisine(table_name)