from openai import OpenAI
import pandas as pd
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Initialize OpenAI client
client = OpenAI(api_key="sk-proj-YpGRHNbs8pQaQpJXqI8MT3BlbkFJV68vFPa3I1q4hNiggdnj")

# Simple cache for probability distributions
cache = {}
CACHE_DURATION = timedelta(hours=1)

def load_restaurant_data() -> List[Dict]:
    """
    Load restaurant data from OSR.csv
    """
    try:
        df = pd.read_csv('OSR.csv')
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading restaurant data: {str(e)}")
        return []

def load_archetype_data() -> List[Dict]:
    """
    Load archetype data from 1generated_archetypes.csv
    """
    try:
        df = pd.read_csv('1generated_archetypes.csv')
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading archetype data: {str(e)}")
        return []

def get_random_archetype(archetypes: List[Dict]) -> Optional[Dict]:
    """
    Get a random archetype from the list
    """
    if archetypes:
        return random.choice(archetypes)
    return None

def format_restaurant_data(restaurants: List[Dict]) -> str:
    """
    Format restaurant data for the prompt in a clean, readable way
    """
    restaurant_info = []
    for r in restaurants:
        info = f"- {r['brand']}"
        if 'cuisine' in r:
            info += f" (Cuisine: {r['cuisine']})"
        if 'price_tier' in r:
            info += f" (Price: {r['price_tier']})"
        restaurant_info.append(info)
    return "\n".join(restaurant_info)

def get_cache_key(archetype: Dict, restaurants: List[Dict], closed_restaurant: str) -> str:
    """
    Generate a cache key for the given parameters
    """
    # Sort restaurant names to ensure consistent keys
    restaurant_names = sorted([r['brand'] for r in restaurants])
    return f"{json.dumps(archetype)}_{','.join(restaurant_names)}_{closed_restaurant}"

def get_restaurant_choice(archetype: Dict, available_restaurants: List[Dict], closed_restaurant: str) -> Dict[str, float]:
    """
    Generate a probability distribution over restaurant choices based on an archetype.

    Parameters:
    - archetype: A dictionary representing the archetype
    - available_restaurants: A list of dictionaries representing available restaurants
    - closed_restaurant: The restaurant that is closed

    Returns:
    - A tuple containing:
        - A dictionary with restaurant names as keys and their probabilities as values
        - A string containing the reasoning for the distribution
    """
    # Check cache first
    cache_key = get_cache_key(archetype, available_restaurants, closed_restaurant)
    if cache_key in cache:
        cached_result, timestamp = cache[cache_key]
        if datetime.now() - timestamp < CACHE_DURATION:
            return cached_result

    # Filter out the closed restaurant
    remaining_restaurants = [r for r in available_restaurants if r['brand'] != closed_restaurant]
    valid_restaurant_names = {r['brand'] for r in remaining_restaurants}

    # Format restaurant data
    formatted_restaurants = format_restaurant_data(remaining_restaurants)

    # Create a detailed prompt for GPT-4
    prompt = f"""Given the following information:
    Archetype Profile:
    - Age: {archetype['Age']}
    - Sex: {archetype['Sex']}
    - Race: {archetype['Race']}
    - Marital Status: {archetype['Marital Status']}
    - Education: {archetype['Education']}
    - Income: {archetype['Income']}
    - Housing: {archetype['Housing']}

    Available Restaurants:
{formatted_restaurants}

    Closed Restaurant: {closed_restaurant}

    Based on the demographic profile and restaurant characteristics, provide:
    1. A probability distribution for restaurant choices
    2. A brief explanation of your reasoning (max 2-3 sentences)

    IMPORTANT: Your response must be a valid JSON object with the following format:
    {{
        "probabilities": {{
            "restaurant_name": probability_value,
            ...
        }},
        "reasoning": "Your brief explanation here"
    }}
    where:
    - restaurant_name must be exactly as shown in the Available Restaurants list
    - probability_value must be a number between 0 and 1
    - probabilities must sum to 1.0
    - use double quotes for strings
    - include only restaurants from the Available Restaurants list
    - provide a concise explanation in the reasoning field (2-3 sentences max)
    - do not include any text before or after the JSON object
    - make sure to include the closing curly brace }}"""

    try:
        # Call OpenAI API with GPT-4
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides probability distributions for restaurant choices based on demographic profiles and restaurant characteristics. You must always respond with a valid JSON object containing restaurant probabilities and a brief reasoning (2-3 sentences). Make sure to include the closing curly brace and ensure probabilities sum to 1.0."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        # Parse the response to extract probabilities and reasoning
        result = parse_response(response, valid_restaurant_names)
        
        # Cache the result
        if result and result.get('probabilities'):
            cache[cache_key] = (result, datetime.now())
            
        return result
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return {'probabilities': {}, 'reasoning': ''}

def parse_response(response, valid_restaurant_names: set) -> Dict:
    """
    Parse the OpenAI response to extract probabilities and reasoning.

    Parameters:
    - response: The response from OpenAI API
    - valid_restaurant_names: Set of valid restaurant names to validate against

    Returns:
    - A dictionary containing probabilities and reasoning
    """
    try:
        # Get the content from the response
        content = response.choices[0].message.content.strip()
        
        # Try to parse the entire response as JSON first
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to fix and parse the response
            # First, remove any trailing commas
            content = content.rstrip(',\n ')
            
            # If we have an incomplete JSON (missing closing brace)
            missing_braces = content.count('{') - content.count('}')
            if missing_braces > 0:
                content += '}' * missing_braces
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract the JSON structure using regex
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except:
                        print("Could not parse response into valid JSON")
                        print("Raw response:", content)
                        return {'probabilities': {}, 'reasoning': ''}
                else:
                    print("Could not find valid JSON structure")
                    print("Raw response:", content)
                    return {'probabilities': {}, 'reasoning': ''}

        # Validate the structure
        if not isinstance(result, dict) or 'probabilities' not in result or 'reasoning' not in result:
            print("Response does not have the expected structure")
            return {'probabilities': {}, 'reasoning': ''}

        probabilities = result['probabilities']
        reasoning = result['reasoning']

        # Validate restaurant names and remove invalid ones
        invalid_restaurants = set(probabilities.keys()) - valid_restaurant_names
        if invalid_restaurants:
            print(f"Warning: Removing invalid restaurants: {invalid_restaurants}")
            probabilities = {k: v for k, v in probabilities.items() if k in valid_restaurant_names}
            
        # Remove any zero or negative probabilities
        probabilities = {k: v for k, v in probabilities.items() if v > 0}
            
        # Ensure all values are numbers and sum to 1.0
        total = sum(probabilities.values())
        if total <= 0:
            print("No valid probabilities found")
            return {'probabilities': {}, 'reasoning': ''}
            
        # Normalize probabilities to sum to 1.0
        probabilities = {k: v/total for k, v in probabilities.items()}
            
        return {'probabilities': probabilities, 'reasoning': reasoning}
            
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        print("Raw response:", content)
        return {'probabilities': {}, 'reasoning': ''}

# Example usage
if __name__ == "__main__":
    # Load data from CSV files
    all_restaurants = load_restaurant_data()
    all_archetypes = load_archetype_data()
    
    if not all_restaurants:
        print("Error: No restaurant data available")
        exit(1)
        
    if not all_archetypes:
        print("Error: No archetype data available")
        exit(1)
    
    # Get a random archetype
    archetype = get_random_archetype(all_archetypes)
    
    if archetype:
        # Example: Close a random restaurant
        closed_restaurant = "Subway"

        # Get probability distribution
        result = get_restaurant_choice(archetype, all_restaurants, closed_restaurant)
        
        print("\nArchetype Profile:")
        for key, value in archetype.items():
            print(f"- {key}: {value}")
            
        if result and result.get('probabilities'):
            print("\nProbability Distribution:")
            # Sort by probability (descending)
            sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
            for restaurant, prob in sorted_probs:
                print(f"- {restaurant}: {prob:.3f}")
            print(f"\nTotal probability: {sum(result['probabilities'].values()):.3f}")
            
            if result.get('reasoning'):
                print("\nReasoning:")
                print(result['reasoning'])
        else:
            print("\nError: Could not generate probability distribution")
    else:
        print("Error: No archetypes available")
