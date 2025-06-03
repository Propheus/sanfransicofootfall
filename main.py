#!/usr/bin/env python
# coding: utf-8

# # Agent-Based Model Simulation for San Francisco Restaurants

# ## 1. Imports
import os
import osmnx as ox
import pyproj
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import random
import time
import pygame
import matplotlib.pyplot as plt
from scipy.stats import poisson
from geopy.geocoders import Nominatim
from math import radians, sin, cos, asin, sqrt
from tqdm import tqdm # For progress bars

# ## 2. Configuration and Constants
# --- File Paths ---
OSR_CSV_PATH = "data/OSR.csv"
LAT_LON_STORES_CSV_PATH = "data/lat_lon_stores.csv"
TRADE_AREA_PROFILING_CSV_PATH = "data/trade_area_profiling.csv"
HOURLY_FOOTFALL_REPORT_PATH = "data/hourly_footfall_report.csv"
AGENT_SUMMARY_REPORT_PATH = "data/agent_summary_report.csv" # Changed to a generic name

# --- Simulation Parameters ---
SIMULATION_ADDRESS = "San Francisco, California, USA"
NETWORK_TYPE = "all"
SIMULATION_ITERATIONS = 1000 # For simulate_traffic
SIMULATION_HOURS_TOTAL = 1 # For the main Pygame loop duration in hours
AGENT_SPEED = 0.00005 # Agent movement speed
AGENT_EATING_TIME_MIN_SECONDS = 120
AGENT_EATING_TIME_MAX_SECONDS = 600
AGENT_DEFAULT_WAITING_TIME_SECONDS = 300

# --- Pygame Display ---
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
FPS = 60 # Frames per second for Pygame clock

# --- Base Store Capacities (can be expanded) ---
BASE_STORE_CAPACITY = {
    'McDonald\'s': 100, 'Pret a Manger': 105, 'Shake Shack': 95,
    'Starbucks': 94, 'Taco Bell': 96, 'Wendy\'s': 94,
    'Popeyes': 98, 'Five Guys': 99, 'Raising Cane\'s': 90,
    'Dominos': 90, 'Chick-fil-A': 92
    # Add other base brands and their typical capacities
}

# --- Income Group Labels (from trade_area_profiling.csv) ---
INCOME_GROUP_LABELS = [
    "BLOCK_Estimate!!Total:!!Less than $10,000", "$10,000 to $14,999",
    "$15,000 to $19,999", "$20,000 to $24,999", "$25,000 to $29,999",
    "$30,000 to $34,999", "$35,000 to $39,999", "$40,000 to $44,999",
    "$45,000 to $49,999", "$50,000 to $59,999", "$60,000 to $74,999",
    "$75,000 to $99,999", "$100,000 to $124,999", "$125,000 to $149,999",
    "$150,000 to $199,999", "$200,000 or more"
]

# ## 3. OSM Data Retrieval and Graph Utilities

def get_graph_from_place(address, network_type="all"):
    """Fetches the OSM graph from an address."""
    try:
        graph = ox.graph_from_place(address, network_type=network_type)
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)
        return nodes_gdf, edges_gdf, graph
    except Exception as e:
        print(f"Error creating graph from place '{address}': {e}")
        return None, None, None

def get_graph_from_point(point, radius, network_type="all"):
    """Fetches the OSM graph from a central point and radius."""
    try:
        graph = ox.graph_from_point(point, dist=radius, network_type=network_type)
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)
        return nodes_gdf, edges_gdf, graph
    except Exception as e:
        print(f"Error creating graph from point '{point}': {e}")
        return None, None, None

def calculate_graph_area(graph):
    """Computes an approximate bounding box area of the graph in square kilometers."""
    if graph is None:
        print("Graph is None, cannot calculate area.")
        return np.nan
    try:
        nodes_gdf, _ = ox.graph_to_gdfs(graph, edges=True) # Ensure nodes are fetched
        if nodes_gdf.empty:
            print("Nodes GeoDataFrame is empty, cannot calculate area.")
            return np.nan

        # Use union_all() instead of deprecated unary_union
        min_lon, min_lat, max_lon, max_lat = nodes_gdf.union_all().bounds
        polygon = Polygon([(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat)])
        
        # Dynamically determine UTM zone based on longitude
        central_lon = (min_lon + max_lon) / 2
        utm_zone = int((central_lon + 180) / 6) + 1
        # Assuming Northern Hemisphere. Add logic for Southern if needed.
        epsg_code = f"epsg:326{utm_zone:02d}" 

        transformer = pyproj.Transformer.from_crs("epsg:4326", epsg_code, always_xy=True)
        x_coords, y_coords = polygon.exterior.xy
        x_utm, y_utm = transformer.transform(x_coords, y_coords)

        area_m2 = 0.5 * np.abs(np.dot(x_utm, np.roll(y_utm, 1)) - np.dot(y_utm, np.roll(x_utm, 1)))
        area_km2 = area_m2 / 1_000_000
        return area_km2
    except Exception as e:
        print(f"Error calculating area: {e}")
        return np.nan

def plot_osm_graph(nodes_gdf, edges_gdf, address_name):
    """Plots the OSM graph using Matplotlib."""
    if nodes_gdf is None or edges_gdf is None:
        print("No graph data to plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 10))
    edges_gdf.plot(ax=ax, linewidth=1, color="black", alpha=0.7, label="Roads")
    nodes_gdf.plot(ax=ax, markersize=5, color="red", alpha=0.6, label="Intersections")
    ax.set_title(f"Street Network of {address_name}")
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def geo_to_pixel(lat, lon, screen_w, screen_h, map_bounds, map_center_lat, map_center_lon, zoom):
    """Converts geographic coordinates to pixel coordinates, considering map center and zoom."""
    min_map_lat, min_map_lon, max_map_lat, max_map_lon = map_bounds
    
    # Calculate the effective "width" and "height" of the map in degrees at current zoom
    map_width_deg = (max_map_lon - min_map_lon) / zoom
    map_height_deg = (max_map_lat - min_map_lat) / zoom

    # Calculate the new visible bounds based on center and zoom
    visible_min_lon = map_center_lon - map_width_deg / 2
    visible_max_lon = map_center_lon + map_width_deg / 2
    visible_min_lat = map_center_lat - map_height_deg / 2
    visible_max_lat = map_center_lat + map_height_deg / 2
    
    if visible_max_lon == visible_min_lon or visible_max_lat == visible_min_lat: # Avoid division by zero
        return screen_w // 2, screen_h // 2

    x_pixel = ((lon - visible_min_lon) / (visible_max_lon - visible_min_lon)) * screen_w
    y_pixel = screen_h - (((lat - visible_min_lat) / (visible_max_lat - visible_min_lat)) * screen_h) # Y is inverted

    return int(x_pixel), int(y_pixel)

# ## 4. Data Loading and Preprocessing Functions

def load_and_prepare_restaurant_data(osr_path, lat_lon_path):
    """Loads and prepares the main restaurant DataFrame."""
    try:
        df = pd.read_csv(osr_path)
    except FileNotFoundError:
        print(f"Error: {osr_path} not found.")
        return None
    
    df = rename_duplicate_brands(df.copy()) # Use .copy() to avoid SettingWithCopyWarning later
    
    try:
        df_lat_lon = pd.read_csv(lat_lon_path)
        df = df.merge(df_lat_lon, on=["placeId"], how="left", suffixes=('_osr', '_store'))
        # Prioritize lat/lon from lat_lon_stores.csv if available, else use OSR
        df['latitude'] = df['lat_store'].fillna(df['lat_osr'])
        df['longitude'] = df['lon_store'].fillna(df['lon_osr'])
        df.drop(columns=['lat_osr', 'lon_osr', 'lat_store', 'lon_store', 'brand_store'], inplace=True, errors='ignore')
        df.rename(columns={'brand_osr': 'brand'}, inplace=True)

    except FileNotFoundError:
        print(f"Warning: {lat_lon_path} not found. Using coordinates from OSR.csv only.")
        df.rename(columns={'lat': 'latitude', 'lon': 'longitude'}, inplace=True)

    df['capacity'] = df.apply(lambda row: assign_capacity(row['brand'], row.get('capacity')), axis=1)
    df["current_occupancy"] = 0
    df["waiting_time"] = 0 # Initialize waiting time
    df["cleaned_brand"] = df['brand'].str.strip().str.lower() # For easier matching
    return df

def rename_duplicate_brands(df_to_rename):
    """Renames duplicate brands by appending a counter."""
    counts = {}
    new_brands = []
    for brand in df_to_rename['brand']:
        base_brand = brand.split(' ')[0] # Consider cases like "Subway 1" if original data has them
        if brand in counts:
            counts[brand] += 1
            new_brands.append(f"{brand} {counts[brand]}")
        elif df_to_rename['brand'].value_counts()[brand] > 1: # It's a duplicate but first encounter
             counts[brand] = 1 
             new_brands.append(f"{brand} {counts[brand]}")
        else: # Not a duplicate or already handled if it was part of an original numbered brand
            counts[brand] = 0 
            new_brands.append(brand)
    df_to_rename['brand'] = new_brands
    return df_to_rename
    
def assign_capacity(brand_name, current_capacity):
    """Assigns capacity based on brand name, using existing if reasonable or base/random."""
    # If current_capacity is valid (e.g. not NaN and within a reasonable range), prefer it.
    if pd.notna(current_capacity) and 5 < current_capacity < 500: # Assuming capacities are reasonable
        return int(current_capacity)
    
    base_brand = brand_name.split(' ')[0] # Handle duplicates like 'McDonald's 1'
    return BASE_STORE_CAPACITY.get(base_brand, random.randint(15, 25)) # Default random if not in base list

def load_trade_area_profile(tap_path):
    """Loads trade area profiling data."""
    try:
        tap_df = pd.read_csv(tap_path)
        return tap_df
    except FileNotFoundError:
        print(f"Error: {tap_path} not found.")
        return None

# ## 5. Traffic and Agent Generation Logic

footfall_report_data = [] # Use a list of dicts for easier appending

def track_footfall(hour, minute, restaurant_brand):
    """Logs footfall, appending to a list of dictionaries."""
    global footfall_report_data
    footfall_report_data.append({"Hour": hour, "Minute": minute, "Restaurant": restaurant_brand, "FootfallIncrement": 1})

def save_footfall_report_hourly(hour):
    """Aggregates and saves the hourly footfall report."""
    global footfall_report_data
    if not footfall_report_data:
        return

    hourly_df = pd.DataFrame(footfall_report_data)
    # Filter for the current hour before aggregation
    current_hour_df = hourly_df[hourly_df['Hour'] == hour]
    if not current_hour_df.empty:
        aggregated_report = current_hour_df.groupby(["Hour", "Restaurant"])["FootfallIncrement"].sum().reset_index()
        aggregated_report.rename(columns={'FootfallIncrement': 'Footfall'}, inplace=True)
        
        # Append to CSV
        try:
            mode = 'a' if os.path.exists(HOURLY_FOOTFALL_REPORT_PATH) else 'w'
            header = not os.path.exists(HOURLY_FOOTFALL_REPORT_PATH)
            aggregated_report.to_csv(HOURLY_FOOTFALL_REPORT_PATH, mode=mode, header=header, index=False)
        except Exception as e:
            print(f"Error saving hourly footfall report: {e}")
    
    # Clear data for the next hour or keep only recent data if memory is an issue
    footfall_report_data = [entry for entry in footfall_report_data if entry['Hour'] != hour]


def create_lambda_df(restaurants_df, num_hours=24):
    """Creates a DataFrame of lambda values for Poisson simulation."""
    # Example: Lambda is proportional to popularity, with a peak around lunch/dinner
    # This is a placeholder; a more sophisticated model would be needed for real-world scenarios
    lambda_data = {}
    hourly_multipliers = [0.1,0.1,0.1,0.1,0.1,0.2,0.3,0.5,0.7,0.8,0.9,1.0,0.9,0.8,0.7,0.6,0.8,0.9,1.0,0.8,0.6,0.4,0.2,0.1] # Simple 24h pattern

    for _, row in restaurants_df.iterrows():
        base_lambda = row['popularity'] * 2 # Scale popularity; adjust as needed
        hourly_lambdas = [base_lambda * mult for mult in hourly_multipliers]
        lambda_data[row['brand']] = hourly_lambdas
        
    lf_df = pd.DataFrame.from_dict(lambda_data, orient='index', columns=range(num_hours))
    return lf_df

def simulate_hourly_traffic(lambda_df, iterations=100):
    """Simulates hourly footfall for restaurants using Poisson distribution."""
    if lambda_df is None or lambda_df.empty:
        print("Lambda DataFrame is empty. Cannot simulate traffic.")
        return None
        
    unique_places = lambda_df.index.tolist()
    num_hours = len(lambda_df.columns)

    all_results = np.zeros((iterations, len(unique_places), num_hours))

    for m in tqdm(range(iterations), desc="Simulating Hourly Traffic", leave=False):
        lf_copy = lambda_df.copy()
        for p_idx, place in enumerate(unique_places):
            for hour_col in range(num_hours):
                mean_lambda = lambda_df.loc[place, hour_col]
                if pd.notna(mean_lambda) and mean_lambda > 0:
                    r = poisson.rvs(mu=mean_lambda)
                    lf_copy.loc[place, hour_col] = r
                else:
                    lf_copy.loc[place, hour_col] = 0 # Handle NaN or non-positive lambdas
            
        all_results[m, :, :] = lf_copy.values

    average_results_array = np.mean(all_results, axis=0)
    average_hourly_footfall_df = pd.DataFrame(average_results_array, index=unique_places, columns=range(num_hours))
    return average_hourly_footfall_df

def get_age_group_range(age_group_val):
    """Maps age to a descriptive age group range string."""
    # Corrected logic for age groups
    if 0 <= age_group_val <= 17: return None # Or a specific "Under 18" category if needed
    elif 18 <= age_group_val <= 20: return '18 to 20 years' # Or map to '21 to 24 years' if intended
    elif 21 <= age_group_val <= 24: return '21 to 24 years'
    elif 25 <= age_group_val <= 29: return '25 to 29 years'
    # ... (add all other elif blocks from original code for completeness) ...
    elif 30 <= age_group_val <= 34: return '30 to 34 years'
    elif 35 <= age_group_val <= 39: return '35 to 39 years'
    elif 40 <= age_group_val <= 44: return '40 to 44 years'
    elif 45 <= age_group_val <= 49: return '45 to 49 years'
    elif 50 <= age_group_val <= 54: return '50 to 54 years'
    elif 55 <= age_group_val <= 59: return '55 to 59 years'
    elif 60 <= age_group_val <= 61: return '60 and 61 years'
    elif 62 <= age_group_val <= 64: return '62 to 64 years'
    elif 65 <= age_group_val <= 66: return '65 and 66 years'
    elif 67 <= age_group_val <= 69: return '67 to 69 years'
    elif 70 <= age_group_val <= 74: return '70 to 74 years'
    elif 75 <= age_group_val <= 79: return '75 to 79 years'
    elif 80 <= age_group_val <= 84: return '80 to 84 years'
    elif age_group_val >= 85: return '85 years and over'
    else: return None

def get_random_income(agent_age, agent_gender, trade_area_df):
    """Assigns a random income value based on age group and gender using trade_area_df."""
    age_group_range_str = get_age_group_range(agent_age)
    if not age_group_range_str: return None

    gender_column_name = f'BLOCK_Estimate!!Total:!!{agent_gender}:!!{age_group_range_str}'
    if gender_column_name not in trade_area_df.columns:
        # Try matching without the "BLOCK_Estimate!!Total:!!" prefix for income_group_labels if necessary
        income_columns_for_gender_age = [col for col in trade_area_df.columns if agent_gender in col and age_group_range_str in col and "Estimate!!Total:!!" in col]
        if not income_columns_for_gender_age : return None # No matching income columns
        
        # This part becomes tricky if the exact column name isn't present.
        # For simplicity, let's assume income_group_labels match the *endings* of columns in trade_area_df
        # and that trade_area_df is indexed by something that aligns with INCOME_GROUP_LABELS's intent.
        # A safer approach would be to ensure trade_area_df has standardized income group rows.
        # Let's use the first matching column group to extract income distribution if direct match fails
        # This is a simplified placeholder logic.
        relevant_tap_series = trade_area_df[income_columns_for_gender_age[0]].copy() # Example: take first matching group
        relevant_tap_series.index = [label.split(':!!')[-1].strip() for label in INCOME_GROUP_LABELS] # Remap index for matching
        
    else:
        # This assumes trade_area_df is indexed by the full income group label string
        relevant_tap_series = trade_area_df[gender_column_name].copy()
        relevant_tap_series.index = [label.split(':!!')[-1].strip() for label in INCOME_GROUP_LABELS]


    income_distribution_values = relevant_tap_series.dropna()
    if income_distribution_values.empty or income_distribution_values.sum() == 0: return None

    probabilities = income_distribution_values / income_distribution_values.sum()
    
    # Ensure indices of probabilities match INCOME_GROUP_LABELS format for choice
    valid_income_groups = [ig.replace("BLOCK_Estimate!!Total:!!", "").strip() for ig in INCOME_GROUP_LABELS if ig.replace("BLOCK_Estimate!!Total:!!", "").strip() in probabilities.index]
    valid_probabilities = probabilities[valid_income_groups].fillna(0) # Get corresponding probabilities
    if valid_probabilities.sum() == 0 : return None
    valid_probabilities = valid_probabilities / valid_probabilities.sum() # Re-normalize

    if not valid_income_groups: return None

    chosen_income_group_str = np.random.choice(valid_income_groups, p=valid_probabilities)
    
    # Convert income group string to a random integer within its range
    if chosen_income_group_str == 'Less than $10,000': return random.randint(0, 9999)
    elif chosen_income_group_str == '$10,000 to $14,999': return random.randint(10000, 14999)
    # ... (add all other elif blocks from original for income mapping) ...
    elif chosen_income_group_str == '$15,000 to $19,999': return random.randint(15000, 19999)
    elif chosen_income_group_str == '$20,000 to $24,999': return random.randint(20000, 24999)
    elif chosen_income_group_str == '$25,000 to $29,999': return random.randint(25000, 29999)
    elif chosen_income_group_str == '$30,000 to $34,999': return random.randint(30000, 34999)
    elif chosen_income_group_str == '$35,000 to $39,999': return random.randint(35000, 39999)
    elif chosen_income_group_str == '$40,000 to $44,999': return random.randint(40000, 44999)
    elif chosen_income_group_str == '$45,000 to $49,999': return random.randint(45000, 49999)
    elif chosen_income_group_str == '$50,000 to $59,999': return random.randint(50000, 59999)
    elif chosen_income_group_str == '$60,000 to $74,999': return random.randint(60000, 74999)
    elif chosen_income_group_str == '$75,000 to $99,999': return random.randint(75000, 99999)
    elif chosen_income_group_str == '$100,000 to $124,999': return random.randint(100000, 124999)
    elif chosen_income_group_str == '$125,000 to $149,999': return random.randint(125000, 149999)
    elif chosen_income_group_str == '$150,000 to $199,999': return random.randint(150000, 199999)
    elif chosen_income_group_str == '$200,000 or more': return random.randint(200000, 500000) # Capped for example
    return None

def generate_agent_demographics_data(num_agents, trade_area_df):
    """Generates a DataFrame with demographic data for a given number of agents."""
    if trade_area_df is None:
        print("Trade area profile data not available. Generating agents with default demographics.")
        data = [{'Agent': i, 'Age_Group': random.randint(18,70), 'Gender': random.choice(['Male', 'Female']), 'Income': random.randint(20000,100000)} for i in range(num_agents)]
        return pd.DataFrame(data)

    agent_demo_list = []
    for i in range(num_agents):
        age = random.randint(18, 85) # Random age
        gender = random.choice(['Male', 'Female'])
        income = get_random_income(age, gender, trade_area_df)
        agent_demo_list.append({'Agent': i, 'Age_Group': age, 'Gender': gender, 'Income': income})
    return pd.DataFrame(agent_demo_list)

def map_placeid_to_brand(target_place_id, restaurants_df):
    """Maps a placeId to its brand name from the restaurants DataFrame."""
    # Assumes 'restaurants_df' has 'placeId' and 'brand' columns
    # and 'brand' contains unique brand names from rename_duplicate_brands
    brand_series = restaurants_df[restaurants_df['placeId'] == target_place_id]['brand']
    if not brand_series.empty:
        return brand_series.iloc[0]
    return "Unknown Brand"


def generate_agents_for_hour(hour, hourly_footfall_df, osm_nodes_gdf, trade_area_df, restaurants_df_main):
    """Generates agents for a specific hour based on footfall data."""
    agents_for_this_hour = []
    total_agents_generated = 0
    agent_demographics_summary = []
    agent_id_offset = Agent.agent_counter # Use the class counter for unique IDs

    if hourly_footfall_df is None or osm_nodes_gdf is None or trade_area_df is None or restaurants_df_main is None:
        print("Missing data for agent generation.")
        return agents_for_this_hour, 0, agent_demographics_summary

    # Ensure hour is treated as integer for .loc access if columns are integers
    hour_str = str(hour) # If columns are strings '0', '1', etc.
    if hour not in hourly_footfall_df.columns and hour_str in hourly_footfall_df.columns:
        current_hour_footfall = hourly_footfall_df[hour_str]
    elif hour in hourly_footfall_df.columns:
        current_hour_footfall = hourly_footfall_df[hour]
    else:
        print(f"Hour {hour} not found in hourly_footfall_df columns.")
        return agents_for_this_hour, 0, agent_demographics_summary

    for brand_name, num_agents_to_spawn in current_hour_footfall.items():
        num_agents_to_spawn = int(round(num_agents_to_spawn))
        if num_agents_to_spawn <= 0:
            continue
        
        total_agents_generated += num_agents_to_spawn

        # Generate demographics for these agents
        agent_demographics_for_brand = generate_agent_demographics_data(num_agents_to_spawn, trade_area_df)

        spawn_intervals_seconds = np.linspace(0, 3540, num_agents_to_spawn, endpoint=False) # Spread over 59 minutes (3540 seconds)

        for i in range(num_agents_to_spawn):
            random_node = osm_nodes_gdf.sample(n=1).iloc[0]
            lat, lon = random_node['y'], random_node['x']
            
            agent_specific_data = agent_demographics_for_brand.iloc[i].to_dict()
            agent_specific_data['store_preference'] = brand_name 
            agent_specific_data['Agent_Global_ID'] = agent_id_offset + total_agents_generated - num_agents_to_spawn + i # Ensure unique ID
            
            agent = Agent(
                store_preference=brand_name,
                lat=lat,
                lon=lon,
                agent_data=agent_specific_data,
                osm_nodes_gdf=osm_nodes_gdf,
                restaurants_df=restaurants_df_main, # Pass the main restaurants_df
                # global_map_bounds, screen_width, screen_height, map_center_lat, map_center_lon, current_zoom for geo_to_pixel
                # These will be accessed via agent.sim_pygame_renderer in a structured setup
            )
            spawn_second = int(round(spawn_intervals_seconds[i]))
            spawn_minute = spawn_second // 60
            
            agents_for_this_hour.append((agent, spawn_minute, hour)) # (agent_object, spawn_minute_of_hour, hour_of_day)
            agent_demographics_summary.append(agent_specific_data)
            
    return agents_for_this_hour, total_agents_generated, agent_demographics_summary


# ## 6. Agent Archetypes and Agent Class

class AgentArchetype:
    def __init__(self, name, store_preference_weight=0.5, distance_weight=0.5):
        self.name = name
        self.store_preference_weight = store_preference_weight
        self.distance_weight = distance_weight
        # self.chosen_restaurant_details = None # Stores the chosen restaurant's full row data

    def choose_restaurant(self, agent_instance, available_restaurants_df):
        """Choose a restaurant based on archetype-specific logic."""
        # Ensure 'latitude' and 'longitude' columns exist
        if 'latitude' not in available_restaurants_df.columns or \
           'longitude' not in available_restaurants_df.columns:
            print("Error: Restaurants DataFrame missing latitude/longitude columns.")
            return None
            
        if available_restaurants_df.empty:
            print(f"Agent {agent_instance.agent_id}: No restaurants available to choose from.")
            return None

        distances = {}
        store_pref_scores = {}

        for _, restaurant_row in available_restaurants_df.iterrows():
            brand = restaurant_row['brand']
            rest_lat = restaurant_row['latitude']
            rest_lon = restaurant_row['longitude']
            
            if pd.isna(rest_lat) or pd.isna(rest_lon): # Skip if location is invalid
                continue

            distance_km = agent_instance.haversine(agent_instance.current_lon, agent_instance.current_lat, rest_lon, rest_lat)
            distances[brand] = distance_km
            store_pref_scores[brand] = 1 if brand == agent_instance.store_preference else 0
        
        if not distances: # No valid restaurants with locations
             print(f"Agent {agent_instance.agent_id}: No valid restaurants with locations found.")
             return None

        max_dist = max(distances.values()) if distances else 1.0
        min_dist = min(distances.values()) if distances else 0.0
        
        # Avoid division by zero if all distances are the same
        dist_range = max_dist - min_dist
        if dist_range == 0: dist_range = 1.0 # Or handle as all equally preferable/non-preferable by distance

        normalized_distance_scores = {
            brand: 1 - ((dist - min_dist) / dist_range) for brand, dist in distances.items()
        }

        restaurant_scores = {}
        for brand_key in distances: # Use keys from distances as they are validated
            score = (self.store_preference_weight * store_pref_scores.get(brand_key, 0)) + \
                    (self.distance_weight * normalized_distance_scores.get(brand_key, 0))
            restaurant_scores[brand_key] = score
        
        if not restaurant_scores:
            print(f"Agent {agent_instance.agent_id}: No restaurants scored.")
            return None

        best_restaurant_brand = max(restaurant_scores, key=restaurant_scores.get)
        chosen_restaurant_series = available_restaurants_df[available_restaurants_df['brand'] == best_restaurant_brand].iloc[0]
        return chosen_restaurant_series # Return the full row (Series) of the chosen restaurant

class YoungProfessional(AgentArchetype):
    def __init__(self):
        super().__init__("YoungProfessional", store_preference_weight=0.7, distance_weight=0.3)
    # choose_restaurant method is inherited from AgentArchetype

class Family(AgentArchetype):
    def __init__(self):
        super().__init__("Family", store_preference_weight=0.5, distance_weight=0.5)
    # choose_restaurant method is inherited from AgentArchetype

class Agent:
    agent_counter = 0 # Class variable for unique agent IDs

    def __init__(self, store_preference, lat, lon, agent_data, osm_nodes_gdf, restaurants_df, 
                 sim_pygame_renderer=None): # Added sim_pygame_renderer
        self.agent_id = Agent.agent_counter
        Agent.agent_counter += 1
        
        self.store_preference = store_preference
        self.initial_lat, self.initial_lon = lat, lon
        self.current_lat, self.current_lon = lat, lon
        
        self.agent_data = agent_data if agent_data else {} # Ensure agent_data is a dict
        self.archetype = self.assign_archetype()
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        self.osm_nodes_gdf = osm_nodes_gdf # GeoDataFrame of OSM nodes
        self.restaurants_df = restaurants_df # Main DataFrame with all restaurant info
        self.sim_pygame_renderer = sim_pygame_renderer # Reference to Pygame renderer for screen params

        self.speed = AGENT_SPEED
        self.target_node_osmid = None
        self.path_osmids = []
        self.path_index = 0
        
        self.chosen_restaurant_brand = None
        self.target_restaurant_lat = None
        self.target_restaurant_lon = None
        
        self.state = 'idle'  # idle, choosing, traveling, waiting, eating, done, re_evaluating
        self.time_spent_eating = 0
        self.time_spent_waiting = 0
        self.eating_duration = random.randint(AGENT_EATING_TIME_MIN_SECONDS, AGENT_EATING_TIME_MAX_SECONDS)
        self.max_wait_duration = AGENT_DEFAULT_WAITING_TIME_SECONDS # Can be archetype specific
        
        # Get initial screen coordinates
        if self.sim_pygame_renderer:
            self.screen_x, self.screen_y = self.sim_pygame_renderer.geo_to_pixel_agent(self.current_lat, self.current_lon)


    def assign_archetype(self):
        age = self.agent_data.get('Age_Group', 30) # Default age if not provided
        # income = self.agent_data.get('Income', 50000) # Default income
        
        # Example: Simplified archetype assignment
        if 18 <= age <= 35:
            return YoungProfessional()
        else:
            return Family()

    def draw(self, screen):
        if self.sim_pygame_renderer:
             self.screen_x, self.screen_y = self.sim_pygame_renderer.geo_to_pixel_agent(self.current_lat, self.current_lon)
             pygame.draw.circle(screen, self.color, (self.screen_x, self.screen_y), 5)


    def haversine(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 6371 * 2 * asin(sqrt(a)) # Earth radius in km

    def _choose_restaurant_action(self, osm_graph):
        chosen_restaurant_series = self.archetype.choose_restaurant(self, self.restaurants_df)
        
        if chosen_restaurant_series is not None and not chosen_restaurant_series.empty:
            self.chosen_restaurant_brand = chosen_restaurant_series['brand']
            self.target_restaurant_lat = chosen_restaurant_series['latitude']
            self.target_restaurant_lon = chosen_restaurant_series['longitude']

            if pd.isna(self.target_restaurant_lat) or pd.isna(self.target_restaurant_lon):
                print(f"Agent {self.agent_id}: Chosen restaurant '{self.chosen_restaurant_brand}' has no valid coordinates.")
                self.state = 're_evaluating' # Or 'done' if no other option
                return

            orig_node_osmid = ox.nearest_nodes(osm_graph, self.current_lon, self.current_lat)
            target_node_osmid_restaurant = ox.nearest_nodes(osm_graph, self.target_restaurant_lon, self.target_restaurant_lat)
            
            try:
                self.path_osmids = ox.shortest_path(osm_graph, orig_node_osmid, target_node_osmid_restaurant, weight='length')
                if self.path_osmids:
                    self.path_index = 0
                    self.target_node_osmid = self.path_osmids[0] # First node in path is current or next
                    self.state = 'traveling'
                    # print(f"Agent {self.agent_id} ({self.archetype.name}) chose {self.chosen_restaurant_brand}. Path found.")
                else: # No path
                    print(f"Agent {self.agent_id}: No path to {self.chosen_restaurant_brand}. Re-evaluating.")
                    self.state = 're_evaluating'
            except Exception as e:
                print(f"Agent {self.agent_id}: Error finding path to {self.chosen_restaurant_brand}: {e}. Re-evaluating.")
                self.state = 're_evaluating'
        else:
            print(f"Agent {self.agent_id}: Could not choose a restaurant. Will retry or terminate.")
            self.state = 'done' # Or re-evaluate after a delay

    def _traveling_action(self, osm_graph):
        if not self.path_osmids or self.path_index >= len(self.path_osmids):
            # Reached the node nearest to the restaurant, now move to exact restaurant location
            if self.target_restaurant_lat is not None:
                dx = self.target_restaurant_lon - self.current_lon
                dy = self.target_restaurant_lat - self.current_lat
                dist_to_rest = sqrt(dx**2 + dy**2)

                if dist_to_rest > self.speed:
                    self.current_lon += (dx / dist_to_rest) * self.speed
                    self.current_lat += (dy / dist_to_rest) * self.speed
                else: # Arrived at restaurant
                    self.current_lon, self.current_lat = self.target_restaurant_lon, self.target_restaurant_lat
                    # print(f"Agent {self.agent_id} arrived at {self.chosen_restaurant_brand}.")
                    
                    # Check capacity
                    restaurant_idx = self.restaurants_df[self.restaurants_df['brand'] == self.chosen_restaurant_brand].index
                    if not restaurant_idx.empty:
                        idx = restaurant_idx[0]
                        if self.restaurants_df.loc[idx, 'current_occupancy'] < self.restaurants_df.loc[idx, 'capacity']:
                            self.restaurants_df.loc[idx, 'current_occupancy'] += 1
                            self.state = 'eating'
                            self.time_spent_eating = 0
                            track_footfall(self.sim_pygame_renderer.current_hour, self.sim_pygame_renderer.current_minute, self.chosen_restaurant_brand)
                        else:
                            self.state = 'waiting' # Wait or re-evaluate
                            self.time_spent_waiting = 0
                            # print(f"Agent {self.agent_id}: {self.chosen_restaurant_brand} is full. Waiting.")
                    else:
                        print(f"Error: Restaurant {self.chosen_restaurant_brand} not found in df for agent {self.agent_id}")
                        self.state = 'done'
            else:
                self.state = 'done' # Should not happen if target_restaurant_lat/lon is set
            return

        # Move along OSM path
        target_node_osmid_in_path = self.path_osmids[self.path_index]
        
        # Check if target_node_osmid_in_path is valid
        if target_node_osmid_in_path not in self.osm_nodes_gdf.index:
            print(f"Agent {self.agent_id}: Invalid node OSMID {target_node_osmid_in_path} in path. Re-evaluating.")
            self.state = 're_evaluating'
            return

        target_node_data = self.osm_nodes_gdf.loc[target_node_osmid_in_path]
        
        next_lat_path, next_lon_path = target_node_data['y'], target_node_data['x']
        
        dx = next_lon_path - self.current_lon
        dy = next_lat_path - self.current_lat
        distance = sqrt(dx**2 + dy**2)

        if distance > self.speed:
            self.current_lon += (dx / distance) * self.speed
            self.current_lat += (dy / distance) * self.speed
        else:
            self.current_lon, self.current_lat = next_lon_path, next_lat_path
            self.path_index += 1
            if self.path_index >= len(self.path_osmids): # Reached end of OSM path
                # Final move to exact restaurant location handled at start of this block
                pass


    def _waiting_action(self):
        self.time_spent_waiting += 1 # Assuming 1 tick = 1 second for now
        if self.time_spent_waiting > self.max_wait_duration:
            # print(f"Agent {self.agent_id} waited too long at {self.chosen_restaurant_brand}. Re-evaluating.")
            self.state = 're_evaluating'
            return

        restaurant_idx = self.restaurants_df[self.restaurants_df['brand'] == self.chosen_restaurant_brand].index
        if not restaurant_idx.empty:
            idx = restaurant_idx[0]
            if self.restaurants_df.loc[idx, 'current_occupancy'] < self.restaurants_df.loc[idx, 'capacity']:
                self.restaurants_df.loc[idx, 'current_occupancy'] += 1
                self.state = 'eating'
                self.time_spent_eating = 0
                track_footfall(self.sim_pygame_renderer.current_hour, self.sim_pygame_renderer.current_minute, self.chosen_restaurant_brand)
                # print(f"Agent {self.agent_id} done waiting, entering {self.chosen_restaurant_brand}.")


    def _eating_action(self):
        self.time_spent_eating += 1 # Assuming 1 tick = 1 second
        if self.time_spent_eating >= self.eating_duration:
            restaurant_idx = self.restaurants_df[self.restaurants_df['brand'] == self.chosen_restaurant_brand].index
            if not restaurant_idx.empty:
                idx = restaurant_idx[0]
                self.restaurants_df.loc[idx, 'current_occupancy'] -= 1
            # print(f"Agent {self.agent_id} finished eating at {self.chosen_restaurant_brand}.")
            self.state = 'done'
            
    def _re_evaluating_action(self, osm_graph):
        # print(f"Agent {self.agent_id} is re-evaluating restaurant choice.")
        # Potentially remove the full restaurant from consideration for a while
        # For now, just choose another one (might choose the same if it's still highest score and now has capacity)
        self.chosen_restaurant_brand = None # Reset
        self.target_restaurant_lat = None
        self.target_restaurant_lon = None
        self.path_osmids = []
        self.path_index = 0
        self._choose_restaurant_action(osm_graph)


    def update(self, osm_graph):
        """Updates the agent's state and position."""
        if self.state == 'idle':
            self.state = 'choosing' # Transition to choosing state

        if self.state == 'choosing':
            self._choose_restaurant_action(osm_graph)
        elif self.state == 'traveling':
            self._traveling_action(osm_graph)
        elif self.state == 'waiting':
            self._waiting_action()
        elif self.state == 'eating':
            self._eating_action()
        elif self.state == 're_evaluating':
            self._re_evaluating_action(osm_graph)
        
        # Update screen coordinates if renderer is available
        if self.sim_pygame_renderer:
             self.screen_x, self.screen_y = self.sim_pygame_renderer.geo_to_pixel_agent(self.current_lat, self.current_lon)


# ## 7. Pygame UI Helper Functions
class PygameRenderer:
    def __init__(self, screen_width, screen_height, initial_map_bounds):
        pygame.init()
        pygame.font.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Restaurant Simulation")
        
        self.font = pygame.font.SysFont('Arial', 20)
        self.brand_colors = {}
        
        self.map_bounds = initial_map_bounds # min_lat, min_lon, max_lat, max_lon
        self.min_lat, self.min_lon, self.max_lat, self.max_lon = initial_map_bounds
        
        self.map_center_lat = (self.min_lat + self.max_lat) / 2
        self.map_center_lon = (self.min_lon + self.max_lon) / 2
        self.zoom_level = 1.0

        self.dragging = False
        self.drag_start_pos = (0,0)

        self.current_hour = 0
        self.current_minute = 0
        self.current_second = 0

    def geo_to_pixel_agent(self, lat, lon): # Simplified for agent, uses renderer's state
        return geo_to_pixel(lat, lon, self.screen_width, self.screen_height, 
                             self.map_bounds, self.map_center_lat, self.map_center_lon, self.zoom_level)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False # Signal to quit

            elif event.type == pygame.MOUSEWHEEL:
                mouse_geo_lat, mouse_geo_lon = self.pixel_to_geo(pygame.mouse.get_pos())
                
                if event.y > 0: self.zoom_level *= 1.1
                elif event.y < 0: self.zoom_level /= 1.1
                self.zoom_level = max(0.1, min(self.zoom_level, 20.0)) # Zoom limits

                # Adjust map_center to keep mouse geo point fixed
                new_mouse_screen_x, new_mouse_screen_y = self.geo_to_pixel_agent(mouse_geo_lat, mouse_geo_lon)
                mouse_original_screen_x, mouse_original_screen_y = pygame.mouse.get_pos()

                # Calculate the shift needed in world coordinates
                delta_screen_x = mouse_original_screen_x - new_mouse_screen_x
                delta_screen_y = mouse_original_screen_y - new_mouse_screen_y

                map_width_deg = (self.max_lon - self.min_lon) / self.zoom_level
                map_height_deg = (self.max_lat - self.min_lat) / self.zoom_level
                
                self.map_center_lon -= (delta_screen_x / self.screen_width) * map_width_deg
                self.map_center_lat += (delta_screen_y / self.screen_height) * map_height_deg


            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    self.dragging = True
                    self.drag_start_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: self.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    current_pos = event.pos
                    dx_pixel = current_pos[0] - self.drag_start_pos[0]
                    dy_pixel = current_pos[1] - self.drag_start_pos[1]
                    
                    map_width_deg_at_zoom = (self.max_lon - self.min_lon) / self.zoom_level
                    map_height_deg_at_zoom = (self.max_lat - self.min_lat) / self.zoom_level

                    self.map_center_lon -= (dx_pixel / self.screen_width) * map_width_deg_at_zoom
                    self.map_center_lat += (dy_pixel / self.screen_height) * map_height_deg_at_zoom
                    
                    self.drag_start_pos = current_pos
        return True

    def pixel_to_geo(self, screen_pos):
        """Converts pixel coordinates back to geographic coordinates."""
        screen_x, screen_y = screen_pos
        
        map_width_deg = (self.max_lon - self.min_lon) / self.zoom_level
        map_height_deg = (self.max_lat - self.min_lat) / self.zoom_level

        visible_min_lon = self.map_center_lon - map_width_deg / 2
        visible_min_lat = self.map_center_lat - map_height_deg / 2
        
        lon = visible_min_lon + (screen_x / self.screen_width) * map_width_deg
        lat = visible_min_lat + ((self.screen_height - screen_y) / self.screen_height) * map_height_deg
        return lat, lon

    def display_time(self, hour, minute, second):
        time_str = f'Hour: {hour:02d} Min: {minute:02d} Sec: {second:02d}'
        text = self.font.render(time_str, True, (0, 0, 0))
        self.screen.blit(text, (10, 10))
        self.current_hour, self.current_minute, self.current_second = hour, minute, second


    def draw_legend(self, restaurants_df):
        y_offset = 40
        # Ensure brand_colors is populated for brands actually in restaurants_df
        for brand in restaurants_df['brand'].unique():
            if brand not in self.brand_colors:
                 self.brand_colors[brand] = (random.randint(50,200), random.randint(50,200), random.randint(50,200))
        
        for brand, color in self.brand_colors.items():
            if y_offset + 20 > self.screen_height: break # Avoid drawing off-screen
            pygame.draw.rect(self.screen, color, (10, y_offset, 20, 20))
            text_surface = self.font.render(brand, True, (0,0,0))
            self.screen.blit(text_surface, (40, y_offset))
            y_offset += 30
            
    def show_popup(self, text, pos):
        popup_surf = self.font.render(text, True, (0,0,0), (220,220,220))
        popup_rect = popup_surf.get_rect(midleft = (pos[0] + 15, pos[1]))
        # Ensure popup is within screen bounds
        popup_rect.clamp_ip(self.screen.get_rect())
        self.screen.blit(popup_surf, popup_rect)

    def draw_osm_edges(self, edges_gdf):
        if edges_gdf is None: return
        for _, edge in edges_gdf.iterrows():
            if 'geometry' in edge and edge['geometry'] is not None:
                geom = edge['geometry']
                if geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                    pixel_coords = [self.geo_to_pixel_agent(c[1], c[0]) for c in coords]
                    if len(pixel_coords) > 1:
                        pygame.draw.lines(self.screen, (200,200,200), False, pixel_coords, 1) # Lighter roads
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        coords = list(line.coords)
                        pixel_coords = [self.geo_to_pixel_agent(c[1], c[0]) for c in coords]
                        if len(pixel_coords) > 1:
                           pygame.draw.lines(self.screen, (200,200,200), False, pixel_coords, 1)

    def draw_stores(self, restaurants_df):
        store_render_info = [] # For hover detection
        if restaurants_df is None: return store_render_info

        for _, store in restaurants_df.iterrows():
            lat, lon = store["latitude"], store["longitude"]
            if pd.isna(lat) or pd.isna(lon): continue

            x, y = self.geo_to_pixel_agent(lat, lon)
            radius = 7 # Smaller radius for stores
            brand = store['brand']
            color = self.brand_colors.get(brand, (255,0,0)) # Default red
            
            pygame.draw.circle(self.screen, color, (x, y), radius)
            store_render_info.append({'brand': brand, 'rect': pygame.Rect(x-radius, y-radius, 2*radius, 2*radius),
                                     'capacity': store['capacity'], 'occupancy': store['current_occupancy']})
        return store_render_info

    def quit(self):
        pygame.quit()


# ## 8. Main Simulation Setup
print("Initializing simulation...")

# --- Load Data ---
restaurants_df = load_and_prepare_restaurant_data(OSR_CSV_PATH, LAT_LON_STORES_CSV_PATH)
trade_area_df = load_trade_area_profile(TRADE_AREA_PROFILING_CSV_PATH)

# --- Get OSM Graph Data ---
print(f"Fetching map data for {SIMULATION_ADDRESS}...")
osm_nodes_gdf, osm_edges_gdf, osm_graph = get_graph_from_place(SIMULATION_ADDRESS, NETWORK_TYPE)

map_bounds = None
if osm_graph:
    area_sq_km = calculate_graph_area(osm_graph)
    print(f"Map data loaded. Approximate Area: {area_sq_km:.2f} sq km")
    min_lon_map, min_lat_map, max_lon_map, max_lat_map = osm_nodes_gdf.union_all().bounds
    map_bounds = (min_lat_map, min_lon_map, max_lat_map, max_lon_map)
else:
    print("Failed to load map data. Exiting.")
    exit()

# --- Initialize Pygame Renderer ---
renderer = PygameRenderer(SCREEN_WIDTH, SCREEN_HEIGHT, map_bounds)

# --- Prepare Lambda Footfall Data ---
lf_df = create_lambda_df(restaurants_df) # Create lambdas based on popularity

# --- Global Simulation Variables ---
active_agents = []
all_agent_demographics_summary = [] # To store all demographics data
clock = pygame.time.Clock()
sim_hour, sim_minute, sim_second = 0, 0, 0
time_scale_factor = 360 # 1 real second = 360 sim seconds (6 sim minutes) -> 10 real minutes for 1 sim hour.

# ## 9. Main Simulation Loop

print("Starting Pygame simulation loop...")
running = True
hourly_footfall_df = None # Will be populated each hour

total_simulation_seconds = SIMULATION_HOURS_TOTAL * 3600
current_simulation_time_seconds = 0

# Pre-calculate initial agent screen positions by passing renderer to Agent constructor
Agent.agent_counter = 0 # Reset agent counter

try:
    while running and current_simulation_time_seconds < total_simulation_seconds :
        if not renderer.handle_events(): # If handle_events returns False, quit
            running = False
            break

        # --- Update Simulation Time ---
        sim_second += time_scale_factor / FPS 
        if sim_second >= 60:
            sim_minute += int(sim_second / 60)
            sim_second %= 60
        if sim_minute >= 60:
            sim_hour += int(sim_minute / 60)
            sim_minute %= 60
        if sim_hour >= 24:
            sim_hour %= 24 # Loop back to hour 0
            # Optionally save daily reports here
            save_footfall_report_hourly(23) # Save for the last hour of the day
            footfall_report_data = [] # Clear for new day

        current_simulation_time_seconds += time_scale_factor / FPS

        # --- Hourly Updates ---
        if sim_minute == 0 and int(sim_second) == 0 and (current_simulation_time_seconds == (time_scale_factor / FPS) or int(sim_second) < (time_scale_factor / FPS) ): # Trigger once at the start of the hour
            print(f"\n--- Hour {sim_hour:02d}:{sim_minute:02d} ---")
            hourly_footfall_df = simulate_hourly_traffic(lf_df, iterations=100) # Fewer iterations for speed in loop
            if hourly_footfall_df is not None:
                 # Generate new agents for the current hour
                new_agents_this_hour, num_spawned, demo_summary = generate_agents_for_hour(
                    sim_hour, hourly_footfall_df, osm_nodes_gdf, trade_area_df, restaurants_df
                )
                # Add renderer to agents being created
                for agent_obj, spawn_min, hr in new_agents_this_hour:
                    agent_obj.sim_pygame_renderer = renderer 
                    agent_obj.screen_x, agent_obj.screen_y = renderer.geo_to_pixel_agent(agent_obj.current_lat, agent_obj.current_lon)


                active_agents.extend(new_agents_this_hour)
                all_agent_demographics_summary.extend(demo_summary)
                print(f"Generated {num_spawned} agents for hour {sim_hour}.")
                save_footfall_report_hourly(sim_hour -1 if sim_hour > 0 else 23) # Save previous hour's data

        # --- Agent Updates & Spawning ---
        agents_to_activate_this_tick = []
        remaining_agents = []
        for agent_obj, spawn_min, hr in active_agents:
            if hr == sim_hour and spawn_min == sim_minute and agent_obj.state == 'idle':
                agents_to_activate_this_tick.append(agent_obj)
            else:
                remaining_agents.append((agent_obj, spawn_min, hr))
        
        active_agents = remaining_agents + [(ag, ag.agent_data.get("spawn_minute_of_hour",0) , ag.agent_data.get("hour_of_day",0)) for ag in agents_to_activate_this_tick if ag.state != 'done']
        
        agents_still_active = []
        for agent_obj, _, _ in active_agents: # Iterate through active ones
             if agent_obj.state != 'done':
                if agent_obj.sim_pygame_renderer is None: agent_obj.sim_pygame_renderer = renderer # Ensure renderer is set
                agent_obj.update(osm_graph)
                agents_still_active.append((agent_obj, agent_obj.agent_data.get("spawn_minute_of_hour",0), agent_obj.agent_data.get("hour_of_day",0))) # Use original spawn details
        active_agents = agents_still_active


        # --- Drawing ---
        renderer.screen.fill((255, 255, 255)) # White background
        renderer.draw_osm_edges(osm_edges_gdf)
        store_render_info = renderer.draw_stores(restaurants_df)
        
        for agent_obj, _, _ in active_agents: # Draw only active agents
            agent_obj.draw(renderer.screen)

        # --- UI Updates (Time, Legend, Popup) ---
        renderer.display_time(sim_hour, sim_minute, int(sim_second))
        # renderer.draw_legend(restaurants_df) # Drawing legend every frame can be slow
        
        mouse_pos = pygame.mouse.get_pos()
        hovered_store_info = None
        for store_info in store_render_info:
            if store_info['rect'].collidepoint(mouse_pos):
                hovered_store_info = store_info
                break
        if hovered_store_info:
            popup_txt = f"{hovered_store_info['brand']} (Occ: {hovered_store_info['occupancy']}/{hovered_store_info['capacity']})"
            renderer.show_popup(popup_txt, mouse_pos)

        pygame.display.flip()
        clock.tick(FPS)

except KeyboardInterrupt:
    print("Simulation interrupted by user.")
finally:
    print("Exiting Pygame and saving final reports...")
    renderer.quit()
    
    # Save final agent demographics summary
    if all_agent_demographics_summary:
        summary_df = pd.DataFrame(all_agent_demographics_summary)
        try:
            summary_df.to_csv(AGENT_SUMMARY_REPORT_PATH, index=False)
            print(f"Full agent demographics summary saved to {AGENT_SUMMARY_REPORT_PATH}")
        except Exception as e:
            print(f"Error saving full agent summary: {e}")
            
    # Save any remaining footfall data
    save_footfall_report_hourly(sim_hour) # Save for the current/last hour
    print("Simulation finished.")

# ## 10. Post-Simulation Analysis / Data Inspection
# Example: Inspecting the hourly footfall data for a specific hour (e.g., hour 11 if generated)
# This cell would be run *after* the simulation loop has generated 'average_hourly_footfall_df'
# or after 'HOURLY_FOOTFALL_REPORT_PATH' has been populated.

# If you want to inspect the data generated by simulate_hourly_traffic *within the loop*:
# You would need to store 'hourly_footfall_df' from the loop or load it from its saved CSV.

# Example: Load the hourly footfall report and inspect
try:
    generated_footfall_report = pd.read_csv(HOURLY_FOOTFALL_REPORT_PATH)
    print("\n--- Sample of Generated Hourly Footfall Report ---")
    print(generated_footfall_report.head())
    
    if not generated_footfall_report.empty and 11 in generated_footfall_report['Hour'].unique():
         print("\n--- Footfall for Hour 11 ---")
         print(generated_footfall_report[generated_footfall_report['Hour'] == 11])
    elif not generated_footfall_report.empty:
         print("\n--- Footfall for First Available Hour ---")
         first_hour = generated_footfall_report['Hour'].unique()[0]
         print(generated_footfall_report[generated_footfall_report['Hour'] == first_hour])
    else:
        print("\nHourly footfall report is empty or hour 11 data not found.")

except FileNotFoundError:
    print(f"\n{HOURLY_FOOTFALL_REPORT_PATH} not found. Run simulation to generate it.")
except Exception as e:
    print(f"Error inspecting footfall report: {e}")

# Example: Inspecting the agent demographics summary
try:
    agent_demographics_report = pd.read_csv(AGENT_SUMMARY_REPORT_PATH)
    print("\n--- Sample of Agent Demographics Summary ---")
    print(agent_demographics_report.head())
except FileNotFoundError:
    print(f"\n{AGENT_SUMMARY_REPORT_PATH} not found. Run simulation to generate it.")
except Exception as e:
    print(f"Error inspecting agent demographics: {e}")