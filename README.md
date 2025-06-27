# San Francisco Restaurant Footfall Simulation

This project is an agent-based model (ABM) that simulates footfall traffic to various restaurants in San Francisco. It uses Pygame for visualization and OSMnx for real-world street network data. Agents with different demographic profiles make decisions on which restaurant to visit based on preferences and distance.

## Simulation Output

![Simulation Output](data/output.gif)

## Features

-   Real-world street data from OpenStreetMap.
-   Dynamic agent generation based on simulated hourly footfall.
-   Agents with distinct archetypes (e.g., 'Young Professional', 'Family') that influence their choices.
-   Interactive Pygame visualization with zoom and pan capabilities.
-   Outputs detailed CSV reports on hourly footfall and agent demographics.

## How to Run

1.  **Install Dependencies:**
    Make sure you have Python 3 installed. Then, install the required libraries:
    ```bash
    pip install pandas pygame osmnx matplotlib geopy shapely pyproj scipy numpy tqdm
    ```

2.  **Required Data:**
    Place the following data files in the same directory as the script:
    -   `OSR.csv`
    -   `lat_lon_stores.csv`
    -   `trade_area_profiling.csv`

3.  **Execute the Script:**
    Run the simulation from your terminal:
    ```bash
    python main.py
    ```

## Configuration

Key simulation parameters (e.g., `SIMULATION_ADDRESS`, `SIMULATION_HOURS_TOTAL`) can be easily modified in the **"Configuration and Constants"** section at the top of the Python script.

##Additional Files

The scripts generate_agent and archetype_restaurant_choice generate agents based on San Francisco census data, categorizing them into 1,000 distinct archetypes derived from that data.
