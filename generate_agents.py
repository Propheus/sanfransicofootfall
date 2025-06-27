import pandas as pd
import numpy as np
from typing import List, Dict
import random

def load_archetype_data() -> pd.DataFrame:
    """
    Load archetype data from 1generated_archetypes.csv
    """
    try:
        df = pd.read_csv('1generated_archetypes.csv')
        return df
    except Exception as e:
        print(f"Error loading archetype data: {str(e)}")
        return pd.DataFrame()

def get_random_age(age_range: str) -> int:
    """
    Generate a random specific age based on the age range.
    
    Parameters:
    - age_range: String containing the age range (e.g., "25 to 34 years")
    
    Returns:
    - Random integer age within the range
    """
    # Extract numbers from the age range
    numbers = [int(s) for s in age_range.split() if s.isdigit()]
    if len(numbers) >= 2:
        return random.randint(numbers[0], numbers[1])
    return 0

def get_racial_combination() -> str:
    """
    Generate a specific racial combination for multi-racial agents.
    
    Returns:
    - String describing the specific racial combination
    """
    racial_groups = [
        "White",
        "Asian",
        "Black or African American",
        "Hispanic or Latino",
        "Native American",
        "Pacific Islander",
        "Middle Eastern"
    ]
    
    # Randomly select 2-3 racial groups
    num_groups = random.randint(2, 3)
    selected_groups = random.sample(racial_groups, num_groups)
    
    return " and ".join(selected_groups)

def generate_agents(num_agents: int = 10000) -> pd.DataFrame:
    """
    Generate a population of agents with demographic profiles.
    
    Parameters:
    - num_agents: Number of agents to generate (default: 10000)
    
    Returns:
    - DataFrame containing agent demographic profiles
    """
    # Load archetype data
    archetypes_df = load_archetype_data()
    if archetypes_df.empty:
        print("Error: No archetype data available")
        return pd.DataFrame()
    
    # Create empty list to store agent profiles
    agent_profiles = []
    
    # Generate agents
    for _ in range(num_agents):
        # Randomly select an archetype
        archetype = archetypes_df.iloc[random.randint(0, len(archetypes_df) - 1)]
        
        # Generate specific age based on age range
        specific_age = get_random_age(archetype['Age'])
        
        # Handle multi-racial cases
        race = archetype['Race']
        if race == "Two or more races":
            race = get_racial_combination()
        
        # Create agent profile
        agent = {
            'agent_id': len(agent_profiles) + 1,
            'archetype_id': archetype.name,  # Index of the archetype
            'age': specific_age,
            'sex': archetype['Sex'],
            'race': race,
            'marital_status': archetype['Marital Status'],
            'education': archetype['Education'],
            'income': archetype['Income'],
            'housing': archetype['Housing']
        }
        
        agent_profiles.append(agent)
    
    # Convert to DataFrame
    agents_df = pd.DataFrame(agent_profiles)
    
    # Add some demographic statistics
    print("\nDemographic Statistics:")
    print(f"Total number of agents: {len(agents_df)}")
    print("\nAge Distribution:")
    print(agents_df['age'].describe())
    print("\nSex Distribution:")
    print(agents_df['sex'].value_counts())
    print("\nRace Distribution:")
    print(agents_df['race'].value_counts())
    print("\nEducation Distribution:")
    print(agents_df['education'].value_counts())
    print("\nIncome Distribution:")
    print(agents_df['income'].value_counts())
    print("\nHousing Distribution:")
    print(agents_df['housing'].value_counts())
    
    return agents_df

def save_agents(agents_df: pd.DataFrame, filename: str = 'generated_agents.csv'):
    """
    Save generated agents to a CSV file.
    
    Parameters:
    - agents_df: DataFrame containing agent profiles
    - filename: Name of the output file (default: 'generated_agents.csv')
    """
    try:
        agents_df.to_csv(filename, index=False)
        print(f"\nSuccessfully saved {len(agents_df)} agents to {filename}")
    except Exception as e:
        print(f"Error saving agents to file: {str(e)}")

def show_sample_agent() -> Dict:
    """
    Generate and display a single sample agent.
    
    Returns:
    - Dictionary containing the sample agent's profile
    """
    # Load archetype data
    archetypes_df = load_archetype_data()
    if archetypes_df.empty:
        print("Error: No archetype data available")
        return {}
    
    # Randomly select an archetype
    archetype = archetypes_df.iloc[random.randint(0, len(archetypes_df) - 1)]
    
    # Generate specific age based on age range
    specific_age = get_random_age(archetype['Age'])
    
    # Handle multi-racial cases
    race = archetype['Race']
    if race == "Two or more races":
        race = get_racial_combination()
    
    # Create agent profile
    agent = {
        'agent_id': 1,
        'archetype_id': archetype.name,  # Index of the archetype
        'age': specific_age,
        'sex': archetype['Sex'],
        'race': race,
        'marital_status': archetype['Marital Status'],
        'education': archetype['Education'],
        'income': archetype['Income'],
        'housing': archetype['Housing']
    }
    
    # Display the sample agent
    print("\nSample Agent Profile:")
    print("-" * 50)
    for key, value in agent.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("-" * 50)
    
    return agent

if __name__ == "__main__":
    # First show a sample agent
    sample_agent = show_sample_agent()
    
    if sample_agent:
        # Ask for confirmation before generating full population
        response = input("\nDoes this sample agent profile look correct? (yes/no): ").lower()
        
        if response == 'yes':
            # Generate full population
            num_agents = int(input("How many agents would you like to generate? (default: 10000): ") or "10000")
            agents_df = generate_agents(num_agents)
            
            if not agents_df.empty:
                # Save to CSV
                save_agents(agents_df)
        else:
            print("Please let me know what changes you'd like to make to the agent profile structure.") 