import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import folium
import pandas as pd

def plot_attention_heatmap(attention_matrix, df):
    """
    Generates a Heatmap of supply probabilities (in %).
    The color scale is dynamically stretched to maximize contrast.
    """
    # 1. Label generation
    labels = [f"{'Source' if row['type_val'] == 0 else 'Station'} {idx}" for idx, row in df.iterrows()]
    
    plt.figure(figsize=(14, 11))
    
    # 2. Conversion to percentage
    percentage_matrix = attention_matrix * 100

    # Dynamic colors
    vmax_dynamic = np.max(percentage_matrix)
    
    # Failsafe: if the matrix is empty or full of zeros
    if vmax_dynamic == 0:
        vmax_dynamic = 10.0 
    
    # 3. Heatmap creation with Viridis palette
    sns.heatmap(percentage_matrix, 
                xticklabels=labels, 
                yticklabels=labels, 
                cmap='viridis', 
                annot=True, 
                fmt=".1f", 
                vmin=0, 
                vmax=vmax_dynamic,
                cbar_kws={'label': "Supply Probability (%)"})

    # Titles and aesthetics
    plt.title("Optimized Hydrogen Flow Distribution (Attention Weights)", fontsize=14, pad=15)
    plt.xlabel("Potential Supplier Site (Key)", fontsize=12)
    plt.ylabel("Demanding Site (Query)", fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_attention_mask(attention_mask, df):
    """
    Displays the hybrid mask. 
    Blocked routes (-1e9) are displayed in black.
    Allowed routes show their distance penalty (from 0 to -1).
    """
    labels = [f"{'Source' if row['type_val'] == 0 else 'Station'} {idx}" for idx, row in df.iterrows()]
    
    # Replace -1e9 with NaN (Not a Number) for display purposes
    visual_mask = np.where(attention_mask <= -1e8, np.nan, attention_mask)
    
    plt.figure(figsize=(14, 11))
    
    # Background color will be black (for NaNs)
    plt.gca().set_facecolor('black')
    
    # Using a Red/Yellow palette for penalties
    sns.heatmap(visual_mask, 
                xticklabels=labels, 
                yticklabels=labels, 
                cmap='YlOrRd_r', 
                annot=True, 
                fmt=".2f", 
                cbar_kws={'label': 'Distance Penalty (Black = Blocked)'})

    plt.title("Attention Mask Inspection\n(Black = Forbidden, Colors = Trip Penalty)", fontsize=14, pad=15)
    plt.xlabel("Potential Supplier Site (Key)", fontsize=12)
    plt.ylabel("Demanding Site (Query)", fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_route_map(df, routes):
    """
    Generates an interactive HTML map with stations and truck routes.
    """
    print("\nGenerating interactive map (Folium)...")
    
    # Map initialization centered on the mean of points
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    # 1. Marker Placement (Plants and Stations)
    for idx, row in df.iterrows():
        if row['type_val'] == 0:  # Source
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup="SOURCE (Production)",
                icon=folium.Icon(color='green', icon='industry', prefix='fa')
            ).add_to(m)
        else:  # Station
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=f"STATION {idx} (Demand: {row['demand (kg/day)']}kg)",
                icon=folium.Icon(color='blue', icon='gas-pump', prefix='fa')
            ).add_to(m)

    # 2. Drawing routes
    colors = ['red', 'blue', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'black']
    
    for i, route in enumerate(routes):
        # Retrieving coordinates in trip order
        route_coords = []
        for node in route:
            lat = df.iloc[node]['latitude']
            lon = df.iloc[node]['longitude']
            route_coords.append([lat, lon])
            
        # Random color for each truck
        color = colors[i % len(colors)]
        
        # Drawing the line
        folium.PolyLine(
            route_coords,
            color=color,
            weight=4,
            opacity=0.8,
            popup=f"Truck {i+1}"
        ).add_to(m)

    # Saving the map to an HTML file
    file_name = "route_map.html"
    m.save(file_name)
    print(f"-> Map saved as : '{file_name}'. Open this file in your browser!")
    
def plot_learning_curve(history):
    """
    Generates a 3-chart dashboard showing KPI evolution.
    """
    print("\nGenerating learning dashboard...")
    
    # Creating a large figure with 3 stacked charts
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    epochs = range(len(history['rewards']))
    # Dynamic smoothing window (50 by default, less if training is short)
    window = min(50, max(1, len(epochs) // 20))

    # --- Chart 1: Score (Reward) ---
    ax1.plot(epochs, history['rewards'], color='lightblue', alpha=0.5)
    ax1.plot(pd.Series(history['rewards']).rolling(window=window, min_periods=1).mean(), color='blue', linewidth=2)
    ax1.set_title("Total Reward Evolution (AI Score)", fontweight='bold', fontsize=12)
    ax1.set_ylabel("Score")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # --- Chart 2: Kilometers Traveled ---
    ax2.plot(epochs, history['distances'], color='lightgreen', alpha=0.5)
    ax2.plot(pd.Series(history['distances']).rolling(window=window, min_periods=1).mean(), color='green', linewidth=2)
    ax2.set_title("Distance Traveled Evolution", fontweight='bold', fontsize=12)
    ax2.set_ylabel("Kilometers (km)")
    ax2.grid(True, linestyle='--', alpha=0.7)

    # --- Chart 3: Wasted Hydrogen ---
    ax3.plot(epochs, history['waste'], color='salmon', alpha=0.5)
    ax3.plot(pd.Series(history['waste']).rolling(window=window, min_periods=1).mean(), color='red', linewidth=2)
    ax3.set_title("Hydrogen Waste Evolution (Empty space brought back to depot)", fontweight='bold', fontsize=12)
    ax3.set_xlabel("Epochs (Training Days)", fontsize=11)
    ax3.set_ylabel("Hydrogen (kg)")
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    file_name = "learning_dashboard.png"
    plt.savefig(file_name, dpi=300)
    plt.show()
    print(f"-> Dashboard saved as : '{file_name}'.")