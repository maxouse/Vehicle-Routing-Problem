from data_processor import H2DataPipeline
from rl_environment import HydrogenLogisticsEnv
from rl_agent import train_rl_agent
from visualizer import plot_route_map, plot_learning_curve
import numpy as np

def _calculate_haversine(coords):
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    R = 6371.0
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    for i in range(n):
        dlat = lat - lat[i]
        dlon = lon - lon[i]
        a = np.sin(dlat/2)**2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        dist_matrix[i, :] = R * c
    return dist_matrix

def main():
    print("=== Starting RL Optimizer (Deep Reinforcement Learning) ===")
    
    data_file = 'données.xlsx' 
    
    print("\n1. Loading data...")
    pipeline = H2DataPipeline()
    df, _ = pipeline.process(data_file)
    
    coords = df[['latitude', 'longitude']].values
    dist_matrix = _calculate_haversine(coords)
    
    # Truck capacity fixed at 700kg to force Split Delivery
    truck_capacity = 700
 
    total_demand = df['demand (kg/day)'].sum()
    
    # The computer calculates the minimum number of trucks required (with +2 for a small margin)
    unlimited_fleet = int((total_demand / truck_capacity) + 2) 
    
    env = HydrogenLogisticsEnv(df, dist_matrix, truck_capacity=truck_capacity, max_trucks=unlimited_fleet)
    
    print("\n3. Training AI Agent...")
    agent, best_route, history = train_rl_agent(env, epochs=10000)
    
    print("\n=== FLEET RESULT (SPLIT DELIVERY MULTI-VRP) ===")
    if not best_route:
        print("The AI failed to complete the mission.")
    else:
        for i, truck_route in enumerate(best_route):
            path_text = []
            for node in truck_route:
                if df.iloc[node]['type_val'] == 0:
                    path_text.append("[DEPOT]")
                else:
                    path_text.append(f"Station {node}")
            print(f"Truck {i+1} : " + " -> ".join(path_text))
            
        print("\n=== FLEET STATISTICS (KPIs) ===")
        # 1. Calculating basic metrics
        used_trucks_count = len(best_route)
        total_demand = df['demand (kg/day)'].sum()
        total_capacity = used_trucks_count * truck_capacity
        
        # 2. Calculating waste and optimization
        unused_hydrogen = total_capacity - total_demand
        fill_rate = (total_demand / total_capacity) * 100
        
        # 3. Displaying report
        print(f"-> Deployed trucks count : {used_trucks_count}")
        print(f"-> Total mobilized capacity  : {total_capacity:.0f} kg")
        print(f"-> Total delivered demand    : {total_demand:.0f} kg")
        print(f"-> Unused hydrogen           : {unused_hydrogen:.0f} kg (Empty space in trailers)")
        print(f"-> Overall fill rate         : {fill_rate:.1f} %")
    
        print("\n4. Generating visualizations...")
        plot_learning_curve(history)
        plot_route_map(df, best_route)

if __name__ == "__main__":
    main()