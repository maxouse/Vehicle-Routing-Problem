import numpy as np

class HydrogenLogisticsEnv:
    def __init__(self, df, dist_matrix, truck_capacity=1500, max_trucks=5):
        self.df = df
        self.dist_matrix = dist_matrix
        self.max_capacity = truck_capacity
        self.max_trucks = max_trucks
        self.n_nodes = len(df)
        self.depots_idx = df[df['type_val'] == 0].index.tolist()
        # By default, the first truck starts from the first source
        self.first_depot = self.depots_idx[0]

    def reset(self):
        self.remaining_demands = self.df['demand (kg/day)'].values.astype(float).copy()
        for d_idx in self.depots_idx:
            self.remaining_demands[d_idx] = 0.0 
        
        self.current_truck_idx = 0
        self.current_truck_pos = self.first_depot
        self.current_truck_load = self.max_capacity
        
        self.fleet_history = [[self.first_depot]] 
        self.total_distance = 0.0
        return self._get_state()
    
    def _get_state(self):
        return {
            'position': self.current_truck_pos,
            'load': self.current_truck_load,
            'demands': self.remaining_demands.copy()
        }

    def step(self, action_node):
        reward = -1.0 # Time penalty
        done = False
        
        distance = self.dist_matrix[self.current_truck_pos][action_node]
        self.total_distance += distance
        reward -= distance 
        
        self.current_truck_pos = action_node
        self.fleet_history[self.current_truck_idx].append(action_node)
        
        # FIX: We check if the action_node belongs to the depots list
        if action_node in self.depots_idx:
            self.current_truck_idx += 1
            if self.current_truck_idx < self.max_trucks:
                self.current_truck_load = self.max_capacity
                # The next truck starts where the previous one stopped (or we spawn a new one at the chosen depot)
                self.fleet_history.append([action_node])
            else:
                done = True
                remaining_demand = np.sum(self.remaining_demands)
                if remaining_demand > 0:
                    reward -= (1000 + (remaining_demand * 10.0))
        else:
            demand = self.remaining_demands[action_node]
            delivered_quantity = min(self.current_truck_load, demand)
            self.current_truck_load -= delivered_quantity
            self.remaining_demands[action_node] -= delivered_quantity
            
        if np.sum(self.remaining_demands) == 0:
            done = True
            
        return self._get_state(), reward, done, self.fleet_history