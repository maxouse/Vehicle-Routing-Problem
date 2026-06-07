import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd

class LogisticsAttentionAgent(tf.keras.Model):
    def __init__(self, d_model=32):
        super().__init__()
        self.d_model = d_model
        
        # --- TRUCK BRAIN ---
        self.truck_hidden = tf.keras.layers.Dense(64, activation='relu')
        self.W_query = tf.keras.layers.Dense(d_model) 

        # --- MAP BRAIN ---
        self.nodes_hidden = tf.keras.layers.Dense(64, activation='relu')
        self.W_key = tf.keras.layers.Dense(d_model)

    def call(self, truck_state, nodes_state, action_mask):
        h_truck = self.truck_hidden(truck_state)
        Q = self.W_query(h_truck)            # [1, d_model]

        h_nodes = self.nodes_hidden(nodes_state)
        K = self.W_key(h_nodes)              # [1, N, d_model]

        Q_expanded = tf.expand_dims(Q, 1)    
        scores = tf.matmul(Q_expanded, K, transpose_b=True) 
        scores = tf.squeeze(scores, 1)       

        # Scaled Dot-Product to prevent exploding numbers
        scores = scores / tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Applying Mask
        scores += action_mask
        
        # Clamping scores between -20 and 20
        scores = tf.clip_by_value(scores, -20.0, 20.0)

        # Soft probabilities
        action_probs = tf.nn.softmax(scores, axis=-1)
        
        return action_probs

def train_rl_agent(env, epochs=1000):
    agent = LogisticsAttentionAgent()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    coords = env.df[['latitude', 'longitude']].values.astype(np.float32)
    
    best_route = []
    best_reward = -float('inf')

    history = {
        'rewards': [],
        'distances': [],
        'waste': [],
        'undelivered': []
    }

    # Calculate total demand once at the beginning
    initial_total_demand = env.df['demand (kg/day)'].sum()

    print("\n[RL] Starting Deep Reinforcement Learning training...")
    progress_bar = tqdm(range(epochs), desc="AI Training", unit="day")
    
    for epoch in progress_bar:
        exploration_rate = max(0.001, 0.10 * (1.0 - (epoch / epochs))) 
        
        state = env.reset()
        step_rewards = []
        log_probs = []
        total_reward = 0
        done = False
        
        trip_counter = 0 
        
        # --- START TENSORFLOW GRADIENT TAPE ---
        with tf.GradientTape() as tape:
            while not done and trip_counter < 300:
                trip_counter += 1
                
                truck_lat, truck_lon = coords[state['position']]
                
                # Scaling
                truck_tensor = tf.constant([[state['load'] / 1000.0, truck_lat / 100.0, truck_lon / 100.0]], dtype=tf.float32)
                
                distances_from_truck = env.dist_matrix[state['position']] / 100.0
                demands_scale = state['demands'] / 1000.0
                coords_scale = coords / 100.0
                
                nodes_features = np.column_stack((demands_scale, coords_scale, distances_from_truck))
                nodes_tensor = tf.constant([nodes_features], dtype=tf.float32)
                
                mask = np.zeros((1, env.n_nodes), dtype=np.float32)
                total_remaining_demand = np.sum(state['demands'])
                
                if state['load'] == 0 or total_remaining_demand == 0:
                    # Truck is empty: Allowed to go to ANY depot
                    for i in range(env.n_nodes):
                        if i not in env.depots_idx:
                            mask[0, i] = -1e9
                else:
                    # Truck is loaded: Forbidden to go to depots
                    for i in range(env.n_nodes):
                        if state['demands'][i] == 0:
                            mask[0, i] = -1e9 
                            
                    mask[0, state['position']] = -1e9 
                    for d_idx in env.depots_idx:
                        mask[0, d_idx] = -1e9
                    
                mask_tensor = tf.constant(mask)

                # Get pure probabilities from the network
                probs = agent(truck_tensor, nodes_tensor, mask_tensor)
                
                # EXPLORATION (Only sample from authorized stations)
                if np.random.rand() < exploration_rate:
                    legal_actions = np.where(mask[0] == 0)[0] 
                    action = np.random.choice(legal_actions)
                else:
                    action = tf.random.categorical(tf.math.log(probs + 1e-10), 1)[0, 0].numpy()
                
                log_prob = tf.math.log(probs[0, action] + 1e-10)
                log_probs.append(log_prob)
                
                state, reward, done, route = env.step(action)
                total_reward += reward
                step_rewards.append(reward)
                
            
            if trip_counter >= 300:
                unsatisfied_demand = np.sum(env.remaining_demands)
                penalty = 1000 + (unsatisfied_demand * 10.0)
                total_reward -= penalty
                
            # 1. Baseline calculation (Moving average of the last 50 trips)
            if len(history['rewards']) > 0:
                baseline = np.mean(history['rewards'][-50:])
            else:
                baseline = total_reward
                
            # 2. Advantage 
            advantage = (total_reward - baseline) / 100.0
            
            # 3. Loss: Update probabilities based on global advantage
            if len(log_probs) > 0:
                loss = tf.reduce_sum([-lp * advantage for lp in log_probs])
            else:
                loss = tf.constant(0.0)
        # --- END TENSORFLOW GRADIENT TAPE ---
        
        # 3. Apply gradients
        gradients = tape.gradient(loss, agent.trainable_variables)
        optimizer.apply_gradients(zip(gradients, agent.trainable_variables))
        
        # --- SAVE KPIs ---
        unsatisfied_demand = np.sum(env.remaining_demands) 
        delivered_demand = initial_total_demand - unsatisfied_demand
        deployed_capacity = len(env.fleet_history) * env.max_capacity
        wasted_hydrogen = deployed_capacity - delivered_demand
        
        history['rewards'].append(total_reward)
        history['distances'].append(env.total_distance)
        history['waste'].append(wasted_hydrogen)
        history['undelivered'].append(unsatisfied_demand)
        
        # Update best route
        if done and total_reward > best_reward:
            best_reward = total_reward
            best_route = route.copy()
            
            progress_bar.set_postfix({
                "Max Score": f"{best_reward:.0f}", 
                "Min Dist": f"{env.total_distance:.0f}km"
            })

        # ==========================================
        # REAL-TIME DISPLAY
        # ==========================================
        if epoch % 50 == 0 and epoch > 0:
            clear_output(wait=True)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))
            
            ax1.plot(history['rewards'], color='lightblue', alpha=0.5)
            ax1.plot(pd.Series(history['rewards']).rolling(window=50, min_periods=1).mean(), color='blue', linewidth=2)
            ax1.set_title(f"Epoch {epoch}/{epochs} - Score Evolution", fontweight='bold')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            ax2.plot(history['distances'], color='lightgreen', alpha=0.5)
            ax2.plot(pd.Series(history['distances']).rolling(window=50, min_periods=1).mean(), color='green', linewidth=2)
            ax2.set_title("Evolution of Distance Traveled (km)", fontweight='bold')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            ax3.plot(history['waste'], color='salmon', alpha=0.5)
            ax3.plot(pd.Series(history['waste']).rolling(window=50, min_periods=1).mean(), color='red', linewidth=2)
            ax3.set_title("Evolution of Hydrogen Waste (kg)", fontweight='bold')
            ax3.grid(True, linestyle='--', alpha=0.7)

            ax4.plot(history['undelivered'], color='thistle', alpha=0.5)
            ax4.plot(pd.Series(history['undelivered']).rolling(window=50, min_periods=1).mean(), color='purple', linewidth=2)
            ax4.set_title("Incomplete Demand (Remaining hydrogen kg on map)", fontweight='bold')
            ax4.set_xlabel("Epochs (Training Days)")
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.show()
            
            print(f"Progress: {epoch}/{epochs} epochs. Best Score: {best_reward:.0f}")

    print("\n[RL] Training completed!")
    return agent, best_route, history