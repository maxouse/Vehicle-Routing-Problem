# H2-Logistics-RL: Deep Reinforcement Learning with Attention Mechanisms for Green Hydrogen Routing

An advanced, end-to-end Machine Learning pipeline designed to optimize multi-vehicle hydrogen distribution networks. This repository leverages a custom **Deep Reinforcement Learning (DRL)** environment coupled with a **Transformer-based Attention Agent** to solve a complex variant of the Vehicle Routing Problem (VRP).

---

## 1. Problem Statement & Logistics Context

In the transition toward sustainable energy ecosystems, green hydrogen ($H_2$) distribution presents a high-stakes logistics bottleneck. Hydrogen possesses a low volumetric energy density, requiring high-pressure or cryogenic transportation in specialized tube trailers. Efficient fleet routing is paramount to minimize carbon footprints and operational costs.

This project mathematically models the distribution network as a **Multi-Vehicle Routing Problem with Split Deliveries (SDVRP)**:
* **The Network:** Consists of multiple production sites (Depots/Sources) and a variable number of refueling stations scattered across a territory, each with a specific daily demand ($kg/day$).
* **Fleet Constraints:** A fleet of heavy-duty transport trucks with fixed payload capacities ($700\text{ kg}$). 
* **Split Deliveries:** If a single station's remaining demand exceeds a truck's available capacity, the truck delivers its remaining load, returns to a depot to refill, and another truck (or the same one on a subsequent trip) fulfills the remainder.
* **Optimization Objectives:** 1. Minimize total fleet mileage (Distance Penalty).
  2. Maximize the trailer fill-rate, minimizing "empty space" brought back to production plants (Hydrogen Waste/Capacity Penalty).
  3. Guarantee 100% demand fulfillment across all stations under strict multi-depot conditions.

---

## 2. Algorithmic Design & Architecture

### Environment (`rl_environment.py`)
Built as a custom Markov Decision Process (MDP) tracking:
* **State Space:** Dynamic fleet configuration (current truck position, remaining trailer capacity) and the global grid state (remaining unsatisfied demand vector across all nodes).
* **Action Space:** Discrete nodes (refueling stations or active production depots).
* **Action Masking:** A strict rule-based hybrid masking matrix prevents illegal behaviors (e.g., visiting an empty station, trying to serve a client when the trailer is empty, or heading to a depot when the truck is already fully loaded).

### The Attention Agent (`rl_agent.py`)
Instead of a rigid Multi-Layer Perceptron (MLP) which binds the model to a fixed input size, this repository utilizes an **Attention-based Policy Network (Transformer Encoder style)**. 
* **Query:** Encodes the *Truck State* (current coordinates, current load).
* **Keys/Values:** Encodes the *Map/Nodes State* (station coordinates, distance from truck, remaining demand).

By computing **Scaled Dot-Product Attention**, the model maps the continuous spatial topology dynamically. This architecture ensures high **generalizability**, allowing the trained agent to handle an arbitrary number of stations without altering the underlying neural network parameters.

---

## 3. Repository Structure

* `main.py`: Main entry point initializing the pipeline, instantiating parameters, launching the training loop, and parsing terminal execution data.
* `data_processor.py`: Advanced geospatial pipeline utilizing `geopy` (Nominatim API) to clean unstructured addresses, parse explicit GPS coordinates, and handle automated min-max feature normalization.
* `rl_environment.py`: Implements the `HydrogenLogisticsEnv` discrete routing state-machine.
* `rl_agent.py`: Houses the TensorFlow 2.x `LogisticsAttentionAgent` and the REINFORCE Policy Gradient training framework.
* `visualizer.py`: Data viz engine mapping outputs into interactive Folium HTML environments and saving technical evaluation metrics.

---

## 4. Installation & Usage

### Prerequisites
Ensure you have Python 3.10+ installed. Clone the repository and install dependencies:

```bash
git clone [https://github.com/your-username/H2-Logistics-RL.git](https://github.com/your-username/H2-Logistics-RL.git)
cd H2-Logistics-RL
pip install -r requirements.txt
