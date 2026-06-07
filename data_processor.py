import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler
import time

class H2DataPipeline:
    """
    Data processing pipeline -> handles geocoding, feature creation, and normalization.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.geolocator = Nominatim(user_agent="hyggle_logistics_ems")

    def load_and_geocode(self, file_path):
        """Loads the excel file, calculates coordinates and creates columns"""
        df = pd.read_excel(file_path)
        lats, longs = [], []

        for idx, row in df.iterrows():
            coord_val = row.get('coordonées', np.nan) # either coordinates or NaN
            address_val = str(row.get('adresse', '')).strip() # either address or '' for geopy / str in case of numbers like 69130 / strip for invisible spaces
            
            # Case A: Coordinates already present
            if pd.notna(coord_val) and str(coord_val).strip() != "":
                try:
                    lat, lon = map(float, str(coord_val).split(',')) # map -> strings inside ['',''] become two floats
                    lats.append(lat)
                    longs.append(lon)
                    continue 
                except ValueError:
                    raise ValueError(f"Invalid coordinate format at row {idx}")

            # Case B: Geocoding via address
            if address_val:
                location = None
                try:
                    # Perfect match attempt
                    location = self.geolocator.geocode(address_val, timeout=10)
                except: pass

                # Precise peeling
                if not location and ',' in address_val:
                    parts = address_val.split(',')
                    keyword = parts[0].split(' ')[-1] 
                    city_part = parts[1].strip()  # 265 Avenue des Belleville, 73600 Moûtiers, France becomes Belleville, 73600 Moûtiers
                    try:
                        location = self.geolocator.geocode(f"{keyword}, {city_part}", timeout=10)
                    except: pass

                # Brutal peeling (only searching for the city)
                if not location and ',' in address_val:
                    parts = address_val.split(',')
                    simplified = ", ".join(parts[1:]).strip()
                    try:
                        location = self.geolocator.geocode(simplified, timeout=10)
                    except: pass

                # Final validation
                if location:
                    lats.append(location.latitude)
                    longs.append(location.longitude)
                    time.sleep(1) # Respecting the API rate limit
                else:
                    print(f"Geocoding failed at row {idx} : {address_val} -> Row dropped")
                    lats.append(np.nan)
                    longs.append(np.nan)
            else:
                lats.append(np.nan)
                longs.append(np.nan)

        # Updating coordinates
        df['latitude'] = lats
        df['longitude'] = longs

        # Creating important columns
        df['demand (kg/day)'] = np.where(df['type'] == 'Site de production ', 0, 400)
        df['type_val'] = df['type'].map({'Site de production ': 0, 'Station': 1})

        # Dropping permanently invalid rows
        return df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)

    def process(self, file_path):
        """Executes the complete pipeline: loading, feature engineering, and normalization."""
        df = self.load_and_geocode(file_path)
        
        # Selecting columns 
        features = df[['latitude', 'longitude', 'demand (kg/day)', 'type_val']].values
        X_normalized = self.scaler.fit_transform(features)
        
        return df, X_normalized