import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class CarPreprocessing:

    def __init__(self):
        self.encoders = {}
        self.scaler = None

    def clean_data(self, df):
        df = df.copy()

        # drop useless
        df.drop('ID', axis=1, inplace=True)

        # Price
        df['Price'] = df['Price'].str.replace('$','').str.replace(',','')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

        # Engine volume
        df['Engine_volume'] = df['Engine_volume'].str.extract(r'(\d+\.?\d*)')
        df['Engine_volume'] = pd.to_numeric(df['Engine_volume'], errors='coerce')

        # Mileage
        df['Mileage'] = df['Mileage'].str.replace('KM','')
        df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')

        # Doors
        df['Doors'] = df['Doors'].str.extract(r'(\d+)')
        df['Doors'] = pd.to_numeric(df['Doors'], errors='coerce')

        return df

    def handle_missing(self, df):
        df = df.copy()

        df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce')
        df['Levy'] = df['Levy'].fillna(df['Levy'].median())

        df['Mileage'] = df['Mileage'].fillna(df['Mileage'].median())

        df['Color'] = df['Color'].fillna(df['Color'].mode()[0])

        return df

    def feature_engineering(self, df):
        df = df.copy()

        from datetime import datetime
        current_year = datetime.now().year

        df['Age_of_car'] = current_year - df['Prod._year']

        return df

    def encoding(self, df):
        df = df.copy()

        cat_cols = [
            'Manufacturer', 'Category', 'Fuel_type',
            'Gear_box_type', 'Drive_wheels',
            'Color', 'Wheel', 'Leather_interior'
        ]

        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        return df

    def scaling(self, df):
        df = df.copy()

        num_cols = df.select_dtypes(include=np.number).columns

        self.scaler = StandardScaler()
        df[num_cols] = self.scaler.fit_transform(df[num_cols])

        return df

    def fit_transform(self, df):
        df = self.clean_data(df)
        df = self.handle_missing(df)
        df = self.feature_engineering(df)
        df = self.encoding(df)

        return df