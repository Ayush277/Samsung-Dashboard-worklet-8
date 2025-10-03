import pandas as pd
import numpy as np
from config import Config
import warnings
warnings.filterwarnings("ignore")


class DataProcessor:
    def __init__(self, scaler, encoder=None):
        self.scaler = scaler
        self.encoder = encoder

    def process_test_data(self, test_df):
        """
        Convert raw test CSV (no Sales/Customers) into the exact scaler-ready
        feature matrix the model was trained on.
        """
        print("╭──────────────────────────────────────────────╮")
        print("│ Processing test data for prediction …        │")
        print("╰──────────────────────────────────────────────╯")
        print("Input columns:", list(test_df.columns))

        df = test_df.copy()
        df["Sales"] = 1000
        df["Customers"] = 100

        try:
            df = self._convert_data_types(df)
            df = self._create_features(df)
            df = self._handle_negative_values(df)
            df = self._feature_selection(df)
            df = self._apply_transformations(df)
            df = self._encode_categorical_features(df)
            df = self._handle_dummy_trap(df)

            # Remove dummy targets, date
            df.drop(['Sales', 'Customers', 'Date'], axis=1, errors='ignore', inplace=True)
            print("Columns after removing targets:", list(df.columns))

            # Use scaler.feature_names_in_ as the source of truth
            if hasattr(self.scaler, "feature_names_in_"):
                expected = list(self.scaler.feature_names_in_)
                print("Scaler feature list loaded from scaler.feature_names_in_")
            else:
                expected = Config.EXPECTED_FEATURES
                print("Scaler has no feature_names_in_; falling back to Config.EXPECTED_FEATURES")

            # Add missing cols with 0, drop extras, enforce order
            df = df.reindex(columns=expected, fill_value=0)
            print("Final column order:", list(df.columns))
            print("Final feature shape before scaling:", df.shape)

            # Scale (use NumPy array to bypass strict name check)
            scaled = self.scaler.transform(df.values)

            out_df = pd.DataFrame(scaled, columns=expected, index=df.index)
            print("Processed features ready:", out_df.shape, "NaNs:", out_df.isna().any().any())
            return out_df

        except Exception as e:
            print("❌  Error during test-data processing:", e)
            import traceback; traceback.print_exc()
            raise Exception(f"Test data processing failed: {e}")


    def _apply_transformations(self, df):
        """Apply square root and log transformations with safety guards."""
        sqrt_features = ['Sales', 'Customers', 'CompetitionOpenNumMonths', 'Promo2NumWeeks']
        for feature in sqrt_features:
            if feature in df.columns:
                df[feature] = np.sqrt(df[feature].clip(lower=0))

        if 'CompetitionDistance' in df.columns:
            df['CompetitionDistance'] = df['CompetitionDistance'].clip(lower=1e-9)
            df['CompetitionDistance'] = np.log(df['CompetitionDistance'])

        return df


    def _encode_categorical_features(self, df):
        """One-hot encode categorical features consistently."""
        if 'PromoInterval' in df.columns:
            df = pd.get_dummies(df, columns=['PromoInterval'])

        categorical_cols = ['StoreType', 'Assortment']
        existing_cat_cols = [c for c in categorical_cols if c in df.columns]

        if existing_cat_cols and (self.encoder is not None):
            try:
                encoded_array = self.encoder.transform(df[existing_cat_cols])
                encoded_features = list(self.encoder.get_feature_names_out(existing_cat_cols))
                for i, col in enumerate(encoded_features):
                    df[col] = encoded_array[:, i]
                df.drop(existing_cat_cols, axis=1, inplace=True)
            except Exception as e:
                print("Encoder failed; falling back to get_dummies:", e)
                df = pd.get_dummies(df, columns=existing_cat_cols)
        else:
            df = pd.get_dummies(df, columns=existing_cat_cols)

        return df


    # ------------------------------
    # Internal steps used in training
    # ------------------------------
    def _convert_data_types(self, df):
        """Convert data types"""
        # Convert date to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Convert StateHoliday to numeric (as in training)
        df["StateHoliday"].replace({'0': 0, 'a': 1, 'b': 1, 'c': 1}, inplace=True)

        # Convert competition and promo columns to int
        cast_map = {
            "CompetitionOpenSinceMonth": int,
            "CompetitionOpenSinceYear": int,
            "Promo2SinceWeek": int,
            "Promo2SinceYear": int
        }
        # Safe astype (ignore missing)
        for col, typ in cast_map.items():
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(typ)

        return df

    def _create_features(self, df):
        """Create new features from date and other columns"""
        # Extract date features
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
        df['Month'] = df['Date'].dt.month.astype(int)
        df['Year'] = df['Date'].dt.year.astype(int)

        # Create competition feature
        df['CompetitionOpenNumMonths'] = (
            (df['Year'] - df['CompetitionOpenSinceYear']) * 12 +
            (df['Month'] - df['CompetitionOpenSinceMonth'])
        )

        # Create promo2 feature
        df['Promo2NumWeeks'] = (
            (df['Year'] - df['Promo2SinceYear']) * 52 +
            (df['WeekOfYear'] - df['Promo2SinceWeek'])
        )

        return df

    def _handle_negative_values(self, df):
        """Handle negative values in engineered features"""
        # Set negative competition months to 0
        if 'CompetitionOpenNumMonths' in df.columns:
            df['CompetitionOpenNumMonths'] = df['CompetitionOpenNumMonths'].apply(lambda x: 0 if x < 0 else x)

        # Set Promo2NumWeeks to 0 where Promo2 is 0
        if 'Promo2' in df.columns and 'Promo2NumWeeks' in df.columns:
            df.loc[df['Promo2'] == 0, 'Promo2NumWeeks'] = 0

        # Set negative Promo2NumWeeks to 0
        if 'Promo2NumWeeks' in df.columns:
            df['Promo2NumWeeks'] = df['Promo2NumWeeks'].apply(lambda x: 0 if x < 0 else x)

        return df

    def _feature_selection(self, df):
        """Drop unnecessary columns (as done during training)"""
        columns_to_drop = [
            'Store', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
            'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'Month', 'Year'
        ]
        for col in columns_to_drop:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        return df

    def _apply_transformations(self, df):
        """Apply square root and log transformations"""
        # Square root transformations (clip to avoid sqrt negatives)
        sqrt_features = ['Sales', 'Customers', 'CompetitionOpenNumMonths', 'Promo2NumWeeks']
        for feature in sqrt_features:
            if feature in df.columns:
                df[feature] = np.sqrt(df[feature].clip(lower=0))

        # Log transformation for CompetitionDistance (avoid log(0))
        if 'CompetitionDistance' in df.columns:
            df['CompetitionDistance'] = df['CompetitionDistance'].clip(lower=1e-9)
            df['CompetitionDistance'] = np.log(df['CompetitionDistance'])

        return df

    def _encode_categorical_features(self, df):
        """One-hot encode categorical features"""
        # PromoInterval via get_dummies
        if 'PromoInterval' in df.columns:
            df = pd.get_dummies(df, columns=['PromoInterval'])

        # StoreType and Assortment using encoder if provided; fallback to dummies
        categorical_cols = ['StoreType', 'Assortment']
        existing_cat_cols = [c for c in categorical_cols if c in df.columns]

        if existing_cat_cols and (self.encoder is not None):
            try:
                encoded_array = self.encoder.transform(df[existing_cat_cols])
                encoded_features = list(self.encoder.get_feature_names_out(existing_cat_cols))
                for i, col in enumerate(encoded_features):
                    df[col] = encoded_array[:, i]
                df.drop(existing_cat_cols, axis=1, inplace=True)
            except Exception as e:
                print("Encoder failed; falling back to get_dummies:", e)
                df = pd.get_dummies(df, columns=existing_cat_cols)
        else:
            df = pd.get_dummies(df, columns=existing_cat_cols)

        return df

    def _handle_dummy_trap(self, df):
        """Drop columns to avoid dummy variable trap, as per training."""
        columns_to_drop = [
            'PromoInterval_Jan,Apr,Jul,Oct',
            'StoreType_c',
            'Assortment_b'
        ]
        for col in columns_to_drop:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        return df
