import pickle
from config import Config
from sklearn.preprocessing import OneHotEncoder


class ModelLoader:
    def __init__(self):
        print("Initializing ModelLoader...")
        self.model = self._load_model()
        print("✓ Model loaded")
        self.scaler = self._load_scaler()
        print("✓ Scaler loaded")
        self.encoder = self._load_encoder()
        print("✓ Encoder loaded")

    def _load_model(self):
        try:
            with open(Config.MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file {Config.MODEL_PATH} not found")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def _load_scaler(self):
        try:
            with open(Config.SCALER_PATH, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Scaler file {Config.SCALER_PATH} not found")
        except Exception as e:
            raise Exception(f"Error loading scaler: {e}")

    def _load_encoder(self):
        try:
            with open(Config.ENCODER_PATH, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("Warning: Encoder file not found, creating dummy encoder")
            return OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')
        except Exception as e:
            print(f"Warning: Error loading encoder ({e}), creating dummy encoder")
            return OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')
