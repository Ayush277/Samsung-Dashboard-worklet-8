import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from sklearn.metrics import r2_score
import hashlib


class FileHandler:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        self._create_upload_folder()

    def _create_upload_folder(self):
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)

    def handle_single_csv_upload(self, file, model, data_processor):
        """
        Handle test data CSV upload for prediction.
        Expects merged CSV with all features EXCEPT Sales/Customers.
        Will compute and log R2 if a Sales column exists in the uploaded file.
        """
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(self.upload_folder, filename)
        file.save(filepath)

        try:
            # Read the test CSV file
            print(f"Reading test file: {filename}")
            test_df = pd.read_csv(filepath)
            print(f"Test file loaded successfully. Shape: {test_df.shape}")
            print(f"Columns found: {list(test_df.columns)}")

            # Keep original for output alignment
            original_df = test_df.copy()

            # Required columns for test data (NO Sales/Customers required)
            required_test_columns = [
                'Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                'CompetitionDistance', 'CompetitionOpenSinceMonth',
                'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                'Promo2SinceYear', 'PromoInterval'
            ]
            missing_cols = [c for c in required_test_columns if c not in test_df.columns]
            if missing_cols:
                return {
                    'success': False,
                    'error': f"Missing required columns: {missing_cols}"
                }

            if 'Sales' in test_df.columns:
                print("Note: Sales column present in upload. It won't be used for features.")
            if 'Customers' in test_df.columns:
                print("Note: Customers column present in upload. It won't be used for features.")

            # Remove any stale Predicted_Sales from input to avoid confusion
            if 'Predicted_Sales' in original_df.columns:
                print("Removing stale Predicted_Sales from uploaded file before processing")
                original_df = original_df.drop(columns=['Predicted_Sales'])

            print("✓ All required test columns found!")
            print("Processing test data through pipeline...")
            processed_features = data_processor.process_test_data(test_df)
            print("Processed features shape:", processed_features.shape)

            # Make predictions
            print("Making predictions with pre-trained model...")
            predictions = model.predict(processed_features)

            # If trained on sqrt(Sales), you can enable the inverse transform:
            # predictions = np.square(predictions)

            # Ensure numeric, 1D, finite predictions
            predictions = np.asarray(predictions, dtype='float64').reshape(-1)
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

            print("original_df rows:", len(original_df))
            print("processed_features rows:", len(processed_features))
            print("predictions len:", len(predictions))
            print("Predictions stats:", {
                "min": float(np.min(predictions)) if len(predictions) else None,
                "max": float(np.max(predictions)) if len(predictions) else None,
                "nan_count": int(np.isnan(predictions).sum())
            })

            # Align predictions with original rows using index
            pred_series = pd.Series(predictions, index=processed_features.index, name='Predicted_Sales')

            output_df = original_df.copy()
            # Reindex predictions to output_df index to avoid blanks due to index mismatch
            output_df['Predicted_Sales'] = pred_series.reindex(output_df.index).values
            # Final safety: coerce numeric and fill NaN with 0.0
            output_df['Predicted_Sales'] = pd.to_numeric(output_df['Predicted_Sales'], errors='coerce').fillna(0.0)

            # Optionally compute R2 if ground-truth Sales exists
            r2 = None
            if 'Sales' in output_df.columns:
                y_true = pd.to_numeric(output_df['Sales'], errors='coerce')
                y_pred = output_df['Predicted_Sales']
                mask = y_true.notna() & np.isfinite(y_true.values) & np.isfinite(y_pred.values)
                if mask.sum() > 1:
                    try:
                        r2 = r2_score(y_true[mask], y_pred[mask])
                        print(f"R2 score on uploaded data (where Sales available): {r2:.6f}")
                    except Exception as e:
                        print("R2 computation failed:", e)
                        r2 = None
                else:
                    print("R2 score: not computed (insufficient valid rows with Sales).")
            else:
                print("R2 score: not computed (no Sales column in upload).")

            print("Final Predicted_Sales non-null count:", int(output_df['Predicted_Sales'].notna().sum()))
            print("First 5 predictions:", output_df['Predicted_Sales'].head().tolist())

            # Build result filename (avoid double "predictions_" prefix)
            base = filename
            if base.lower().startswith("predictions_"):
                base = base[len("predictions_"):]
            results_filename = f"predictions_{base}"
            results_path = os.path.join(self.upload_folder, results_filename)

            # Save results
            output_df.to_csv(results_path, index=False)

            # Print file checksum for verification
            with open(results_path, 'rb') as f:
                content = f.read()
            print("Saved file SHA1:", hashlib.sha1(content).hexdigest())
            print(f"✓ Predictions completed! Saved to: {results_filename}")

            return {
                'success': True,
                'row_count': len(output_df),
                'download_url': f"/download/{results_filename}",
                'message': f"Successfully predicted sales for {len(output_df)} records",
                'r2': None if r2 is None else float(r2)
            }

        except Exception as e:
            print(f"Error processing test file: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': f'Error processing file: {str(e)}'}
