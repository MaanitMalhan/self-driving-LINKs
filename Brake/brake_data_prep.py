import pandas as pd
import numpy as np
import os
import random
import json
import pickle

DATA_DIR = "data" 
CONTROLS_CSV_PATH = os.path.join(DATA_DIR, "controls_data/controls.csv")
CAMERA_DATA_DIR = os.path.join(DATA_DIR, "camera_data/")
LIDAR_DATA_DIR = os.path.join(DATA_DIR, "lidar_data/")

LIDAR_STATS_PATH = os.path.join(DATA_DIR, "lidar_norm_stats.json") 

TRAIN_DATA_PKL_PATH_BRAKE = os.path.join(DATA_DIR, "train_data_brake.pkl") 
VAL_DATA_PKL_PATH_BRAKE = os.path.join(DATA_DIR, "val_data_brake.pkl")   
TEST_DATA_PKL_PATH_BRAKE = os.path.join(DATA_DIR, "test_data_brake.pkl")  

TIMESTAMP_TOLERANCE = 0.005 

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

def sync_data_brake(controls_path, camera_dir, lidar_dir, tolerance=0.005):
    print("Starting data synchronization for braking model...")
    try:
        controls_df = pd.read_csv(controls_path)
        print(f"Loaded controls data: {len(controls_df)} records.")
        required_cols = ['Timestamp', 'BrakeIntensity_Commanded']
        if not all(col in controls_df.columns for col in required_cols):
            print(f"ERROR: Controls CSV missing one or more required columns: {required_cols}")
            print(f"Found columns: {controls_df.columns.tolist()}")
            return []

    except FileNotFoundError:
        print(f"ERROR: Controls file not found at {controls_path}")
        return []
    except Exception as e:
        print(f"ERROR reading controls CSV: {e}")
        return []


    controls_df.sort_values(by='Timestamp', inplace=True)
    matched_data = []
    skipped_camera = 0
    skipped_lidar = 0

    print(f"Attempting synchronization using controls timestamps and file existence with tolerance: {tolerance}s")

    for index, row in controls_df.iterrows():
        timestamp = row['Timestamp']
        brake_intensity_commanded = row['BrakeIntensity_Commanded'] 

        ts_str = f"{timestamp:.3f}" 
        camera_filename = f"camera_{ts_str}.png"
        camera_filepath = os.path.join(camera_dir, camera_filename)
        lidar_filename = f"lidar_{ts_str}.csv"
        lidar_filepath = os.path.join(lidar_dir, lidar_filename) 

        camera_exists = os.path.exists(camera_filepath)
        lidar_exists = os.path.exists(lidar_filepath) 

        if camera_exists and lidar_exists:
            matched_data.append({
                'timestamp': timestamp,
                'brake_intensity_commanded': brake_intensity_commanded, 

                'camera_path': camera_filepath,
                'lidar_path': lidar_filepath
            })
        else:
            if not camera_exists: skipped_camera += 1
            if not lidar_exists: skipped_lidar += 1


    print("-" * 30)
    print(f"Synchronization Complete for braking model!")
    print(f"Total matched records with existing camera and lidar files: {len(matched_data)}")
    print(f"Skipped due to missing Camera file: {skipped_camera}")
    print(f"Skipped due to missing Lidar file: {skipped_lidar}")
    print("Note: Only samples with both Camera and Lidar files were kept.")
    print("-" * 30)
    return matched_data

def calculate_lidar_norm_stats(training_data):
    print(f"Calculating Lidar normalization stats using {len(training_data)} training samples...")
    if not training_data:
        print("Warning: Training data is empty for lidar stats. Returning mean=0, std=1.")
        return 0.0, 1.0

    all_lidar_ranges = []
    skipped_files = 0

    for data_item in training_data:
        lidar_path = data_item.get('lidar_path')
        if lidar_path and os.path.exists(lidar_path):
            try:
                ranges = np.loadtxt(lidar_path, delimiter=',')
                all_lidar_ranges.extend(ranges.flatten()) 
            except Exception as e:
                print(f"Warning: Could not load or process lidar file {lidar_path}: {e}. Skipping.")
                skipped_files += 1
        else:
             skipped_files += 1


    if not all_lidar_ranges:
        print(f"Warning: No valid lidar range values found from {len(training_data)} files. Returning mean=0, std=1.")
        return 0.0, 1.0

    lidar_values_np = np.array(all_lidar_ranges)

    valid_ranges = lidar_values_np[np.isfinite(lidar_values_np) & (lidar_values_np < 20.1)] 

    if not valid_ranges.size > 0:
        print("Warning: No finite or valid range values found after filtering. Returning mean=0, std=1.")
        return 0.0, 1.0


    mean = np.mean(valid_ranges)
    std_dev = np.std(valid_ranges)

    if std_dev < 1e-6:
        print(f"Warning: Standard deviation for lidar ranges is near zero ({std_dev:.6f}). Using std_dev = 1.0.")
        std_dev = 1.0

    print(f"Lidar Ranges (Filtered) - Mean: {mean:.6f}, Std Dev: {std_dev:.6f}")
    if skipped_files > 0:
        print(f"Skipped {skipped_files} lidar files during stats calculation.")
    print("-" * 30)
    return mean, std_dev


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    all_synced_data_brake = sync_data_brake(
        CONTROLS_CSV_PATH,
        CAMERA_DATA_DIR,
        LIDAR_DATA_DIR,
        tolerance=TIMESTAMP_TOLERANCE
    )

    if not all_synced_data_brake:
        print("No data synchronized for the braking model. Exiting.")
    else:
        random.shuffle(all_synced_data_brake)

        total_samples = len(all_synced_data_brake)
        train_end_idx = int(total_samples * TRAIN_RATIO)
        val_end_idx = train_end_idx + int(total_samples * VAL_RATIO)

        train_data_brake = all_synced_data_brake[:train_end_idx]
        val_data_brake = all_synced_data_brake[train_end_idx:val_end_idx]
        test_data_brake = all_synced_data_brake[val_end_idx:] 

        print(f"Data Splitting for braking model:")
        print(f"Total synchronized samples (with camera/lidar files): {total_samples}")
        print(f"Training samples: {len(train_data_brake)}")
        print(f"Validation samples: {len(val_data_brake)}")
        print(f"Test samples: {len(test_data_brake)}")
        print("-" * 30)

        lidar_mean, lidar_std = calculate_lidar_norm_stats(train_data_brake)


        print(f"Saving Lidar normalization stats to: {LIDAR_STATS_PATH}")
        lidar_stats_dict = {"mean": lidar_mean, "std_dev": lidar_std}
        try:
            with open(LIDAR_STATS_PATH, 'w') as f: 
                json.dump(lidar_stats_dict, f, indent=4)
            print("Lidar stats JSON saved successfully.")
        except IOError as e:
            print(f"ERROR: Could not write lidar stats JSON file: {e}")
        print("-" * 30)

        print("Saving data splits (train/val/test lists) using Pickle for braking model...")
        try:
            with open(TRAIN_DATA_PKL_PATH_BRAKE, 'wb') as f: 
                pickle.dump(train_data_brake, f)
            print(f"Training data saved to: {TRAIN_DATA_PKL_PATH_BRAKE}")

            with open(VAL_DATA_PKL_PATH_BRAKE, 'wb') as f:
                pickle.dump(val_data_brake, f)
            print(f"Validation data saved to: {VAL_DATA_PKL_PATH_BRAKE}")

            with open(TEST_DATA_PKL_PATH_BRAKE, 'wb') as f:
                pickle.dump(test_data_brake, f)
            print(f"Test data saved to: {TEST_DATA_PKL_PATH_BRAKE}")

        except IOError as e:
            print(f"ERROR: Could not write Pickle file: {e}")
        except pickle.PicklingError as e:
             print(f"ERROR: Could not pickle data splits: {e}")
        print("-" * 30)

        print("Data preparation script for braking model finished.")