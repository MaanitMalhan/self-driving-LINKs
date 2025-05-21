import pandas as pd
import numpy as np
import os
import random
import json
import pickle

DATA_DIR = "data" 
CONTROLS_CSV_PATH = os.path.join(DATA_DIR, "controls_data/controls.csv")
GYRO_CSV_PATH = os.path.join(DATA_DIR, "gyro_data/gyro.csv")
CAMERA_DATA_DIR = os.path.join(DATA_DIR, "camera_data/")
LIDAR_DATA_DIR = os.path.join(DATA_DIR, "lidar_data/")

GYRO_STATS_PATH = os.path.join(DATA_DIR, "gyro_norm_stats.json")

TRAIN_DATA_PKL_PATH = os.path.join(DATA_DIR, "train_data.pkl") 
VAL_DATA_PKL_PATH = os.path.join(DATA_DIR, "val_data.pkl")   
TEST_DATA_PKL_PATH = os.path.join(DATA_DIR, "test_data.pkl")  

GYRO_AXIS_INDEX = 2 

TIMESTAMP_TOLERANCE = 0.005 

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

def sync_data(controls_path, gyro_path, camera_dir, lidar_dir, tolerance=0.005):
    print("Starting data synchronization...")
    try:
        controls_df = pd.read_csv(controls_path)
        print(f"Loaded controls data: {len(controls_df)} records.")
    except FileNotFoundError:
        print(f"ERROR: Controls file not found at {controls_path}")
        return []
    try:
        gyro_df = pd.read_csv(gyro_path)
        gyro_df.set_index('Timestamp', inplace=True)
        gyro_df.sort_index(inplace=True)
        print(f"Loaded gyro data: {len(gyro_df)} records.")
    except FileNotFoundError:
        print(f"ERROR: Gyro file not found at {gyro_path}")
        return []

    controls_df.sort_values(by='Timestamp', inplace=True)
    matched_data = []
    skipped_camera = 0
    skipped_lidar = 0

    print(f"Attempting synchronization using pd.merge_asof with tolerance: {tolerance}s")
    sync_df = pd.merge_asof(
        controls_df, gyro_df, on='Timestamp', direction='nearest', tolerance=tolerance
    )
    sync_df.dropna(subset=['X', 'Y', 'Z'], inplace=True)
    print(f"Found {len(sync_df)} potential matches based on timestamp tolerance after merging.")

    for index, row in sync_df.iterrows():
        timestamp = row['Timestamp']
        steering_angle = row['SteeringAngle_Commanded']
        gyro_values = (row['X'], row['Y'], row['Z'])

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
                'steering_angle': steering_angle, 
                'camera_path': camera_filepath,
                'gyro_data': gyro_values, 
                'lidar_path': lidar_filepath
            })
        else:
            if not camera_exists: skipped_camera += 1
            if not lidar_exists: skipped_lidar += 1


    print("-" * 30)
    print(f"Synchronization Complete!")
    print(f"Total matched records with existing camera and lidar files: {len(matched_data)}")
    print(f"Skipped due to missing Camera file: {skipped_camera}")
    print(f"Skipped due to missing Lidar file: {skipped_lidar}")
    print("-" * 30)
    return matched_data

def calculate_gyro_norm_stats(training_data, axis_index):
    print(f"Calculating Gyro normalization stats for axis {axis_index} using {len(training_data)} training samples...")
    if not training_data:
        print("Warning: Training data is empty. Returning mean=0, std=1.")
        return 0.0, 1.0
    try:
        gyro_values = [data['gyro_data'][axis_index] for data in training_data if 'gyro_data' in data and len(data['gyro_data']) > axis_index]
        if not gyro_values: 
             print(f"Warning: No valid gyro values found for axis {axis_index}. Returning mean=0, std=1.")
             return 0.0, 1.0

    except IndexError:
        print(f"ERROR: Invalid gyro axis index {axis_index} during stats calculation. Gyro data might not be tuples of sufficient length.")
        return 0.0, 1.0
    except KeyError:
         print(f"ERROR: 'gyro_data' key not found in some training data dictionaries during stats calculation.")
         return 0.0, 1.0
    except Exception as e:
         print(f"An unexpected error occurred during gyro stats calculation: {e}")
         return 0.0, 1.0


    gyro_values_np = np.array(gyro_values)
    mean = np.mean(gyro_values_np)
    std_dev = np.std(gyro_values_np)

    if std_dev < 1e-6:
        print(f"Warning: Standard deviation for axis {axis_index} is near zero ({std_dev:.6f}). Using std_dev = 1.0.")
        std_dev = 1.0

    print(f"Gyro Axis {axis_index} - Mean: {mean:.6f}, Std Dev: {std_dev:.6f}")
    print("-" * 30)
    return mean, std_dev

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    all_synced_data = sync_data(
        CONTROLS_CSV_PATH,
        GYRO_CSV_PATH,
        CAMERA_DATA_DIR,
        LIDAR_DATA_DIR, 
        tolerance=TIMESTAMP_TOLERANCE
    )

    if not all_synced_data:
        print("No data synchronized. Exiting.")
    else:
        random.shuffle(all_synced_data)

        total_samples = len(all_synced_data)
        train_end_idx = int(total_samples * TRAIN_RATIO)
        val_end_idx = train_end_idx + int(total_samples * VAL_RATIO)

        train_data = all_synced_data[:train_end_idx]
        val_data = all_synced_data[train_end_idx:val_end_idx]
        test_data = all_synced_data[val_end_idx:] 

        print(f"Data Splitting:")
        print(f"Total synchronized samples (with camera/lidar files): {total_samples}")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")
        print("-" * 30)

        gyro_mean, gyro_std = calculate_gyro_norm_stats(train_data, GYRO_AXIS_INDEX)

        print(f"Saving Gyro normalization stats to: {GYRO_STATS_PATH}")
        stats_dict = {"mean": gyro_mean, "std_dev": gyro_std}
        try:
            with open(GYRO_STATS_PATH, 'w') as f: 
                json.dump(stats_dict, f, indent=4)
            print("Stats JSON saved successfully.")
        except IOError as e:
            print(f"ERROR: Could not write stats JSON file: {e}")
        print("-" * 30)

        print("Saving data splits (train/val/test lists) using Pickle...")
        try:
            with open(TRAIN_DATA_PKL_PATH, 'wb') as f: 
                pickle.dump(train_data, f)
            print(f"Training data saved to: {TRAIN_DATA_PKL_PATH}")

            with open(VAL_DATA_PKL_PATH, 'wb') as f:
                pickle.dump(val_data, f)
            print(f"Validation data saved to: {VAL_DATA_PKL_PATH}")

            with open(TEST_DATA_PKL_PATH, 'wb') as f:
                pickle.dump(test_data, f)
            print(f"Test data saved to: {TEST_DATA_PKL_PATH}")

        except IOError as e:
            print(f"ERROR: Could not write Pickle file: {e}")
        except pickle.PicklingError as e:
             print(f"ERROR: Could not pickle data splits: {e}")
        print("-" * 30)

        print("Data preparation script finished.")