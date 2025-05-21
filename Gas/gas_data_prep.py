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
SPEED_STATS_PATH = os.path.join(DATA_DIR, "speed_norm_stats.json") 

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
        required_cols = ['Timestamp', 'SteeringAngle_Commanded', 'Speed_kmh_GPS', 'CommandedSpeed_kmh']
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


    try:
        gyro_df = pd.read_csv(gyro_path)
        gyro_df.set_index('Timestamp', inplace=True)
        gyro_df.sort_index(inplace=True)
        print(f"Loaded gyro data: {len(gyro_df)} records.")
    except FileNotFoundError:
        print(f"ERROR: Gyro file not found at {gyro_path}")
        return []
    except Exception as e:
        print(f"ERROR reading gyro CSV: {e}")
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

    print("Checking for corresponding sensor files...")
    for index, row in sync_df.iterrows():
        timestamp = row['Timestamp']
        steering_angle_commanded = row['SteeringAngle_Commanded']

        actual_speed_gps = row['Speed_kmh_GPS']
        commanded_speed = row['CommandedSpeed_kmh'] 

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
                'steering_angle_commanded': steering_angle_commanded, 

                'actual_speed_kmh': actual_speed_gps, 
                'commanded_speed_kmh': commanded_speed, 

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
    print("Note: Only samples with both Camera and Lidar files were kept for potential future multi-sensor models.")
    print("-" * 30)
    return matched_data

def calculate_norm_stats(data_list, key_name):
    print(f"Calculating normalization stats for '{key_name}' using {len(data_list)} samples...")
    if not data_list:
        print(f"Warning: Data list is empty for '{key_name}'. Returning mean=0, std=1.")
        return 0.0, 1.0
    try:
        values = [data[key_name] for data in data_list if key_name in data]
        if not values: 
             print(f"Warning: No valid values found for '{key_name}'. Returning mean=0, std=1.")
             return 0.0, 1.0

    except KeyError:
         print(f"ERROR: Key '{key_name}' not found in some data dictionaries during stats calculation.")
         return 0.0, 1.0
    except Exception as e:
         print(f"An unexpected error occurred during stats calculation for '{key_name}': {e}")
         return 0.0, 1.0

    values_np = np.array(values)
    mean = np.mean(values_np)
    std_dev = np.std(values_np)

    if std_dev < 1e-6:
        print(f"Warning: Standard deviation for '{key_name}' is near zero ({std_dev:.6f}). Using std_dev = 1.0.")
        std_dev = 1.0

    print(f"'{key_name}' - Mean: {mean:.6f}, Std Dev: {std_dev:.6f}")
    print("-" * 30)
    return mean, std_dev


def calculate_gyro_norm_stats(data_list, axis_index):
    print(f"Calculating Gyro normalization stats for axis {axis_index} using {len(data_list)} samples...")
    if not data_list:
        print("Warning: Data list is empty for gyro. Returning mean=0, std=1.")
        return 0.0, 1.0
    try:
        gyro_values = [data['gyro_data'][axis_index] for data in data_list if 'gyro_data' in data and isinstance(data['gyro_data'], (list, tuple)) and len(data['gyro_data']) > axis_index]
        if not gyro_values: 
             print(f"Warning: No valid gyro values found for axis {axis_index}. Returning mean=0, std=1.")
             return 0.0, 1.0

    except IndexError:
        print(f"ERROR: Invalid gyro axis index {axis_index} during stats calculation. Gyro data might not be tuples of sufficient length.")
        return 0.0, 1.0
    except KeyError:
         print(f"ERROR: 'gyro_data' key not found in some data dictionaries during stats calculation.")
         return 0.0, 1.0
    except Exception as e:
         print(f"An unexpected error occurred during gyro stats calculation: {e}")
         return 0.0, 1.0


    gyro_values_np = np.array(gyro_values)
    mean = np.mean(gyro_values_np)
    std_dev = np.std(gyro_values_np)

    if std_dev < 1e-6:
        print(f"Warning: Standard deviation for gyro axis {axis_index} is near zero ({std_dev:.6f}). Using std_dev = 1.0.")
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
        actual_speed_mean, actual_speed_std = calculate_norm_stats(train_data, 'actual_speed_kmh')
        commanded_speed_mean, commanded_speed_std = calculate_norm_stats(train_data, 'commanded_speed_kmh')


        print(f"Saving Normalization stats to: {GYRO_STATS_PATH} and {SPEED_STATS_PATH}")
        gyro_stats_dict = {"mean": gyro_mean, "std_dev": gyro_std}
        try:
            with open(GYRO_STATS_PATH, 'w') as f: 
                json.dump(gyro_stats_dict, f, indent=4)
            print("Gyro stats JSON saved successfully.")
        except IOError as e:
            print(f"ERROR: Could not write gyro stats JSON file: {e}")

        speed_stats_dict = {
            "actual_speed_kmh": {"mean": actual_speed_mean, "std_dev": actual_speed_std},
            "commanded_speed_kmh": {"mean": commanded_speed_mean, "std_dev": commanded_speed_std}
        }
        try:
            with open(SPEED_STATS_PATH, 'w') as f: 
                json.dump(speed_stats_dict, f, indent=4)
            print("Speed stats JSON saved successfully.")
        except IOError as e:
            print(f"ERROR: Could not write speed stats JSON file: {e}")

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