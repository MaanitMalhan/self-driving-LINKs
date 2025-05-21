import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import json
import pickle
import os
import random
import time
from copy import deepcopy 
import multiprocessing 

DATA_DIR = "data"
GYRO_STATS_PATH = os.path.join(DATA_DIR, "gyro_norm_stats.json") 
SPEED_STATS_PATH = os.path.join(DATA_DIR, "speed_norm_stats.json") 

TRAIN_DATA_PKL_PATH = os.path.join(DATA_DIR, "train_data.pkl")
VAL_DATA_PKL_PATH = os.path.join(DATA_DIR, "val_data.pkl")

IMG_HEIGHT = 66
IMG_WIDTH = 200
IMG_CHANNELS = 3

USE_GYRO_INPUT = True 
GYRO_AXIS_INDEX = 2 
GYRO_INPUT_FEATURES = 1 

ACTUAL_SPEED_INPUT_FEATURES = 1 
COMMANDED_SPEED_TARGET_FEATURES = 1 


BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_DATA_WORKERS = 4 

EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.2

MODEL_SAVE_PATH = 'gas_model_camera_speed_pytorch.pth' 
LOSS_PLOT_PATH = 'gas_model_loss_plot.png' 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print("-" * 30)


class SpeedDataset(Dataset):
    def __init__(self, data_list, gyro_axis_idx, gyro_mean, gyro_std, actual_speed_mean, actual_speed_std, commanded_speed_mean, commanded_speed_std, img_h, img_w, img_c, use_gyro=True, augment=False):
        self.data_list = data_list
        self.gyro_axis_idx = gyro_axis_idx
        self.gyro_mean = gyro_mean
        self.gyro_std = gyro_std
        self.actual_speed_mean = actual_speed_mean
        self.actual_speed_std = actual_speed_std
        self.commanded_speed_mean = commanded_speed_mean
        self.commanded_speed_std = commanded_speed_std
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.use_gyro = use_gyro
        self.augment = augment

        self.gyro_effective_std = self.gyro_std if self.gyro_std >= 1e-6 else 1.0
        self.actual_speed_effective_std = self.actual_speed_std if self.actual_speed_std >= 1e-6 else 1.0
        self.commanded_speed_effective_std = self.commanded_speed_std if self.commanded_speed_std >= 1e-6 else 1.0


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        image_path = data_item.get('camera_path')
        actual_speed = data_item.get('actual_speed_kmh', 0.0)
        commanded_speed = data_item.get('commanded_speed_kmh', 0.0)
        gyro_data_tuple = data_item.get('gyro_data', (0.0, 0.0, 0.0)) 


        if image_path is None or not os.path.exists(image_path):
             print(f"Warning: Image path invalid or not found for index {idx}: {image_path}. Returning zero image tensor.")
             img_tensor = torch.zeros((self.img_c, self.img_h, self.img_w), dtype=torch.float32)
        else:
            try:
                img = cv2.imread(image_path)
                if img is None: raise IOError(f"Failed to load image: {image_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                img = cv2.resize(img, (self.img_w, self.img_h)) 

                if self.augment:
                     if random.random() > 0.5:
                         delta = random.uniform(-0.2, 0.2) * 255 
                         img = cv2.add(img, delta)
                         img = np.clip(img, 0, 255).astype(np.uint8)

                img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

            except Exception as e:
                 print(f"Warning: Error processing image {image_path} for index {idx}: {e}. Returning zero tensor.")
                 img_tensor = torch.zeros((self.img_c, self.img_h, self.img_w), dtype=torch.float32)


        normalized_actual_speed = (float(actual_speed) - self.actual_speed_mean) / self.actual_speed_effective_std
        actual_speed_tensor = torch.tensor([normalized_actual_speed], dtype=torch.float32) 

        normalized_commanded_speed = (float(commanded_speed) - self.commanded_speed_mean) / self.commanded_speed_effective_std
        commanded_speed_tensor = torch.tensor([normalized_commanded_speed], dtype=torch.float32) 


        gyro_tensor = torch.zeros(GYRO_INPUT_FEATURES, dtype=torch.float32) 
        if self.use_gyro and gyro_data_tuple is not None and isinstance(gyro_data_tuple, (list, tuple)) and len(gyro_data_tuple) > self.gyro_axis_idx:
            try:
                gyro_value = gyro_data_tuple[self.gyro_axis_idx]
                normalized_gyro = (float(gyro_value) - self.gyro_mean) / self.gyro_effective_std
                gyro_tensor = torch.tensor([normalized_gyro], dtype=torch.float32) 
            except Exception as e:
                print(f"Warning: Error processing gyro for item {idx}: {e}. Using 0.")


        inputs_list = [img_tensor, actual_speed_tensor]
        if self.use_gyro:
             inputs_list.append(gyro_tensor)

        target = commanded_speed_tensor

        return inputs_list, target 


class SpeedModel(nn.Module):
    def __init__(self, img_c=3, img_h=66, img_w=200, actual_speed_features=1, gyro_features=1, use_gyro=True):
        super().__init__()

        self.camera_path = nn.Sequential(
            nn.BatchNorm2d(img_c), 
            nn.Conv2d(img_c, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )

        try:
            with torch.no_grad(): 
                dummy_input_img = torch.randn(1, img_c, img_h, img_w).to(DEVICE)
                flattened_size = self.camera_path(dummy_input_img).shape[1]
            print(f"  Dynamically calculated flattened image feature size: {flattened_size}")
        except Exception as e:
             print(f"ERROR calculating flattened size: {e}. Check image dimensions and Conv layers.")
             raise RuntimeError(f"Failed to calculate flattened size: {e}") 


        non_image_input_size = actual_speed_features + (gyro_features if use_gyro else 0)
        print(f"  Total non-image input feature size: {non_image_input_size}")

        if non_image_input_size > 0:
            self.non_image_path = nn.Sequential(
                nn.Linear(non_image_input_size, 32), nn.ReLU(), 
                nn.BatchNorm1d(32) 
            )
            non_image_processed_size = 32 
        else:
            self.non_image_path = nn.Identity() 
            non_image_processed_size = 0


        combined_features_size = flattened_size + non_image_processed_size
        print(f"  Total combined feature size: {combined_features_size}")

        if combined_features_size > 0:
            self.combined_path = nn.Sequential(
                nn.Linear(combined_features_size, 100), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(100, 50), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(50, 10), nn.ReLU(),
                nn.Linear(10, 1) 
            )
        else:
             self.combined_path = nn.Linear(1, 1) 


        self.use_gyro = use_gyro 


    def forward(self, inputs):
        img_batch = inputs[0]
        actual_speed_batch = inputs[1]

        img_features = self.camera_path(img_batch)

        if self.use_gyro:
            gyro_batch = inputs[2]
            non_image_inputs = torch.cat((actual_speed_batch, gyro_batch), dim=1)
        else:
             non_image_inputs = actual_speed_batch

        if self.non_image_path is not nn.Identity():
             non_image_features = self.non_image_path(non_image_inputs)
        else:
             non_image_features = torch.empty(img_features.size(0), 0).to(img_features.device)


        combined_features = torch.cat((img_features, non_image_features), dim=1) 

        speed_output = self.combined_path(combined_features)
        return speed_output


if __name__ == '__main__':

    GYRO_MEAN = 0.0 
    GYRO_STD = 1.0
    if USE_GYRO_INPUT:
        print(f"Loading gyro normalization stats from: {GYRO_STATS_PATH}")
        try:
            with open(GYRO_STATS_PATH, 'r') as f: gyro_stats = json.load(f)
            GYRO_MEAN = gyro_stats["mean"]
            GYRO_STD = gyro_stats["std_dev"]
            print(f"  Loaded Gyro Mean: {GYRO_MEAN:.6f}, Std Dev: {GYRO_STD:.6f}")
            if GYRO_STD < 1e-6:
                print("Warning: Loaded gyro standard deviation is near zero. Setting to 1.0.")
                GYRO_STD = 1.0
        except FileNotFoundError:
            print(f"WARNING: Gyro stats file '{GYRO_STATS_PATH}' not found. Disabling gyro input.")
            USE_GYRO_INPUT = False 
        except KeyError:
            print("WARNING: Gyro stats file missing 'mean' or 'std_dev' keys. Disabling gyro input.")
            USE_GYRO_INPUT = False 
    else:
        print("Gyro input is disabled for the speed model based on configuration.")


    print(f"Loading speed normalization stats from: {SPEED_STATS_PATH}")
    ACTUAL_SPEED_MEAN = 0.0 
    ACTUAL_SPEED_STD = 1.0
    COMMANDED_SPEED_MEAN = 0.0 
    COMMANDED_SPEED_STD = 1.0
    try:
        with open(SPEED_STATS_PATH, 'r') as f: speed_stats = json.load(f)
        ACTUAL_SPEED_MEAN = speed_stats["actual_speed_kmh"]["mean"]
        ACTUAL_SPEED_STD = speed_stats["actual_speed_kmh"]["std_dev"]
        COMMANDED_SPEED_MEAN = speed_stats["commanded_speed_kmh"]["mean"]
        COMMANDED_SPEED_STD = speed_stats["commanded_speed_kmh"]["std_dev"]

        print(f"  Loaded Actual Speed Mean: {ACTUAL_SPEED_MEAN:.6f}, Std Dev: {ACTUAL_SPEED_STD:.6f}")
        print(f"  Loaded Commanded Speed Mean: {COMMANDED_SPEED_MEAN:.6f}, Std Dev: {COMMANDED_SPEED_STD:.6f}")

        if ACTUAL_SPEED_STD < 1e-6:
             print("Warning: Loaded actual speed standard deviation is near zero. Setting to 1.0.")
             ACTUAL_SPEED_STD = 1.0
        if COMMANDED_SPEED_STD < 1e-6:
             print("Warning: Loaded commanded speed standard deviation is near zero. Setting to 1.0.")
             COMMANDED_SPEED_STD = 1.0

    except FileNotFoundError:
        print(f"ERROR: Speed stats file '{SPEED_STATS_PATH}' not found.")
        print("Please ensure the data preparation script was run successfully first.")
        exit()
    except KeyError:
        print("ERROR: Speed stats file missing expected keys ('actual_speed_kmh' or 'commanded_speed_kmh').")
        exit()
    print("-" * 30)


    print("Loading data splits from Pickle files...")
    try:
        with open(TRAIN_DATA_PKL_PATH, 'rb') as f: train_data = pickle.load(f)
        print(f"Loaded {len(train_data)} training samples from: {TRAIN_DATA_PKL_PATH}")
        with open(VAL_DATA_PKL_PATH, 'rb') as f: val_data = pickle.load(f)
        print(f"Loaded {len(val_data)} validation samples from: {VAL_DATA_PKL_PATH}")

    except FileNotFoundError as e:
        print(f"ERROR: Data split file not found: {e}")
        print("Please ensure the data preparation script was run successfully first.")
        exit()
    except pickle.UnpicklingError as e:
         print(f"ERROR: Could not unpickle data file: {e}")
         exit()
    except Exception as e:
         print(f"An unexpected error occurred loading data: {e}")
         exit()
    print("-" * 30)

    train_dataset = SpeedDataset(train_data, GYRO_AXIS_INDEX, GYRO_MEAN, GYRO_STD,
                                ACTUAL_SPEED_MEAN, ACTUAL_SPEED_STD, COMMANDED_SPEED_MEAN, COMMANDED_SPEED_STD,
                                IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, use_gyro=USE_GYRO_INPUT, augment=True)
    val_dataset = SpeedDataset(val_data, GYRO_AXIS_INDEX, GYRO_MEAN, GYRO_STD,
                              ACTUAL_SPEED_MEAN, ACTUAL_SPEED_STD, COMMANDED_SPEED_MEAN, COMMANDED_SPEED_STD,
                              IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, use_gyro=USE_GYRO_INPUT, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_DATA_WORKERS, pin_memory=True, persistent_workers=(NUM_DATA_WORKERS > 0))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_DATA_WORKERS, pin_memory=True, persistent_workers=(NUM_DATA_WORKERS > 0))

    print("PyTorch DataLoaders created for Speed Model.")
    print("-" * 30)

    print("Building PyTorch CNN model for Speed Prediction...")
    model = SpeedModel(img_c=IMG_CHANNELS, img_h=IMG_HEIGHT, img_w=IMG_WIDTH,
                       actual_speed_features=ACTUAL_SPEED_INPUT_FEATURES,
                       gyro_features=GYRO_INPUT_FEATURES, use_gyro=USE_GYRO_INPUT).to(DEVICE)
    print(model)
    print("-" * 30)


    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)

    print("Starting model training for Speed Prediction...")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None 

    train_loss_history = []
    val_loss_history = []

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train() 
        running_loss = 0.0
        running_mae = 0.0
        train_samples = 0

        for i, (inputs, labels) in enumerate(train_loader):
            device_inputs = [inp.to(DEVICE) for inp in inputs]

            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(device_inputs) 
            loss = criterion(outputs.squeeze(), labels.squeeze()) 


            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0) 
            mae = torch.abs(outputs.squeeze() - labels.squeeze()).mean().item()
            running_mae += mae * labels.size(0)
            train_samples += labels.size(0)


            if (i + 1) % 100 == 0 or i == len(train_loader) - 1: 
                print(f'\rEpoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, MAE: {mae:.4f}', end='', flush=True)


        epoch_train_loss = running_loss / train_samples
        epoch_train_mae = running_mae / train_samples
        print(f'\rEpoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {epoch_train_loss:.4f}, Train MAE: {epoch_train_mae:.4f}') 
        train_loss_history.append(epoch_train_loss)


        model.eval() 
        val_loss = 0.0
        val_mae = 0.0
        val_samples = 0

        with torch.no_grad(): 
            for inputs, labels in val_loader:
                device_inputs = [inp.to(DEVICE) for inp in inputs]
                labels = labels.to(DEVICE)

                outputs = model(device_inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze())

                val_loss += loss.item() * labels.size(0)
                mae = torch.abs(outputs.squeeze() - labels.squeeze()).mean().item()
                val_mae += mae * labels.size(0)
                val_samples += labels.size(0)


        epoch_val_loss = val_loss / val_samples
        epoch_val_mae = val_mae / val_samples
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Val Loss: {epoch_val_loss:.4f}, Val MAE: {epoch_val_mae:.4f}')
        val_loss_history.append(epoch_val_loss)


        if epoch_val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} --> {epoch_val_loss:.4f}). Saving model...\n")
            best_val_loss = epoch_val_loss
            best_model_state = deepcopy(model.state_dict()) 
            torch.save(best_model_state, MODEL_SAVE_PATH)
            epochs_no_improve = 0 
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve from {best_val_loss:.4f}. ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")


        scheduler.step(epoch_val_loss) 


        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.")
            if best_model_state:
                 print("Restoring best model weights...")
                 model.load_state_dict(best_model_state)
            break 

    print("-" * 30)
    training_time = time.time() - start_time
    print(f"Training finished in {training_time // 60:.0f}m {training_time % 60:.0f}s.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")
    print("-" * 30)

    print("Generating loss plot...")
    epochs_ran = range(1, len(train_loss_history) + 1) 

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_ran, train_loss_history, label='Training Loss')
    plt.plot(epochs_ran, val_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss (MSE) - Speed Model') 
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(LOSS_PLOT_PATH) 
        print(f"Loss plot saved to {LOSS_PLOT_PATH}")
    except Exception as e:
        print(f"Error saving loss plot: {e}")