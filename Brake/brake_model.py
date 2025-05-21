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

DATA_DIR = "data"
LIDAR_STATS_PATH = os.path.join(DATA_DIR, "lidar_norm_stats.json") 

TRAIN_DATA_PKL_PATH_BRAKE = os.path.join(DATA_DIR, "train_data_brake.pkl")
VAL_DATA_PKL_PATH_BRAKE = os.path.join(DATA_DIR, "val_data_brake.pkl")

IMG_HEIGHT = 66
IMG_WIDTH = 200
IMG_CHANNELS = 3

LIDAR_HORIZONTAL_RESOLUTION = 2048 
BRAKE_INTENSITY_TARGET_FEATURES = 1 


BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_DATA_WORKERS = 4 

EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.2

MODEL_SAVE_PATH = 'brake_model_camera_lidar_pytorch.pth' 
LOSS_PLOT_PATH = 'brake_model_loss_plot.png' 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print("-" * 30)


class BrakeDataset(Dataset):
    def __init__(self, data_list, lidar_mean, lidar_std, img_h, img_w, img_c, lidar_res, augment=False):
        self.data_list = data_list
        self.lidar_mean = lidar_mean
        self.lidar_std = lidar_std
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.lidar_res = lidar_res
        self.augment = augment

        self.lidar_effective_std = self.lidar_std if self.lidar_std >= 1e-6 else 1.0


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        image_path = data_item.get('camera_path')
        lidar_path = data_item.get('lidar_path')
        brake_intensity = data_item.get('brake_intensity_commanded', 0.0)


        if image_path is None or not os.path.exists(image_path):
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
                 img_tensor = torch.zeros((self.img_c, self.img_h, self.img_w), dtype=torch.float32)


        lidar_tensor = torch.zeros(self.lidar_res, dtype=torch.float32) 
        if lidar_path is None or not os.path.exists(lidar_path):
             pass 
        else:
            try:
                ranges = np.loadtxt(lidar_path, delimiter=',')
                if ranges.size != self.lidar_res:
                    print(f"Warning: Lidar data size mismatch for {lidar_path}. Expected {self.lidar_res}, got {ranges.size}. Returning zero tensor.")
                else:
                    ranges[np.isinf(ranges)] = 20.1 
                    ranges[np.isnan(ranges)] = 20.1 

                    normalized_ranges = (ranges - self.lidar_mean) / self.lidar_effective_std
                    lidar_tensor = torch.tensor(normalized_ranges, dtype=torch.float32) 

            except Exception as e:
                pass 


        brake_intensity_tensor = torch.tensor([float(brake_intensity)], dtype=torch.float32) 


        inputs_list = [img_tensor, lidar_tensor]

        target = brake_intensity_tensor

        return inputs_list, target 


class BrakeModel(nn.Module):
    def __init__(self, img_c=3, img_h=66, img_w=200, lidar_res=2048): 
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


        self.lidar_path = nn.Sequential(
            nn.BatchNorm1d(1), 
            nn.Conv1d(1, 32, kernel_size=5, stride=2), nn.ReLU(), 
            nn.Conv1d(32, 64, kernel_size=3, stride=2), nn.ReLU(), 
            nn.Conv1d(64, 128, kernel_size=3, stride=2), nn.ReLU(), 
            nn.Flatten()
        )

        try:
             with torch.no_grad():
                 dummy_input_lidar = torch.randn(1, 1, lidar_res).to(DEVICE)
                 lidar_flattened_size = self.lidar_path(dummy_input_lidar).shape[1]
             print(f"  Dynamically calculated flattened lidar feature size: {lidar_flattened_size}")
        except Exception as e:
             print(f"ERROR calculating lidar flattened size: {e}. Check lidar resolution and Conv1d layers.")
             raise RuntimeError(f"Failed to calculate lidar flattened size: {e}")


        combined_features_size = flattened_size + lidar_flattened_size
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


    def forward(self, inputs):
        img_batch = inputs[0]
        lidar_batch = inputs[1]

        img_features = self.camera_path(img_batch)

        lidar_batch_reshaped = lidar_batch.unsqueeze(1) 
        lidar_features = self.lidar_path(lidar_batch_reshaped)


        combined_features = torch.cat((img_features, lidar_features), dim=1) 

        brake_output = self.combined_path(combined_features)
        return brake_output


if __name__ == '__main__':

    print(f"Loading lidar normalization stats from: {LIDAR_STATS_PATH}")
    LIDAR_MEAN = 0.0 
    LIDAR_STD = 1.0
    try:
        with open(LIDAR_STATS_PATH, 'r') as f: lidar_stats = json.load(f)
        LIDAR_MEAN = lidar_stats["mean"]
        LIDAR_STD = lidar_stats["std_dev"]
        print(f"  Loaded Lidar Mean: {LIDAR_MEAN:.6f}, Std Dev: {LIDAR_STD:.6f}")
        if LIDAR_STD < 1e-6:
            print("Warning: Loaded lidar standard deviation is near zero. Setting to 1.0.")
            LIDAR_STD = 1.0
    except FileNotFoundError:
        print(f"ERROR: Lidar stats file '{LIDAR_STATS_PATH}' not found.")
        print("Please ensure the data preparation script was run successfully first.")
        exit()
    except KeyError:
        print("ERROR: Lidar stats file missing 'mean' or 'std_dev' keys.")
        exit()
    print("-" * 30)


    print("Loading data splits from Pickle files for braking model...")
    try:
        with open(TRAIN_DATA_PKL_PATH_BRAKE, 'rb') as f: train_data_brake = pickle.load(f)
        print(f"Loaded {len(train_data_brake)} training samples from: {TRAIN_DATA_PKL_PATH_BRAKE}")
        with open(VAL_DATA_PKL_PATH_BRAKE, 'rb') as f: val_data_brake = pickle.load(f)
        print(f"Loaded {len(val_data_brake)} validation samples from: {VAL_DATA_PKL_PATH_BRAKE}")

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

    train_dataset_brake = BrakeDataset(train_data_brake, LIDAR_MEAN, LIDAR_STD,
                                       IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, LIDAR_HORIZONTAL_RESOLUTION, augment=True)
    val_dataset_brake = BrakeDataset(val_data_brake, LIDAR_MEAN, LIDAR_STD,
                                     IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, LIDAR_HORIZONTAL_RESOLUTION, augment=False)

    train_loader_brake = DataLoader(train_dataset_brake, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_DATA_WORKERS, pin_memory=True, persistent_workers=(NUM_DATA_WORKERS > 0))
    val_loader_brake = DataLoader(val_dataset_brake, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_DATA_WORKERS, pin_memory=True, persistent_workers=(NUM_DATA_WORKERS > 0))

    print("PyTorch DataLoaders created for Braking Model.")
    print("-" * 30)

    print("Building PyTorch CNN model for Braking Prediction...")
    model = BrakeModel(img_c=IMG_CHANNELS, img_h=IMG_HEIGHT, img_w=IMG_WIDTH,
                       lidar_res=LIDAR_HORIZONTAL_RESOLUTION).to(DEVICE)
    print(model)
    print("-" * 30)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)

    print("Starting model training for Braking Prediction...")

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

        for i, (inputs, labels) in enumerate(train_loader_brake): 
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


            if (i + 1) % 100 == 0 or i == len(train_loader_brake) - 1: 
                print(f'\rEpoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader_brake)}], Loss: {loss.item():.4f}, MAE: {mae:.4f}', end='', flush=True)


        epoch_train_loss = running_loss / train_samples
        epoch_train_mae = running_mae / train_samples
        print(f'\rEpoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {epoch_train_loss:.4f}, Train MAE: {epoch_train_mae:.4f}') 
        train_loss_history.append(epoch_train_loss)


        model.eval() 
        val_loss = 0.0
        val_mae = 0.0
        val_samples = 0

        with torch.no_grad(): 
            for inputs, labels in val_loader_brake: 
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
    plt.title('Training and Validation Loss (MSE) - Braking Model') 
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(LOSS_PLOT_PATH) 
        print(f"Loss plot saved to {LOSS_PLOT_PATH}")
    except Exception as e:
        print(f"Error saving loss plot: {e}")