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
GYRO_STATS_PATH = os.path.join(DATA_DIR, "gyro_norm_stats.json")
TRAIN_DATA_PKL_PATH = os.path.join(DATA_DIR, "train_data.pkl")
VAL_DATA_PKL_PATH = os.path.join(DATA_DIR, "val_data.pkl")

IMG_HEIGHT = 66
IMG_WIDTH = 200
IMG_CHANNELS = 3
GYRO_AXIS_INDEX = 2
GYRO_INPUT_FEATURES = 1

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_DATA_WORKERS = 4

EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.2

MODEL_SAVE_PATH = 'steering_model_camera_gyro_pytorch.pth'
LOSS_PLOT_PATH = 'steering_model_loss_plot.png' 

class DrivingDataset(Dataset):
    def __init__(self, data_list, gyro_axis_idx, gyro_mean, gyro_std, img_h, img_w, img_c, augment=False):
        self.data_list = data_list
        self.gyro_axis_idx = gyro_axis_idx
        self.gyro_mean = gyro_mean
        self.gyro_std = gyro_std
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.augment = augment
        self.gyro_effective_std = self.gyro_std if self.gyro_std >= 1e-6 else 1.0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        image_path = data_item['camera_path']
        gyro_data_tuple = data_item['gyro_data']
        steering_angle = data_item['steering_angle']
        try:
            img = cv2.imread(image_path)
            if img is None: raise IOError(f"Failed load: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_w, self.img_h))
            if self.augment:
                if random.random() > 0.5: 
                    delta = random.uniform(-0.2, 0.2) * 255
                    img = cv2.add(img, delta)
                    img = np.clip(img, 0, 255).astype(np.uint8)
                if random.random() > 0.5: 
                    img = cv2.flip(img, 1)
                    steering_angle = -steering_angle
            img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        except Exception as e:
             img_tensor = torch.zeros((self.img_c, self.img_h, self.img_w), dtype=torch.float32)
        try:
            gyro_value = gyro_data_tuple[self.gyro_axis_idx]
            normalized_gyro = (gyro_value - self.gyro_mean) / self.gyro_effective_std
        except IndexError: normalized_gyro = 0.0
        except Exception: normalized_gyro = 0.0
        gyro_tensor = torch.tensor([normalized_gyro], dtype=torch.float32)
        steering_tensor = torch.tensor([steering_angle], dtype=torch.float32)
        return {'image': img_tensor, 'gyro': gyro_tensor}, steering_tensor

class SteeringModel(nn.Module):
    def __init__(self, img_c=3, img_h=66, img_w=200, gyro_features=1):
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
        dummy_input = torch.randn(1, img_c, img_h, img_w)
        flattened_size = self.camera_path(dummy_input).shape[1]
        self.gyro_path = nn.Sequential(
            nn.Linear(gyro_features, 16), nn.ReLU(),
            nn.BatchNorm1d(16)
        )
        self.combined_path = nn.Sequential(
            nn.Linear(flattened_size + 16, 100), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(100, 50), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, image, gyro):
        img_features = self.camera_path(image)
        gyro_features = self.gyro_path(gyro)
        combined_features = torch.cat((img_features, gyro_features), dim=1)
        steering_output = self.combined_path(combined_features)
        return steering_output

if __name__ == '__main__':

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print("-" * 30)

    print(f"Loading gyro normalization stats from: {GYRO_STATS_PATH}")
    try:
        with open(GYRO_STATS_PATH, 'r') as f: gyro_stats = json.load(f)
        GYRO_MEAN = gyro_stats["mean"]
        GYRO_STD = gyro_stats["std_dev"]
        print(f"  Gyro Mean: {GYRO_MEAN:.6f}, Std Dev: {GYRO_STD:.6f}")
        if GYRO_STD < 1e-6: GYRO_STD = 1.0
    except FileNotFoundError: print(f"ERROR: Gyro stats file '{GYRO_STATS_PATH}' not found."); exit()
    except KeyError: print("ERROR: Gyro stats file missing 'mean' or 'std_dev' keys."); exit()
    print("-" * 30)

    print("Loading data splits from Pickle files...")
    try:
        with open(TRAIN_DATA_PKL_PATH, 'rb') as f: train_data = pickle.load(f)
        print(f"Loaded {len(train_data)} training samples from: {TRAIN_DATA_PKL_PATH}")
        with open(VAL_DATA_PKL_PATH, 'rb') as f: val_data = pickle.load(f)
        print(f"Loaded {len(val_data)} validation samples from: {VAL_DATA_PKL_PATH}")
    except FileNotFoundError as e: print(f"ERROR: Data split file not found: {e}. Run prep script first."); exit()
    except Exception as e: print(f"An unexpected error occurred loading data: {e}"); exit()
    print("-" * 30)

    train_dataset = DrivingDataset(train_data, GYRO_AXIS_INDEX, GYRO_MEAN, GYRO_STD, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, augment=True)
    val_dataset = DrivingDataset(val_data, GYRO_AXIS_INDEX, GYRO_MEAN, GYRO_STD, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_DATA_WORKERS, pin_memory=True, persistent_workers=(NUM_DATA_WORKERS > 0))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_DATA_WORKERS, pin_memory=True, persistent_workers=(NUM_DATA_WORKERS > 0))
    print("PyTorch DataLoaders created.")
    print("-" * 30)

    print("Building PyTorch CNN model...")
    model = SteeringModel(img_c=IMG_CHANNELS, img_h=IMG_HEIGHT, img_w=IMG_WIDTH, gyro_features=GYRO_INPUT_FEATURES).to(DEVICE)
    print(model)
    print("-" * 30)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)

    print("Starting model training...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    start_time = time.time()

    train_loss_history = []
    val_loss_history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        train_samples = 0
        for i, (inputs, labels) in enumerate(train_loader):
            img_batch = inputs['image'].to(DEVICE)
            gyro_batch = inputs['gyro'].to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(img_batch, gyro_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img_batch.size(0)
            mae = torch.abs(outputs - labels).mean().item()
            running_mae += mae * img_batch.size(0)
            train_samples += img_batch.size(0)
            if (i + 1) % 100 == 0 or i == len(train_loader) - 1:
                 print(f'\rEpoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, MAE: {mae:.4f}', end='')

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
                img_batch = inputs['image'].to(DEVICE)
                gyro_batch = inputs['gyro'].to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(img_batch, gyro_batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * img_batch.size(0)
                mae = torch.abs(outputs - labels).mean().item()
                val_mae += mae * img_batch.size(0)
                val_samples += img_batch.size(0)

        epoch_val_loss = val_loss / val_samples
        epoch_val_mae = val_mae / val_samples
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Val Loss: {epoch_val_loss:.4f}, Val MAE: {epoch_val_mae:.4f}')
        val_loss_history.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            print(f'Validation loss improved ({best_val_loss:.4f} --> {epoch_val_loss:.4f}). Saving model...')
            best_val_loss = epoch_val_loss
            best_model_state = deepcopy(model.state_dict())
            torch.save(best_model_state, MODEL_SAVE_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'Validation loss did not improve from {best_val_loss:.4f}. ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})')

        scheduler.step(epoch_val_loss)

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.')
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
    plt.title('Training and Validation Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(LOSS_PLOT_PATH) 
        print(f"Loss plot saved to {LOSS_PLOT_PATH}")
    except Exception as e:
        print(f"Error saving loss plot: {e}")
    plt.show()