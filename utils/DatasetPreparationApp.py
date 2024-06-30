# utils/DatasetPreparationApp.py
import json
import os
import numpy as np
import cv2
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import fiona
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess

# Code to hide the console window
import ctypes
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

# Define directory paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
PROCESSED_DATASET_DIR = os.path.join(PROJECT_DIR, "processed_dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
PROCESSED_IMAGES_DIR = os.path.join(PROCESSED_DATASET_DIR, "images")
SHAPES_DIR = os.path.join(PROCESSED_DATASET_DIR, "shapes")

# Ensure the directories for processed data and shape files exist
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
os.makedirs(SHAPES_DIR, exist_ok=True)

# Path to the main application file
MAIN_FILE_PATH = os.path.abspath(os.path.join(PROJECT_DIR, os.pardir, "SegformerMenuApp.py"))

def convert_geotiff_to_png_and_shape(geotiff_path, png_path, shp_path, tiff_output_path, mask_path):
    with rasterio.open(geotiff_path) as src:
        image = src.read([2, 3, 4])
        image = np.moveaxis(image, 0, -1)

        image_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        image_normalized = cv2.flip(image_normalized, 0)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not found at path: {mask_path}")

        mask = cv2.flip(mask, 0)

        tiff_output_file = os.path.join(tiff_output_path, os.path.basename(geotiff_path))
        with rasterio.open(tiff_output_file, 'w', **src.meta) as dst:
            dst.write(src.read())

        mask_for_shapes = (mask > 0).astype(np.uint8)
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(mask_for_shapes, mask=mask_for_shapes, transform=src.transform))
        )
        polygons = [shape(geom['geometry']) for geom in results if geom['properties']['raster_val'] == 1]

        if polygons:
            schema = {
                'geometry': 'Polygon',
                'properties': {'id': 'int'},
            }
            with fiona.open(shp_path, 'w', 'ESRI Shapefile', schema) as shp:
                for i, polygon in enumerate(polygons):
                    shp.write({
                        'geometry': mapping(polygon),
                        'properties': {'id': i},
                    })

        cv2.imwrite(png_path, image_normalized)

def process_dataset(dataset_path, output_path, progress_callback=None):
    geotiffs_path = os.path.join(dataset_path, 'geotiffs')
    masks_path = os.path.join(dataset_path, 'masks')
    images_path = os.path.join(output_path, 'images')
    shapes_path = os.path.join(output_path, 'shapes')
    tiff_output_path = os.path.join(output_path, 'geotiffs')

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(shapes_path, exist_ok=True)
    os.makedirs(tiff_output_path, exist_ok=True)

    geotiff_files = [f for f in os.listdir(geotiffs_path) if f.endswith('.tif')]
    mask_files = {os.path.splitext(f)[0]: os.path.join(masks_path, f) for f in os.listdir(masks_path) if f.endswith('.bmp')}

    total_files = len(geotiff_files)
    processed_files = 0

    for geotiff_file in geotiff_files:
        file_id = os.path.splitext(geotiff_file)[0]
        geotiff_path = os.path.join(geotiffs_path, geotiff_file)
        png_path = os.path.join(images_path, f'{file_id}.png')
        shp_path = os.path.join(shapes_path, f'{file_id}.shp')
        if file_id in mask_files:
            mask_path = mask_files[file_id]
            convert_geotiff_to_png_and_shape(geotiff_path, png_path, shp_path, tiff_output_path, mask_path)

        processed_files += 1
        if progress_callback:
            progress_callback(processed_files / total_files * 100)

def split_data(images, masks, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = kf.split(images)
    return [(train_idx, val_idx) for train_idx, val_idx in splits]

class MiningDataset(Dataset):
    def __init__(self, image_paths, mask_paths, feature_extractor, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
            raise IndexError("index out of range")
        
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")
        if mask is None:
            raise ValueError(f"Mask not found at path: {mask_path}")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        inputs = self.feature_extractor(images=image, return_tensors="pt", do_rescale=False)
        mask = mask.clone().detach().long()

        if len(mask.shape) == 3:
            mask = mask.squeeze(0)

        return inputs["pixel_values"].squeeze(0), mask

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# GUI for data preparation
class DataPreparationApp:
    def __init__(self, master):
        self.master = master
        self.normal_geometry = master.geometry()  # Initialize with the current geometry
        self.load_window_geometry()  # Load saved geometry if exists
        master.title("Подготовка данных/датасета")
        master.iconbitmap(r'hiik.ico')
        master.configure(bg='#E3F3F2')

        self.label = tk.Label(master, text="Подготовка данных", font=("Bahnschrift", 16), bg='#E3F3F2')
        self.label.pack(pady=20)

        self.progress = ttk.Progressbar(master, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=20)

        self.result_label = tk.Label(master, text="", bg='#E3F3F2', font=('Bahnschrift', 12))
        self.result_label.pack()

        self.prepare_button = tk.Button(master, text="Подготовить данные", command=self.prepare_data, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.prepare_button.pack(pady=10)

        self.return_button = tk.Button(master, text="Вернуться на главную", command=self.go_back, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.return_button.pack(pady=10)
        
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def process_data(self):
        def progress_callback(progress):
            self.progress['value'] = progress
            self.master.update_idletasks()

        process_dataset(DATASET_DIR, PROCESSED_DATASET_DIR, progress_callback)
        self.result_label.config(text=f"Обработка завершена. Данные сохранены в {PROCESSED_DATASET_DIR}")
        self.prepare_button.config(state=tk.NORMAL)
        self.return_button.config(state=tk.NORMAL)

    def prepare_data(self):
        try:
            self.prepare_button.config(state=tk.DISABLED)
            self.return_button.config(state=tk.DISABLED)
            self.process_data()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def go_back(self):
        self.save_window_geometry()
        self.master.destroy()
        subprocess.Popen(["python", MAIN_FILE_PATH])
        
    def save_window_geometry(self):
        if self.master.state() == 'normal':
            self.normal_geometry = self.master.geometry()  # Save the current normal size and position
        geometry = self.normal_geometry
        is_maximized = self.master.state() == 'zoomed'
        with open("window_geometry.json", "w") as f:
            json.dump({"geometry": geometry, "is_maximized": is_maximized}, f)
            
    def load_window_geometry(self):
        try:
            with open("window_geometry.json", "r") as f:
                data = json.load(f)
                self.normal_geometry = data["geometry"]
                self.master.geometry(self.normal_geometry)
                if data["is_maximized"]:
                    self.master.state('zoomed')
        except (FileNotFoundError, KeyError):
            pass
  
    def on_closing(self):
        self.save_window_geometry()
        self.master.destroy()

def main():
    root = tk.Tk()
    app = DataPreparationApp(root)
    root.minsize(500, 500)
    root.mainloop()

if __name__ == "__main__":
    main()
