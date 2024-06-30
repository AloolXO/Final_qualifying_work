#PredictSegformerApp.py
import json
import os
import shutil
import torch
import cv2
import rasterio
import numpy as np
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import fiona
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from utils.GpuUtilsApp import limit_gpu_memory, get_device
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import subprocess

# Code to hide the console window
import ctypes
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

class PredictSegformerGUI:
    def __init__(self, master):
        self.master = master
        self.normal_geometry = master.geometry()  # Initialize with the current geometry
        self.load_window_geometry()  # Load saved geometry if exists
        master.title("Работа с архитектурой Segformer")
        master.iconbitmap(r'hiik.ico')
        master.configure(bg='#E3F3F2')

        self.create_widgets()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        self.label = tk.Label(self.master, text="Работа с архитектурой Segformer", font=("Bahnschrift", 16), bg='#E3F3F2')
        self.label.pack(pady=20)

        self.btn_load_geotiff = tk.Button(self.master, text="Загрузить GeoTIFF", command=self.load_geotiff, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_load_geotiff.pack(pady=10)

        self.btn_start_prediction = tk.Button(self.master, text="Начать распознавание", command=self.start_prediction, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_start_prediction.pack(pady=10)

        self.log_text = tk.Text(self.master, height=10, width=80, bg='#E3F3F2', font=('Bahnschrift', 12), fg='black')
        self.log_text.pack(pady=10)

        self.progress = ttk.Progressbar(self.master, orient='horizontal', length=400, mode='determinate')
        self.progress.pack(pady=20)

        self.results_frame = tk.Frame(self.master, bg='#E3F3F2')
        self.results_frame.pack(pady=10)

        self.btn_return = tk.Button(self.master, text="Вернуться", command=self.return_to_main, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_return.pack(pady=10)
        
        self.btn_open_dir = tk.Button(self.master, text="Открыть директорию с результатами", command=self.open_output_dir, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_open_dir.pack(pady=10)

    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def load_geotiff(self):
        self.geotiff_path = filedialog.askopenfilename(filetypes=[("GeoTIFF files", "*.tif"), ("All files", "*.*")])
        if self.geotiff_path:
            self.log_message(f"Загружен GeoTIFF файл: {self.geotiff_path}")

    def start_prediction(self):
        if not hasattr(self, 'geotiff_path'):
            self.log_message("Пожалуйста, загрузите GeoTIFF файл.")
            return

        self.log_message("Запуск распознавания...")
        self.btn_start_prediction.config(state=tk.DISABLED)
        threading.Thread(target=self.predict).start()

    def return_to_main(self):
        self.save_window_geometry()
        self.master.destroy()
        try:
            subprocess.Popen(["python", "SegformerMenuApp.py"])
        except Exception as e:
            self.log_message(f"Ошибка при возврате к главному файлу: {e}")
            
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
            
    def open_output_dir(self):
        output_dir = './prediction_results'
        if os.path.exists(output_dir):
            subprocess.Popen(f'explorer {os.path.realpath(output_dir)}')
        else:
            self.log_message("Директория с результатами не существует.")

    def predict(self):
        model_path = './model_fold_10'
        model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        feature_extractor = SegformerImageProcessor.from_pretrained(model_path)

        device = get_device()
        model.to(device)
        limit_gpu_memory()

        output_dir = './prediction_results'
        os.makedirs(output_dir, exist_ok=True)

        file_id = os.path.splitext(os.path.basename(self.geotiff_path))[0]
        segmentation = self.run_prediction(self.geotiff_path, model, feature_extractor, device)
        self.save_prediction_results(self.geotiff_path, segmentation, output_dir, file_id)

        self.display_results(output_dir, file_id)
        self.log_message("Распознавание завершено.")
        self.btn_start_prediction.config(state=tk.NORMAL)

    def run_prediction(self, geotiff_path, model, feature_extractor, device):
        with rasterio.open(geotiff_path) as src:
            tile_size = 256 if src.width == 10980 and src.height == 10980 else 512  # Adjust tile size for specific resolution
            height, width = src.shape
            num_tiles_y = (height + tile_size - 1) // tile_size
            num_tiles_x = (width + tile_size - 1) // tile_size

            segmentation = np.zeros((height, width), dtype=np.uint8)

            for y in range(num_tiles_y):
                for x in range(num_tiles_x):
                    window = rasterio.windows.Window(x * tile_size, y * tile_size, min(tile_size, width - x * tile_size), min(tile_size, height - y * tile_size))
                    image = src.read([2, 3, 4], window=window)
                    image = np.moveaxis(image, 0, -1)
                    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

                    inputs = feature_extractor(images=image, return_tensors="pt", do_rescale=False)
                    inputs = inputs.to(device)

                    model.eval()
                    with torch.no_grad():
                        outputs = model(**inputs)
                    logits = outputs.logits
                    tile_segmentation = torch.argmax(logits, dim=1).cpu().numpy()[0]

                    # Resize tile_segmentation to match the target shape
                    tile_segmentation_resized = cv2.resize(tile_segmentation, (window.width, window.height), interpolation=cv2.INTER_NEAREST)

                    segmentation[y * tile_size:(y + 1) * tile_size, x * tile_size:x * tile_size + window.width] = tile_segmentation_resized

        return segmentation

    def save_prediction_results(self, geotiff_path, segmentation, output_dir, file_id):
        output_png_path = os.path.join(output_dir, f'{file_id}_original.png')
        output_mask_bmp_path = os.path.join(output_dir, f'{file_id}_mask.bmp')
        output_object_png_path = os.path.join(output_dir, f'{file_id}_object.png')
        output_transparent_object_png_path = os.path.join(output_dir, f'{file_id}_transparent_object.png')
        output_shape_path = os.path.join(output_dir, f'{file_id}.shp')
        output_geotiff_path = os.path.join(output_dir, os.path.basename(geotiff_path))

        # Copy the original GeoTIFF file to the output directory
        shutil.copy(geotiff_path, output_geotiff_path)

        with rasterio.open(geotiff_path) as src:
            image = src.read([2, 3, 4])
            image = np.moveaxis(image, 0, -1)

        image_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # Ensure binary mask: all intermediate values become black
        binary_segmentation = np.where(segmentation == 1, 255, 0).astype(np.uint8)

        cv2.imwrite(output_png_path, image_normalized)
        cv2.imwrite(output_mask_bmp_path, binary_segmentation)

        object_image = np.where(binary_segmentation[:, :, np.newaxis] == 255, image_normalized, 0)
        cv2.imwrite(output_object_png_path, object_image)

        transparent_object_image = np.zeros_like(image_normalized, dtype=np.uint8)
        transparent_object_image[binary_segmentation == 255] = image_normalized[binary_segmentation == 255]
        transparent_object_image = cv2.cvtColor(transparent_object_image, cv2.COLOR_RGB2RGBA)
        transparent_object_image[:, :, 3] = np.where(binary_segmentation == 255, 255, 0)
        cv2.imwrite(output_transparent_object_png_path, transparent_object_image)

        mask_for_shapes = binary_segmentation.astype(np.uint8)
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(mask_for_shapes, mask=mask_for_shapes, transform=src.transform))
        )
        polygons = [shape(geom['geometry']) for geom in results if geom['properties']['raster_val'] == 255]

        if polygons:
            schema = {
                'geometry': 'Polygon',
                'properties': {'id': 'int'},
            }
            with fiona.open(output_shape_path, 'w', 'ESRI Shapefile', schema, crs=src.crs) as shp:
                for i, polygon in enumerate(polygons):
                    shp.write({
                        'geometry': mapping(polygon),
                        'properties': {'id': i},
                    })


    def display_results(self, output_dir, file_id):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        image_files = [
            f'{file_id}_original.png',
            f'{file_id}_mask.bmp',
            f'{file_id}_object.png',
            f'{file_id}_transparent_object.png'
        ]

        for image_file in image_files:
            image_path = os.path.join(output_dir, image_file)
            if os.path.exists(image_path):
                self.add_image_button(image_path)

    def add_image_button(self, image_path):
        img = Image.open(image_path)
        img.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(img)

        btn_image = tk.Button(self.results_frame, image=img_tk, command=lambda p=image_path: self.show_fullscreen_image(p))
        btn_image.image = img_tk
        btn_image.pack(side=tk.LEFT, padx=10, pady=10)

    def show_fullscreen_image(self, image_path):
        fullscreen = tk.Toplevel(self.master)
        fullscreen.attributes('-fullscreen', True)
        fullscreen.configure(bg='#E3F3F2')

        img = Image.open(image_path)
        img_tk = ImageTk.PhotoImage(img)

        canvas = tk.Canvas(fullscreen, bg='#E3F3F2')
        canvas.pack(fill=tk.BOTH, expand=True)

        def resize_image(event):
            canvas_width = event.width
            canvas_height = event.height
            img_resized = img.copy()
            img_resized.thumbnail((canvas_width, canvas_height))
            img_tk_resized = ImageTk.PhotoImage(img_resized)
            canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk_resized)
            canvas.image = img_tk_resized

        canvas.bind('<Configure>', resize_image)

        btn_close = tk.Button(fullscreen, text="Закрыть", command=fullscreen.destroy, bg='white', fg='black', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        btn_close.place(relx=0.95, rely=0.05, anchor=tk.NE)

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictSegformerGUI(root)
    root.mainloop()
