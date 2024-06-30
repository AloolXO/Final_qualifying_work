#SatelliteConverterApp.py
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import glob
import subprocess
from tkinter import ttk
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import threading
import rasterio
import shutil

# Code to hide the console window
import ctypes
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

class SatelliteConverterApp:
    def __init__(self, root):
        self.root = root
        self.normal_geometry = root.geometry()  # Initialize with the current geometry
        self.load_window_geometry()  # Load saved geometry if exists
        self.root.title("Конвертация JP2 в GeoTIFF")
        self.root.iconbitmap(r'hiik.ico')
        self.root.configure(bg='#E3F3F2')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


        self.selected_bands_dir = ""
        self.selected_metadata_file = ""
        self.selected_output_dir = ""

        # Создание метки с инструкциями
        self.instruction_label = tk.Label(root, text="Выберите компоненты для конвертации", padx=10, pady=10, bg='#E3F3F2', font=('Bahnschrift', 14))
        self.instruction_label.pack()

        # Создание кнопки для выбора директории с файлами каналов
        self.bands_dir_button = tk.Button(root, text="Выберите директорию с файлами каналов", command=self.select_bands_directory, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.bands_dir_button.pack(pady=10)

        # Создание кнопки для выбора файла метаданных
        self.metadata_file_button = tk.Button(root, text="Выберите файл метаданных", command=self.select_metadata_file, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.metadata_file_button.pack(pady=10)

        # Создание кнопки для выбора директории сохранения результатов
        self.output_dir_button = tk.Button(root, text="Выберите директорию для сохранения результатов", command=self.select_output_directory, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.output_dir_button.pack(pady=10)

        # Создание кнопки для запуска конвертации
        self.convert_button = tk.Button(root, text="Конвертировать", command=self.start_conversion_thread, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.convert_button.pack(pady=10)

        # Создание кнопки для объединения слоев в RGB
        self.merge_button = tk.Button(root, text="Объединить слои в RGB", command=self.start_merge_thread, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.merge_button.pack(pady=10)

        # Создание текстовых полей для отображения статуса операций
        self.bands_dir_status = tk.Label(root, text="", padx=10, pady=5, bg='#E3F3F2', font=('Bahnschrift', 12))
        self.bands_dir_status.pack()

        self.metadata_file_status = tk.Label(root, text="", padx=10, pady=5, bg='#E3F3F2', font=('Bahnschrift', 12))
        self.metadata_file_status.pack()

        self.output_dir_status = tk.Label(root, text="", padx=10, pady=5, bg='#E3F3F2', font=('Bahnschrift', 12))
        self.output_dir_status.pack()

        self.convert_status = tk.Label(root, text="", padx=10, pady=5, bg='#E3F3F2', font=('Bahnschrift', 12))
        self.convert_status.pack()

        self.merge_status = tk.Label(root, text="", padx=10, pady=5, bg='#E3F3F2', font=('Bahnschrift', 12))
        self.merge_status.pack()

        # Создание прогресс-бара
        self.progress_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack()

        # Кнопка для возвращения на главное окно
        self.return_button = tk.Button(root, text="На главное окно", command=self.return_to_main_window, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.return_button.pack(pady=10)

    def select_bands_directory(self):
        bands_dir = filedialog.askdirectory()
        if bands_dir:
            if self.selected_bands_dir == bands_dir:
                messagebox.showwarning("Предупреждение", "Вы выбрали ту же самую директорию. Пожалуйста, выберите другую директорию.")
            else:
                self.selected_bands_dir = bands_dir
                self.bands_dir_status.config(text=f"Выбрана директория с файлами каналов: {bands_dir}")

    def select_metadata_file(self):
        metadata_file = filedialog.askopenfilename(filetypes=[("Metadata files", "*.xml"), ("All files", "*.*")])
        if metadata_file:
            self.selected_metadata_file = metadata_file
            self.metadata_file_status.config(text=f"Выбран файл метаданных: {metadata_file}")

    def select_output_directory(self):
        output_dir = filedialog.askdirectory()
        if output_dir:
            if self.selected_output_dir == output_dir:
                messagebox.showwarning("Предупреждение", "Вы выбрали ту же самую директорию. Пожалуйста, выберите другую директорию.")
            else:
                self.selected_output_dir = output_dir
                self.output_dir_status.config(text=f"Выбрана директория для сохранения результатов: {output_dir}")

    def start_conversion_thread(self):
        self.convert_status.config(text="Начало конвертации...")
        self.toggle_buttons(state=tk.DISABLED)
        conversion_thread = threading.Thread(target=self.convert_files)
        conversion_thread.start()

    def convert_files(self):
        if not (self.selected_bands_dir and self.selected_metadata_file and self.selected_output_dir):
            messagebox.showerror("Ошибка", "Пожалуйста, выберите все компоненты.")
            self.toggle_buttons(state=tk.NORMAL)
            return

        try:
            with rasterio.open(self.selected_metadata_file) as src_metadata:
                projection = src_metadata.crs
                geotransform = src_metadata.transform

            band_files = glob.glob(os.path.join(self.selected_bands_dir, "*.jp2"))
            total_files = len(band_files)

            for i, band_file in enumerate(band_files, start=1):
                with rasterio.open(band_file) as src_band:
                    band_data = src_band.read(1)

                scaler = MinMaxScaler(feature_range=(0, 1))
                band_data_normalized = scaler.fit_transform(band_data.reshape(-1, 1)).reshape(band_data.shape)

                output_file = os.path.join(self.selected_output_dir, os.path.basename(band_file).replace(".jp2", ".tif"))
                with rasterio.open(
                    output_file, 'w',
                    driver='GTiff',
                    height=band_data_normalized.shape[0],
                    width=band_data_normalized.shape[1],
                    count=1,
                    dtype=rasterio.float32,
                    crs=projection,
                    transform=geotransform,
                    compress='lzw'
                ) as dst:
                    dst.write(band_data_normalized, 1)

                progress_value = (i / total_files) * 100
                self.progress_bar["value"] = progress_value
                self.root.update_idletasks()

            self.convert_status.config(text="Конвертация завершена.")
            messagebox.showinfo("Успех", "Конвертация выполнена успешно.")
        except Exception as e:
            self.convert_status.config(text="Ошибка конвертации.")
            messagebox.showerror("Ошибка", f"Ошибка во время конвертации: {str(e)}")
        
        self.toggle_buttons(state=tk.NORMAL)

    def start_merge_thread(self):
        self.merge_status.config(text="Начало объединения слоев...")
        self.toggle_buttons(state=tk.DISABLED)
        merge_thread = threading.Thread(target=self.merge_layers)
        merge_thread.start()

    def merge_layers(self):
        if not self.selected_output_dir:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите директорию для сохранения результатов.")
            self.toggle_buttons(state=tk.NORMAL)
            return

        try:
            self.merge_status.config(text="Начало объединения слоев...")

            output_files = glob.glob(os.path.join(self.selected_output_dir, "*.tif"))

            output_merged_tif_file = os.path.join(self.selected_output_dir, "merged.tif")
            with rasterio.open(output_files[0]) as src0:
                meta = src0.meta

            meta.update(count=len(output_files))

            with rasterio.open(output_merged_tif_file, 'w', **meta) as dst:
                for idx, file in enumerate(output_files):
                    with rasterio.open(file) as src:
                        band_data = src.read(1)
                        band_data_normalized = (band_data - band_data.min()) / (band_data.max() - band_data.min())
                        dst.write_band(idx + 1, band_data_normalized)

            self.merge_status.config(text="Объединение слоев завершено.")
            messagebox.showinfo("Успех", f"Объединение слоев выполнено успешно. Результаты сохранены в {output_merged_tif_file}.")
        except Exception as e:
            self.merge_status.config(text="Ошибка объединения слоев.")
            messagebox.showerror("Ошибка", f"Ошибка во время объединения слоев: {str(e)}")
        
        self.toggle_buttons(state=tk.NORMAL)

    def toggle_buttons(self, state):
        self.bands_dir_button.config(state=state)
        self.metadata_file_button.config(state=state)
        self.output_dir_button.config(state=state)
        self.convert_button.config(state=state)
        self.merge_button.config(state=state)
        self.return_button.config(state=state)

    def return_to_main_window(self):
        self.save_window_geometry()
        self.root.destroy()
        subprocess.Popen(["python", "Final_qualifying_work.py"])
        
    def save_window_geometry(self):
        if self.root.state() == 'normal':
            self.normal_geometry = self.root.geometry()  # Save the current normal size and position
        geometry = self.normal_geometry
        is_maximized = self.root.state() == 'zoomed'
        with open("window_geometry.json", "w") as f:
            json.dump({"geometry": geometry, "is_maximized": is_maximized}, f)
        
    def load_window_geometry(self):
        try:
            with open("window_geometry.json", "r") as f:
                data = json.load(f)
                self.normal_geometry = data["geometry"]
                self.root.geometry(self.normal_geometry)
                if data["is_maximized"]:
                    self.root.state('zoomed')
        except (FileNotFoundError, KeyError):
            pass
  
    def on_closing(self):
        self.save_window_geometry()
        self.root.destroy()

# Создание главного окна
root = tk.Tk()
app = SatelliteConverterApp(root)

# Запуск главного цикла обработки событий
root.mainloop()
