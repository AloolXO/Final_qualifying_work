#GeoTIFFtoPNGConverterApp.py
import json
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar

import concurrent.futures
import threading

import rasterio
from PIL import Image, ImageTk, ImageEnhance
import queue
import subprocess

# Code to hide the console window
import ctypes
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

class GeoTIFFtoPNGConverter(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Конвертация GeoTIFF в PNG")
        self.normal_geometry = self.geometry()  # Initialize with the current geometry
        self.load_window_geometry()  # Load saved geometry if exists
        self.iconbitmap(r'hiik.ico')
        self.geometry("800x600")
        self.configure(bg='#E3F3F2')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        
        self.file_paths = tk.StringVar()
        self.status_text = tk.StringVar(value="Waiting for operation...")
        self.converting = False
        self.converted_count = 0
        self.total_files = 0
        
        self.lock = threading.Lock()
        
        self.create_widgets()
    
    def create_widgets(self):
        self.canvas = tk.Canvas(self, width=800, height=600, bg='#E3F3F2')
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        
        self.inner_frame = tk.Frame(self.canvas, bg='#E3F3F2')
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        
        tk.Label(self.inner_frame, text="Файлы для конвертации: ", wraplength=200, justify=tk.LEFT, bg='#E3F3F2', font=('Bahnschrift', 12)).grid(row=0, column=0, sticky=tk.W)
        tk.Label(self.inner_frame, textvariable=self.file_paths, wraplength=400, justify=tk.LEFT, bg='#E3F3F2', font=('Bahnschrift', 12)).grid(row=0, column=1, sticky=tk.W)
        
        self.select_button = tk.Button(self.inner_frame, text="Выберите файлы", command=self.select_files, width=20, bg='#003e87', fg='white', font=('Bahnschrift', 12), relief='raised', bd=3)
        self.select_button.grid(row=1, columnspan=2, pady=5)
        
        self.convert_button = tk.Button(self.inner_frame, text="Конвертация", command=self.convert_files, width=20, bg='#003e87', fg='white', font=('Bahnschrift', 12), relief='raised', bd=3)
        self.convert_button.grid(row=2, columnspan=2, pady=5)
        
        tk.Label(self.inner_frame, text="Operation Status: ", wraplength=200, justify=tk.LEFT, bg='#E3F3F2', font=('Bahnschrift', 12)).grid(row=3, column=0, sticky=tk.W)
        tk.Label(self.inner_frame, textvariable=self.status_text, wraplength=400, justify=tk.LEFT, bg='#E3F3F2', font=('Bahnschrift', 12)).grid(row=3, column=1, sticky=tk.W)
        
        self.progress = Progressbar(self.inner_frame, orient=tk.HORIZONTAL, length=500, mode='determinate')
        self.progress.grid(row=4, columnspan=2, pady=5)
        
        self.image_frame = tk.Frame(self.inner_frame, bg='#E3F3F2')
        self.image_frame.grid(row=5, columnspan=2, pady=10, sticky="we")
        
        self.btn_return = tk.Button(self.inner_frame, text="Вернуться", command=self.return_to_main, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_return.grid(row=6, columnspan=2, pady=10)

        self.queue = queue.Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    
    def on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def on_mousewheel(self, event):
        self.canvas.yview_scroll(-1 * int(event.delta / 120), "units")
        
    def select_files(self):
        if self.converting:
            messagebox.showwarning("Conversion in progress", "Please wait until the current conversion is completed.")
            return
        
        file_paths = filedialog.askopenfilenames(filetypes=[("GeoTIFF files", "*.tif"), ("All files", "*.*")], multiple=True)
        if file_paths:
            self.file_paths.set(", ".join(file_paths))
    
    def convert_files(self):
        if self.converting:
            messagebox.showwarning("Conversion in progress", "Please wait until the current conversion is completed.")
            return
        
        file_paths = self.file_paths.get().split(", ")
        if not file_paths or file_paths == [""]:
            messagebox.showwarning("No files selected", "Please select GeoTIFF files to convert.")
            return
        
        self.converting = True
        self.converted_count = 0
        self.status_text.set("Starting conversion...")
        self.progress['value'] = 0

        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        self.total_files = len(file_paths)
        for file_path in file_paths:
            self.executor.submit(self.convert_file, file_path.strip())
        
        self.convert_button.config(state=tk.DISABLED)
    
    def update_progress(self, progress_value, file_path, img, error=False):
        if error:
            self.status_text.set(f"Error converting file: {file_path}")
            messagebox.showerror("Error", f"Error converting {file_path}")
        else:
            self.progress['value'] = progress_value
            self.status_text.set(f"Converting file: {file_path}")
            
            img_label = tk.Label(self.image_frame, image=img, bg='white')
            img_label.image = img
            img_label.grid(row=self.converted_count // 6, column=self.converted_count % 6, padx=5, pady=5, sticky="we")
            
            self.converted_count += 1
            
            self.canvas.update_idletasks()
            self.on_canvas_configure(None)
            
            if self.converted_count == self.total_files:
                self.converting = False
                self.convert_button.config(state=tk.NORMAL)
                self.status_text.set("Conversion completed.")
                messagebox.showinfo("Conversion completed", "All files have been successfully converted.")
    
    def convert_file(self, file_path):
        try:
            with rasterio.open(file_path) as src:
                image_data = src.read([3, 2, 1])
                image_data = np.moveaxis(image_data, 0, -1)
                image_data_min = image_data.min()
                image_data_max = image_data.max()
                if image_data_max != image_data_min:
                    image_data = ((image_data - image_data_min) / (image_data_max - image_data_min) * 255).astype(np.uint8)
                else:
                    image_data = np.zeros_like(image_data, dtype=np.uint8)
                
                img = Image.fromarray(image_data)
                
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                
                brightness_enhancer = ImageEnhance.Brightness(img)
                img = brightness_enhancer.enhance(1.5)
                
                contrast_enhancer = ImageEnhance.Contrast(img)
                img = contrast_enhancer.enhance(1.4)
                
                output_dir = os.path.join(os.path.dirname(file_path), "PNG")
                with self.lock:
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                
                png_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + '.png')
                with self.lock:
                    img.save(png_path)
                
                img.thumbnail((300, 300))
                img_tk = ImageTk.PhotoImage(img)

                progress_value = ((self.converted_count + 1) / self.total_files) * 100
                self.queue.put((progress_value, file_path, img_tk, False))
                self.update_gui()

        except Exception as e:
            self.queue.put((0, file_path, None, True))
            messagebox.showerror("Error", f"Error converting {file_path}: {e}")

    def update_gui(self):
        while not self.queue.empty():
            progress_value, file_path, img_tk, error = self.queue.get()
            self.update_progress(progress_value, file_path, img_tk, error)

    def return_to_main(self):
        self.save_window_geometry()
        self.destroy()
        try:
            subprocess.Popen(["python", "Final_qualifying_work.py"])
        except Exception as e:
            self.status_text.set(f"Ошибка при возврате к главному файлу: {e}")
            
    def save_window_geometry(self):
        if self.state() == 'normal':
            self.normal_geometry = self.geometry()  # Save the current normal size and position
        geometry = self.normal_geometry
        is_maximized = self.state() == 'zoomed'
        with open("window_geometry.json", "w") as f:
            json.dump({"geometry": geometry, "is_maximized": is_maximized}, f)
        
    def load_window_geometry(self):
        try:
            with open("window_geometry.json", "r") as f:
                data = json.load(f)
                self.normal_geometry = data["geometry"]
                self.geometry(self.normal_geometry)
                if data["is_maximized"]:
                    self.state('zoomed')
        except (FileNotFoundError, KeyError):
            pass
  
    def on_closing(self):
        self.save_window_geometry()
        self.destroy()

if __name__ == "__main__":
    app = GeoTIFFtoPNGConverter()
    app.mainloop()
