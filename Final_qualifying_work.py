#Final_qualifying_work.py
import json
import tkinter as tk
from tkinter import filedialog
import subprocess

# Code to hide the console window
import ctypes
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

class MainWindow:
    def __init__(self, master):
        self.master = master
        self.normal_geometry = master.geometry()  # Initialize with the current geometry
        self.load_window_geometry()  # Load saved geometry if exists
        master.title("Выпускная квалификационная работа, Леуненко АО, ХИИК СибГУТИ")
        master.iconbitmap(r'hiik.ico')
        master.configure(bg='#E3F3F2')
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Поля для изображений
        self.image_frame = tk.Frame(master, bg='#E3F3F2')
        self.image_frame.pack(pady=10)

        # Загрузка изображения
        self.load_image()

        # Приветственный текст на прозрачном фоне
        self.label = tk.Label(master, text="Добро пожаловать!", font=("Bahnschrift", 16), bg='#E3F3F2')
        self.label.pack(pady=20)

        # Кнопки
        self.btn_preprocessing = tk.Button(master, text="Предобработка файлов проекта", command=self.open_folder_explorer, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_preprocessing.pack(pady=10)

        self.btn_conversion = tk.Button(master, text="Конвертация спутниковых снимков", command=self.open_conversion_window, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_conversion.pack(pady=10)

        self.btn_cropping = tk.Button(master, text="Кадрирование изображения GeoTIFF", command=self.open_cropper_window, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_cropping.pack(pady=10)
        
        self.btn_conversionpng = tk.Button(master, text="Конвертация GeoTIFF в PNG", command=self.open_conversionpng_window, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_conversionpng.pack(pady=10)
        
        self.btn_segformet = tk.Button(master, text="Работа с распознованием объектов", command=self.open_segformet_window, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_segformet.pack(pady=10)
        
        self.btn_exit = tk.Button(master, text="Выход", command=master.quit, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_exit.pack(pady=10)

    def load_image(self):
        try:
            img = tk.PhotoImage(file="hiik.png")
            img = img.subsample(4)  # уменьшение размера изображения
            label = tk.Label(self.image_frame, image=img, bg='#E3F3F2')
            label.image = img  # сохранение ссылки на изображение, чтобы избежать сбора мусора
            label.pack(side="left", padx=10, pady=10)
        except tk.TclError:
            error_label = tk.Label(self.image_frame, text="Ошибка загрузки изображения.", bg='#E3F3F2', fg='red')
            error_label.pack(side="left", padx=10, pady=10)

    def open_folder_explorer(self):
        self.save_window_geometry()
        self.master.destroy()
        try:
            subprocess.Popen(["python", "FolderExplorerApp.py"])
        except Exception as e:
            print(f"Ошибка при открытии FolderExplorerApp: {e}")

    def open_conversion_window(self):
        self.save_window_geometry()
        self.master.destroy()
        try:
            subprocess.Popen(["python", "SatelliteConverterApp.py"])
        except Exception as e:
            print(f"Ошибка при открытии SatelliteConverterApp: {e}")

    def open_cropper_window(self):
        self.save_window_geometry()
        self.master.destroy()
        try:
            subprocess.Popen(["python", "GeoTIFFCropperApp.py"])
        except Exception as e:
            print(f"Ошибка при открытии GeoTIFFCropperApp: {e}")
            
    def open_conversionpng_window(self):
        self.save_window_geometry()
        self.master.destroy()
        try:
            subprocess.Popen(["python", "ConvertGeoTIFFTOPNGApp.py"])
        except Exception as e:
            print(f"Ошибка при открытии ConvertGeoTIFFTOPNGApp: {e}")
            
    def open_segformet_window(self):
        self.save_window_geometry()
        self.master.destroy()
        try:
            subprocess.Popen(["python", "SegformerMenuApp.py"])
        except Exception as e:
            print(f"Ошибка при открытии SegformerMenuApp: {e}")
            
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
    app = MainWindow(root)
    root.minsize(500, 500)  # Минимальный размер окна
    root.mainloop()

if __name__ == "__main__":
    main()
