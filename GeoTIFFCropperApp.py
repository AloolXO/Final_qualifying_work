#GeoTIFFCropperApp.py
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import os
import threading
import subprocess
import rasterio
from rasterio.windows import Window

# Code to hide the console window
import ctypes
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

class GeoTIFFCropperApp:
    def __init__(self, root):
        self.root = root
        self.normal_geometry = root.geometry()  # Initialize with the current geometry
        self.load_window_geometry()  # Load saved geometry if exists
        self.root.title("Кадрирование GeoTIFF")
        self.root.iconbitmap(r'hiik.ico')
        self.root.configure(bg='#E3F3F2')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.crop_width = tk.IntVar()
        self.crop_height = tk.IntVar()
        self.crop_width.set(512)
        self.crop_height.set(512)

        # Создание графического интерфейса
        tk.Label(root, text="Выбор файла кадрирования: ", font=('Bahnschrift', 12), bg='#E3F3F2').grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.input_file_label = tk.Label(root, text="", font=('Bahnschrift', 12), bg='#E3F3F2', width=50)
        self.input_file_label.grid(row=0, column=1, sticky="w", padx=10, pady=5, columnspan=3)

        tk.Button(root, text="Выбрать файл", command=self.select_input_file, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0).grid(row=1, column=0, columnspan=4, pady=5)

        tk.Label(root, text="Разрешение кадрирования:", font=('Bahnschrift', 12), bg='#E3F3F2').grid(row=2, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(root, textvariable=self.crop_width, font=('Bahnschrift', 12), width=10).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        tk.Entry(root, textvariable=self.crop_height, font=('Bahnschrift', 12), width=10).grid(row=2, column=2, padx=5, pady=5, sticky="w")
        tk.Button(root, text="Сохранить размеры", command=self.save_crop_size, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0).grid(row=2, column=3, padx=10, pady=5)

        self.crop_button = tk.Button(root, text="Кадрирование", command=self.start_crop, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.crop_button.grid(row=3, column=0, columnspan=4, pady=5)

        tk.Label(root, text="Статус операции: ", font=('Bahnschrift', 12), bg='#E3F3F2').grid(row=4, column=0, sticky="w", padx=10, pady=5)
        self.operation_status_label = tk.Label(root, text="", font=('Bahnschrift', 12), bg='#E3F3F2')
        self.operation_status_label.grid(row=4, column=1, sticky="w", padx=10, pady=5, columnspan=3)

        self.progress_bar = Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress_bar.grid(row=5, column=0, columnspan=4, sticky="we", padx=10, pady=5)

        self.check_files_button = tk.Button(root, text="Проверить файлы на кратность", command=self.start_check_files, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.check_files_button.grid(row=6, column=0, columnspan=4, pady=5)

        self.return_button = tk.Button(root, text="На главное окно", command=self.return_to_main_window, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.return_button.grid(row=7, column=0, columnspan=4, pady=10)

        self.root.grid_rowconfigure(5, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def select_input_file(self):
        filename = filedialog.askopenfilename(filetypes=[("GeoTIFF files", "*.tif *.tiff")])
        if filename:
            self.input_file_label.config(text=filename)

    def save_crop_size(self):
        width = self.crop_width.get()
        height = self.crop_height.get()

        if width % 256 != 0 or height % 256 != 0:
            messagebox.showwarning("Предупреждение", "Размеры кадрирования должны быть кратны 256.")
        else:
            messagebox.showinfo("Успех", f"Разрешение кадрирования установлено: {width}x{height}")

    def start_crop(self):
        input_file = self.input_file_label.cget("text")
        crop_width = self.crop_width.get()
        crop_height = self.crop_height.get()

        if not input_file:
            messagebox.showerror("Ошибка", "Выберите файл для кадрирования")
            return

        if os.path.getsize(input_file) == 0:
            messagebox.showerror("Ошибка", "Выбранный файл пуст")
            return

        if crop_width % 256 != 0 or crop_height % 256 != 0:
            messagebox.showwarning("Предупреждение", "Размеры кадрирования должны быть кратны 256.")
            return

        self.crop_button.config(state=tk.DISABLED)
        # Создание и запуск потока для кадрирования
        crop_thread = threading.Thread(target=self.crop_geotiff, args=(input_file, crop_width, crop_height, self.progress_bar, self.operation_status_label))
        crop_thread.start()

    def crop_geotiff(self, input_file, fragment_width, fragment_height, progress_bar, status_label):
        try:
            status_label.config(text="Начало кадрирования...")

            # Создание папки для сохранения в директории исходного файла
            input_dir = os.path.dirname(input_file)
            output_dir = os.path.join(input_dir, "cropped_files")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Открытие GeoTIFF файла
            with rasterio.open(input_file) as dataset:
                width = dataset.width
                height = dataset.height
                profile = dataset.profile

                # Прогресс-бар
                total_fragments = (width // fragment_width) * (height // fragment_height)
                progress_bar['maximum'] = total_fragments
                progress_bar['value'] = 0

                # Кадрирование и сохранение
                for i in range(0, width, fragment_width):
                    for j in range(0, height, fragment_height):
                        # Вычисление размера фрагмента для обработки граничных случаев
                        w = min(fragment_width, width - i)
                        h = min(fragment_height, height - j)
                        window = Window(i, j, w, h)
                        transform = dataset.window_transform(window)
                        profile.update({
                            'height': h,
                            'width': w,
                            'transform': transform
                        })

                        with rasterio.open(
                            os.path.join(output_dir, f"cropped_{i}_{j}.tif"),
                            'w',
                            **profile
                        ) as dst:
                            for k in range(1, dataset.count + 1):
                                dst.write(dataset.read(k, window=window), k)

                        progress_bar.step(1)
                        self.root.update_idletasks()

            progress_bar['value'] = progress_bar['maximum']
            status_label.config(text="Кадрирование успешно завершено!")
            messagebox.showinfo("Успех", "Кадрирование успешно завершено!")
        except Exception as e:
            status_label.config(text="Ошибка при кадрировании!")
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")
        finally:
            self.crop_button.config(state=tk.NORMAL)

    def start_check_files(self):
        self.check_files_button.config(state=tk.DISABLED)
        check_files_thread = threading.Thread(target=self.check_files)
        check_files_thread.start()

    def check_files(self):
        input_file = self.input_file_label.cget("text")
        input_dir = os.path.dirname(input_file)
        output_dir = os.path.join(input_dir, "cropped_files")
        if not os.path.exists(output_dir):
            messagebox.showerror("Ошибка", "Папка с кадрированными файлами не найдена.")
            self.check_files_button.config(state=tk.NORMAL)
            return

        files = [f for f in os.listdir(output_dir) if f.endswith(".tif")]
        total_files = len(files)
        self.progress_bar['maximum'] = total_files
        self.progress_bar['value'] = 0

        for file in files:
            file_path = os.path.join(output_dir, file)
            try:
                with rasterio.open(file_path) as dataset:
                    width = dataset.width
                    height = dataset.height

                    if width % 256 != 0 or height % 256 != 0:
                        os.remove(file_path)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

            self.progress_bar.step(1)
            self.root.update_idletasks()

        messagebox.showinfo("Успех", "Проверка файлов завершена. Все некратные файлы удалены.")
        self.check_files_button.config(state=tk.NORMAL)

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
app = GeoTIFFCropperApp(root)

# Запуск главного цикла обработки событий
root.mainloop()
