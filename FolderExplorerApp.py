#FolderExplorerApp.py
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import re
import glob
import subprocess

# Code to hide the console window
import ctypes
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

class FolderExplorerApp:
    def __init__(self, root):
        self.root = root
        self.normal_geometry = root.geometry()  # Initialize with the current geometry
        self.load_window_geometry()  # Load saved geometry if exists
        root.title("Выбор и работа с папкой")
        root.iconbitmap(r'hiik.ico')
        root.configure(bg='#E3F3F2')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


        self.selected_folder = None
        self.status_label = None

        # Приветственный текст
        self.label = tk.Label(root, text="Выберите папку для работы", bg='#E3F3F2', font=('Bahnschrift', 14))
        self.label.pack(pady=20)

        # Кнопка выбора папки
        self.button = tk.Button(root, text="Выбрать папку", command=self.select_folder, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.button.pack(pady=10)

        # Кнопка выполнения операций
        self.process_button = tk.Button(root, text="Выполнить операции", command=self.process_folder, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.process_button.pack(pady=10)

        # Кнопка возвращения на главное окно
        self.return_button = tk.Button(root, text="На главное окно", command=self.return_to_main_window, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.return_button.pack(pady=10)

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            if self.selected_folder == folder_path:
                messagebox.showwarning("Предупреждение", "Вы выбрали ту же самую директорию. Пожалуйста, выберите другую директорию.")
            else:
                self.selected_folder = folder_path
                self.label.config(text="Выбранная папка: " + self.selected_folder)
        else:
            self.label.config(text="Папка не выбрана")

    def process_folder(self):
        if not self.selected_folder:
            self.set_status("Папка не выбрана")
            return

        # Блокируем кнопки
        self.toggle_buttons(state=tk.DISABLED)
        
        self.create_tiff_folder()
        self.process_r20m_files()
        self.process_r10m_files()
        self.set_status("Операции выполнены успешно.")

        # Разблокируем кнопки
        self.toggle_buttons(state=tk.NORMAL)

    def toggle_buttons(self, state):
        self.button.config(state=state)
        self.process_button.config(state=state)
        self.return_button.config(state=state)

    def create_tiff_folder(self):
        # Создание папки TIFF в выбранной папке
        tiff_folder = os.path.join(self.selected_folder, "TIFF")
        if not os.path.exists(tiff_folder):
            try:
                os.mkdir(tiff_folder)
            except OSError:
                self.set_status("Не удалось создать папку TIFF.")

    def process_r20m_files(self):
        # Получаем папку IMG_DATA R20m
        img_data_folders = glob.glob(os.path.join(self.selected_folder, "GRANULE", "L2A_*", "IMG_DATA"))
        for img_data_folder in img_data_folders:
            r20m_folder = os.path.join(img_data_folder, "R20m")
            if not os.path.exists(r20m_folder):
                self.set_status(f"Папка R20m не найдена в {img_data_folder}.")
                continue

            # Ищем файлы *B12_20m.jp2; *B11_20m.jp2
            matching_files = glob.glob(os.path.join(r20m_folder, "*B12_20m.jp2")) + \
                             glob.glob(os.path.join(r20m_folder, "*B11_20m.jp2"))

            # Копировать файлы в папку GRANULE\L2A_*\IMG_DATA и убрать "_20m" из их названия
            for file_path in matching_files:
                file_name = os.path.basename(file_path)
                new_file_name = file_name.replace("_20m", "")
                new_file_path = os.path.join(img_data_folder, new_file_name)
                try:
                    shutil.copy(file_path, new_file_path)
                except FileNotFoundError:
                    self.set_status(f"Файл {file_path} уже был перемещен или переименован.")

    def process_r10m_files(self):
        # Получаем папку IMG_DATA R10m
        img_data_folders = glob.glob(os.path.join(self.selected_folder, "GRANULE", "L2A_*", "IMG_DATA"))
        for img_data_folder in img_data_folders:
            r10m_folder = os.path.join(img_data_folder, "R10m")
            if not os.path.exists(r10m_folder):
                self.set_status(f"Папка R10m не найдена в {img_data_folder}.")
                continue

            # Ищем файлы *B02_10m.jp2; *B03_10m.jp2; *B04_10m.jp2; *B08_10m.jp2
            matching_files = glob.glob(os.path.join(r10m_folder, "*B02_10m.jp2")) + \
                             glob.glob(os.path.join(r10m_folder, "*B03_10m.jp2")) + \
                             glob.glob(os.path.join(r10m_folder, "*B04_10m.jp2")) + \
                             glob.glob(os.path.join(r10m_folder, "*B08_10m.jp2"))

            # Копировать файлы в папку GRANULE\L2A_*\IMG_DATA и убрать "_10m" из их названия
            for file_path in matching_files:
                file_name = os.path.basename(file_path)
                new_file_name = file_name.replace("_10m", "")
                new_file_path = os.path.join(img_data_folder, new_file_name)
                try:
                    shutil.copy(file_path, new_file_path)
                except FileNotFoundError:
                    self.set_status(f"Файл {file_path} уже был перемещен или переименован.")

    def set_status(self, status_text):
        if self.status_label is None:
            self.status_label = tk.Label(self.root, text=status_text, padx=10, pady=10)
            self.status_label.pack()
        else:
            self.status_label.config(text=status_text)

    def return_to_main_window(self):
        self.save_window_geometry()
        # Закрываем текущее окно
        self.root.destroy()
        
        # Открываем новое окно (главное окно)
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
app = FolderExplorerApp(root)

# Запуск главного цикла обработки событий
root.mainloop()
