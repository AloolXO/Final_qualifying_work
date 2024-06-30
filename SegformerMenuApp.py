#SegformerMenuApp.py
import json
import subprocess
import tkinter as tk
from tkinter import filedialog

# Code to hide the console window
import ctypes
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

class MainWindow:
    def __init__(self, master):
        self.master = master
        self.normal_geometry = master.geometry()  # Initialize with the current geometry
        self.load_window_geometry()  # Load saved geometry if exists
        master.title("Система выделения объектов горной промышленности")
        master.iconbitmap(r'hiik.ico')
        master.configure(bg='#E3F3F2')
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        
        self.image_frame = tk.Frame(master, bg='#E3F3F2')
        self.image_frame.pack(pady=10)
        
        self.load_image()

        self.label = tk.Label(master, text="Добро пожаловать в систему выделения объектов горной промышленности!", font=("Bahnschrift", 16), bg='#E3F3F2')
        self.label.pack(pady=20)

        self.btn_prepare_data = tk.Button(master, text="Подготовить данные", command=self.open_data_preparation, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_prepare_data.pack(pady=10)

        self.btn_train_model = tk.Button(master, text="Обучить модель", command=self.train_model, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_train_model.pack(pady=10)
        
        self.btn_train_model = tk.Button(master, text="Просмотр графиков метрик", command=self.comparison_of_training, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_train_model.pack(pady=10)

        self.btn_predict_model = tk.Button(master, text="Использовать модель для предсказания", command=self.predict_model, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_predict_model.pack(pady=10)
        
        self.btn_return = tk.Button(self.master, text="Вернуться", command=self.return_to_main, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_return.pack(pady=10)

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


    def open_data_preparation(self):
        self.save_window_geometry()
        self.master.destroy()
        subprocess.Popen(["python", "utils/DatasetPreparationApp.py"])

    def train_model(self):
        self.save_window_geometry()
        self.master.destroy()
        subprocess.Popen(["python", "TrainSegformerApp.py"])

    def comparison_of_training(self):
        self.save_window_geometry()
        self.master.destroy()
        subprocess.Popen(["python", "ComparisonOfTrainingApp.py"])

    def predict_model(self):
        self.save_window_geometry()
        self.master.destroy()
        subprocess.Popen(["python", "PredictSegformerApp.py"])
        
    def return_to_main(self):
        self.save_window_geometry()
        self.master.destroy()
        try:
            subprocess.Popen(["python", "Final_qualifying_work.py"])
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




def main():
    root = tk.Tk()
    app = MainWindow(root)
    root.minsize(500, 500)
    root.mainloop()

if __name__ == '__main__':
    main()
