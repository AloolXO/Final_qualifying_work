#ComparisonOfTrainingApp.py
import os
import json
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Code to hide the console window
import ctypes
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

class ComparisonOfTraining:
    def __init__(self, master):
        self.master = master
        self.normal_geometry = master.geometry()  # Initialize with the current geometry
        self.load_window_geometry()  # Load saved geometry if exists
        master.title("Сравнение обучения Segformer")
        master.iconbitmap(r'hiik.ico')
        master.configure(bg='#E3F3F2')

        self.create_widgets()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)


    def create_widgets(self):
        self.label = tk.Label(self.master, text="Сравнение обучения Segformer", font=("Bahnschrift", 16), bg='#E3F3F2')
        self.label.pack(pady=20)

        self.metrics_file_label = tk.Label(self.master, text="Выберите файл с метриками обучения:", bg='#E3F3F2', font=('Bahnschrift', 12))
        self.metrics_file_label.pack(pady=5)

        self.metrics_file_combobox = ttk.Combobox(self.master, state="readonly", font=('Bahnschrift', 12))
        self.metrics_file_combobox.pack(pady=5)

        self.average_file_label = tk.Label(self.master, text="Выберите файл с усреднёнными метриками:", bg='#E3F3F2', font=('Bahnschrift', 12))
        self.average_file_label.pack(pady=5)

        self.average_file_combobox = ttk.Combobox(self.master, state="readonly", font=('Bahnschrift', 12))
        self.average_file_combobox.pack(pady=5)

        self.scan_files()

        self.plot_button = tk.Button(self.master, text="Построить графики", command=self.plot_metrics, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12))
        self.plot_button.pack(pady=10)

        self.return_button = tk.Button(self.master, text="Вернуться", command=self.return_to_main, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12))
        self.return_button.pack(pady=10)

        # Create a notebook for tabs
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

    def create_tab(self, title):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=title)
        return tab

    def scan_files(self):
        training_files = []
        average_files = []
        for file in os.listdir('.'):
            if file.endswith('training_metrics_gpu.json') or file.endswith('training_metrics_cpu.json'):
                training_files.append(file)
            elif file.endswith('average_metrics_gpu.json') or file.endswith('average_metrics_cpu.json'):
                average_files.append(file)
        if training_files:
            self.metrics_file_combobox['values'] = training_files
            self.metrics_file_combobox.current(0)
        else:
            messagebox.showwarning("Внимание", "Файлы training_metrics_gpu.json или training_metrics_cpu.json не найдены.")

        if average_files:
            self.average_file_combobox['values'] = average_files
            self.average_file_combobox.current(0)
        else:
            messagebox.showwarning("Внимание", "Файлы average_metrics_gpu.json или average_metrics_cpu.json не найдены.")

    def plot_metrics(self):
        self.plot_button.config(state=tk.DISABLED)
        self.return_button.config(state=tk.DISABLED)
        selected_training_file = self.metrics_file_combobox.get()
        selected_average_file = self.average_file_combobox.get()
        if not selected_training_file:
            messagebox.showwarning("Внимание", "Сначала выберите файл с метриками обучения.")
            return
        if not selected_average_file:
            messagebox.showwarning("Внимание", "Сначала выберите файл с усреднёнными метриками.")
            return

        with open(selected_training_file, 'r') as f:
            metrics = json.load(f)

        with open(selected_average_file, 'r') as f:
            summary_metrics = json.load(f)

        # Clear existing tabs
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)

        title = "Обучение с помощью GPU" if 'gpu' in selected_training_file else "Обучение с помощью CPU"
        self.plot_summary_metrics(summary_metrics, title)

        for fold in metrics['metrics']:
            self.plot_fold_metrics(fold['metrics'], fold['fold'], metrics['times'])

        # Select the first tab (summary metrics)
        self.notebook.select(0)

    def plot_summary_metrics(self, summary_metrics, title):
        tab = self.create_tab("Средние значения метрик")

        folds = [m['fold'] for m in summary_metrics]
        avg_train_losses = [m['train_loss'] for m in summary_metrics]
        avg_accuracies = [m['accuracy'] for m in summary_metrics]
        avg_f1_scores = [m['f1_score'] for m in summary_metrics]
        avg_precisions = [m['precision'] for m in summary_metrics]
        avg_recalls = [m['recall'] for m in summary_metrics]
        avg_log_losses = [m['log_loss'] for m in summary_metrics]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(folds, avg_train_losses, label='Train Loss', marker='o')
        ax.plot(folds, avg_accuracies, label='Accuracy', marker='o')
        ax.plot(folds, avg_f1_scores, label='F1 Score', marker='o')
        ax.plot(folds, avg_precisions, label='Precision', marker='o')
        ax.plot(folds, avg_recalls, label='Recall', marker='o')
        ax.plot(folds, avg_log_losses, label='Log Loss', marker='o')

        for i in range(len(folds)):
            ax.text(folds[i], avg_train_losses[i], f'{avg_train_losses[i]:.2f}')
            ax.text(folds[i], avg_accuracies[i], f'{avg_accuracies[i]:.2f}')
            ax.text(folds[i], avg_f1_scores[i], f'{avg_f1_scores[i]:.2f}')
            ax.text(folds[i], avg_precisions[i], f'{avg_precisions[i]:.2f}')
            ax.text(folds[i], avg_recalls[i], f'{avg_recalls[i]:.2f}')
            ax.text(folds[i], avg_log_losses[i], f'{avg_log_losses[i]:.2f}')

        ax.set_xlabel('Fold')
        ax.set_ylabel('Metric')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def plot_fold_metrics(self, fold_metrics, fold, times):
        tab = self.create_tab(f"Fold {fold}")

        epochs = [m['epoch'] for m in fold_metrics]
        train_losses = [m['train_loss'] for m in fold_metrics]
        accuracies = [m['accuracy'] for m in fold_metrics]
        f1_scores = [m['f1_score'] for m in fold_metrics]
        precisions = [m['precision'] for m in fold_metrics]
        recalls = [m['recall'] for m in fold_metrics]
        log_losses = [m['log_loss'] for m in fold_metrics]

        # Проверка наличия данных по времени для текущего фолда
        times_fold = []
        for t in times:
            if t['fold'] == fold:
                times_fold = [time_entry['time'] for time_entry in t['times']]
                break

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
        ax1.plot(epochs, accuracies, label='Accuracy', marker='o')
        ax1.plot(epochs, f1_scores, label='F1 Score', marker='o')
        ax1.plot(epochs, precisions, label='Precision', marker='o')
        ax1.plot(epochs, recalls, label='Recall', marker='o')
        ax1.plot(epochs, log_losses, label='Log Loss', marker='o')

        for i in range(len(epochs)):
            ax1.text(epochs[i], train_losses[i], f'{train_losses[i]:.2f}')
            ax1.text(epochs[i], accuracies[i], f'{accuracies[i]:.2f}')
            ax1.text(epochs[i], f1_scores[i], f'{f1_scores[i]:.2f}')
            ax1.text(epochs[i], precisions[i], f'{precisions[i]:.2f}')
            ax1.text(epochs[i], recalls[i], f'{recalls[i]:.2f}')
            ax1.text(epochs[i], log_losses[i], f'{log_losses[i]:.2f}')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Metric')
        ax1.set_title(f'Fold {fold} Metrics')
        ax1.legend()
        ax1.grid(True)

        if times_fold:
            ax2.plot(epochs, times_fold, label='Time', color='orange', marker='o')
            for i in range(len(epochs)):
                ax2.text(epochs[i], times_fold[i], f'{times_fold[i]:.2f}')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Time (s)')
            ax2.set_title(f'Fold {fold} Training Time')
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, 'No time data available', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.plot_button.config(state=tk.NORMAL)
        self.return_button.config(state=tk.NORMAL)

    def return_to_main(self):
        self.save_window_geometry()
        self.master.destroy()
        try:
            subprocess.Popen(["python", "SegformerMenuApp.py"])
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при возврате к главному файлу: {e}")
            
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
    app = ComparisonOfTraining(root)
    root.minsize(800, 600)
    root.mainloop()

if __name__ == "__main__":
    main()
