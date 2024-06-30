#TrainSegformerApp.py
import os
import time
import json
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
import psutil
import ctypes
import threading
import tkinter as tk
from tkinter import ttk
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from utils.DatasetPreparationApp import process_dataset, split_data, MiningDataset, transform
from utils.GpuUtilsApp import limit_gpu_memory, get_device
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss

# Code to hide the console window
import ctypes
ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

def train(model, dataloader, optimizer, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    for batch in dataloader:
        inputs, masks = batch
        inputs, masks = inputs.to(device), masks.to(device)

        if len(masks.shape) == 4:
            masks = masks.squeeze(1)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(pixel_values=inputs, labels=masks)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"Loss: {loss.item()}")

class TrainSegformerGUI:
    def __init__(self, master):
        self.master = master
        self.normal_geometry = master.geometry()  # Initialize with the current geometry
        self.load_window_geometry()  # Load saved geometry if exists
        master.title("Обучение архитектуры Segformer")
        master.iconbitmap(r'hiik.ico')
        master.configure(bg='#E3F3F2')

        self.dataset_path = tk.StringVar(value='./utils/dataset')
        self.output_path = tk.StringVar(value='./utils/processed_dataset')

        self.create_widgets()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        self.label = tk.Label(self.master, text="Обучение архитектуры Segformer", font=("Bahnschrift", 16), bg='#E3F3F2')
        self.label.pack(pady=20)

        self.btn_start_training = tk.Button(self.master, text="Начать обучение", command=self.start_training, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_start_training.pack(pady=10)

        self.log_text = tk.Text(self.master, height=10, width=80, bg='#E3F3F2', fg='black', font=('Bahnschrift', 12))
        self.log_text.pack(pady=10)

        self.progress = ttk.Progressbar(self.master, orient='horizontal', length=400, mode='determinate')
        self.progress.pack(pady=20)

        self.btn_return = tk.Button(self.master, text="Вернуться", command=self.return_to_main, bg='#003e87', fg='white', relief='raised', bd=3, padx=15, pady=7, font=('Bahnschrift', 12), borderwidth=5, highlightthickness=0)
        self.btn_return.pack(pady=10)

    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def start_training(self):
        self.log_message("Запуск обучения...")
        self.btn_start_training.config(state=tk.DISABLED)
        self.btn_return.config(state=tk.DISABLED)
        threading.Thread(target=self.train).start()

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

    def set_resource_limits(self):
        p = psutil.Process()
        cpu_count = len(p.cpu_affinity())
        p.cpu_affinity(p.cpu_affinity()[:cpu_count // 2])

        total_memory = psutil.virtual_memory().total
        max_memory = total_memory // 3

        if psutil.WINDOWS:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(ctypes.c_void_p(-1), ctypes.c_size_t(-1), ctypes.c_size_t(max_memory))
        else:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (max_memory, hard))

    def train(self):
        self.set_resource_limits()

        dataset_path = self.dataset_path.get()
        output_path = self.output_path.get()
        process_dataset(dataset_path, output_path)

        images_path = os.path.join(output_path, 'images')
        masks_path = os.path.join(dataset_path, 'masks')
        geotiffs_path = os.path.join(output_path, 'geotiffs')
        shapes_path = os.path.join(output_path, 'shapes')

        image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(('.png', '.tiff'))]
        mask_files = [os.path.join(masks_path, f) for f in os.listdir(masks_path) if f.endswith('.bmp')]
        tiff_files = [os.path.join(geotiffs_path, f) for f in os.listdir(geotiffs_path) if f.endswith('.tiff')]
        shape_files = [os.path.join(shapes_path, f) for f in os.listdir(shapes_path) if f.endswith('.shp')]

        if len(image_files) != len(mask_files):
            self.log_message("Ошибка: Количество изображений и масок не совпадает.")
            self.log_message(f"Количество изображений: {len(image_files)}")
            self.log_message(f"Количество масок: {len(mask_files)}")

            image_base_names = {os.path.splitext(os.path.basename(f))[0] for f in image_files}
            mask_base_names = {os.path.splitext(os.path.basename(f))[0] for f in mask_files}

            missing_masks = image_base_names - mask_base_names
            missing_images = mask_base_names - image_base_names

            if missing_masks:
                self.log_message(f"Маски отсутствуют для следующих изображений: {missing_masks}")
            if missing_images:
                self.log_message(f"Изображения отсутствуют для следующих масок: {missing_images}")

            self.btn_start_training.config(state=tk.NORMAL)
            return

        kfold_splits = split_data(image_files, mask_files, n_splits=10)
        feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", do_reduce_labels=True)

        device = get_device()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if device == 'cuda':
            limit_gpu_memory()
            metrics_filename = 'training_metrics_gpu.json'
            avg_metrics_filename = 'average_metrics_gpu.json'
            self.log_message("Используется GPU для обучения с CuDNN.")
        else:
            metrics_filename = 'training_metrics_cpu.json'
            avg_metrics_filename = 'average_metrics_cpu.json'
            self.log_message("Используется CPU для обучения.")

        metrics = []
        times = []

        total_epochs = 15
        total_steps = total_epochs * len(kfold_splits)

        for fold, (train_idx, val_idx) in enumerate(kfold_splits):
            self.log_message(f"Fold {fold + 1}")

            train_images = [image_files[i] for i in train_idx]
            train_masks = [mask_files[i] for i in train_idx]
            val_images = [image_files[i] for i in val_idx]
            val_masks = [mask_files[i] for i in val_idx]

            model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            model.to(device)

            train_dataset = MiningDataset(train_images, train_masks, feature_extractor, transform)
            val_dataset = MiningDataset(val_images, val_masks, feature_extractor, transform)

            train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=6, pin_memory=True)
            val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=6, pin_memory=True)

            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

            fold_metrics = []
            fold_times = []

            for epoch in range(total_epochs):
                start_time = time.time()
                train_loss, accuracy, f1, precision, recall, logloss = self.train_epoch(model, train_dataloader, optimizer, device)
                
                if train_loss is None:
                    self.log_message("Произошла ошибка во время обучения. Пропуск эпохи.")
                    continue

                val_loss, val_accuracy, val_f1, val_precision, val_recall, val_logloss = self.validate_epoch(model, val_dataloader, device)

                scheduler.step()
                end_time = time.time()

                epoch_time = end_time - start_time
                self.log_message(f"Epoch {epoch + 1} completed. Time taken: {epoch_time:.2f} seconds")
                self.log_message(f"Train Loss: {train_loss}, Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}, Log Loss: {logloss}")
                self.log_message(f"Validation Loss: {val_loss}, Val Accuracy: {val_accuracy}, Val F1: {val_f1}, Val Precision: {val_precision}, Val Recall: {val_recall}, Val Log Loss: {val_logloss}")

                fold_metrics.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'log_loss': logloss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'val_f1_score': val_f1,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_log_loss': val_logloss,
                    'time': epoch_time
                })

                fold_times.append({
                    'epoch': epoch + 1,
                    'time': epoch_time
                })

                current_step = (fold * total_epochs) + epoch + 1
                self.progress['value'] = (current_step / total_steps) * 100

            metrics.append({'fold': fold + 1, 'metrics': fold_metrics})
            times.append({'fold': fold + 1, 'times': fold_times})

            model.save_pretrained(f'./model_fold_{fold + 1}')
            feature_extractor.save_pretrained(f'./model_fold_{fold + 1}')

        with open(metrics_filename, 'w') as f:
            json.dump({'metrics': metrics, 'times': times}, f, indent=4)

        self.save_average_metrics(metrics, avg_metrics_filename)
        self.plot_metrics(metrics)
        self.log_message("Обучение завершено.")
        self.btn_start_training.config(state=tk.NORMAL)
        self.btn_return.config(state=tk.NORMAL)

    def save_average_metrics(self, metrics, filename):
        avg_metrics = []
        for fold in metrics:
            avg_fold_metrics = {
                'fold': fold['fold'],
                'train_loss': np.mean([epoch['train_loss'] for epoch in fold['metrics']]),
                'accuracy': np.mean([epoch['accuracy'] for epoch in fold['metrics']]),
                'f1_score': np.mean([epoch['f1_score'] for epoch in fold['metrics']]),
                'precision': np.mean([epoch['precision'] for epoch in fold['metrics']]),
                'recall': np.mean([epoch['recall'] for epoch in fold['metrics']]),
                'log_loss': np.mean([epoch['log_loss'] for epoch in fold['metrics']]),
                'val_loss': np.mean([epoch['val_loss'] for epoch in fold['metrics']]),
                'val_accuracy': np.mean([epoch['val_accuracy'] for epoch in fold['metrics']]),
                'val_f1_score': np.mean([epoch['val_f1_score'] for epoch in fold['metrics']]),
                'val_precision': np.mean([epoch['val_precision'] for epoch in fold['metrics']]),
                'val_recall': np.mean([epoch['val_recall'] for epoch in fold['metrics']]),
                'val_log_loss': np.mean([epoch['val_log_loss'] for epoch in fold['metrics']]),
            }
            avg_metrics.append(avg_fold_metrics)
        
        with open(filename, 'w') as f:
            json.dump(avg_metrics, f, indent=4)

    def train_epoch(self, model, dataloader, optimizer, device):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []

        scaler = torch.cuda.amp.GradScaler()
        for inputs, masks in dataloader:
            inputs, masks = inputs.to(device), masks.to(device)

            if len(masks.shape) == 4:
                masks = masks.squeeze(1)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=inputs, labels=masks)
                loss = outputs.loss

            if loss is None:
                self.log_message("Error: Model output loss is None. Check input data and model configuration.")
                return None, None, None, None, None, None

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            logits = outputs.logits
            preds = logits.argmax(dim=1)
            preds = F.interpolate(preds.unsqueeze(1).float(), size=masks.shape[-2:], mode='nearest').squeeze(1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(masks.cpu().numpy())

        avg_loss = train_loss / len(dataloader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
        f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='weighted')
        precision = precision_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
        recall = recall_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
        logloss = log_loss(all_labels.flatten(), all_preds.flatten()) if len(np.unique(all_labels.flatten())) > 1 else np.nan

        return avg_loss, accuracy, f1, precision, recall, logloss

    def validate_epoch(self, model, dataloader, device):
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, masks in dataloader:
                inputs, masks = inputs.to(device), masks.to(device)

                if len(masks.shape) == 4:
                    masks = masks.squeeze(1)

                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values=inputs, labels=masks)
                    loss = outputs.loss

                if loss is None:
                    self.log_message("Error: Model output loss is None. Check input data and model configuration.")
                    continue

                val_loss += loss.item()

                logits = outputs.logits
                preds = logits.argmax(dim=1)
                preds = F.interpolate(preds.unsqueeze(1).float(), size=masks.shape[-2:], mode='nearest').squeeze(1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(masks.cpu().numpy())

        avg_loss = val_loss / len(dataloader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
        f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
        precision = precision_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
        recall = recall_score(all_labels.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
        logloss = log_loss(all_labels.flatten(), all_preds.flatten()) if len(np.unique(all_labels.flatten())) > 1 else np.nan

        return avg_loss, accuracy, f1, precision, recall, logloss

    def plot_metrics(self, metrics):
        for fold_metrics in metrics:
            fold = fold_metrics['fold']
            epochs = [m['epoch'] for m in fold_metrics['metrics']]
            train_losses = [m['train_loss'] for m in fold_metrics['metrics']]
            accuracies = [m['accuracy'] for m in fold_metrics['metrics']]
            f1_scores = [m['f1_score'] for m in fold_metrics['metrics']]
            precisions = [m['precision'] for m in fold_metrics['metrics']]
            recalls = [m['recall'] for m in fold_metrics['metrics']]
            log_losses = [m['log_loss'] for m in fold_metrics['metrics']]
            val_losses = [m['val_loss'] for m in fold_metrics['metrics']]
            val_accuracies = [m['val_accuracy'] for m in fold_metrics['metrics']]
            val_f1_scores = [m['val_f1_score'] for m in fold_metrics['metrics']]
            val_precisions = [m['val_precision'] for m in fold_metrics['metrics']]
            val_recalls = [m['val_recall'] for m in fold_metrics['metrics']]
            val_log_losses = [m['val_log_loss'] for m in fold_metrics['metrics']]

            plt.figure(figsize=(12, 8))
            plt.plot(epochs, train_losses, label='Train Loss')
            plt.plot(epochs, accuracies, label='Accuracy')
            plt.plot(epochs, f1_scores, label='F1 Score')
            plt.plot(epochs, precisions, label='Precision')
            plt.plot(epochs, recalls, label='Recall')
            plt.plot(epochs, log_losses, label='Log Loss')
            plt.plot(epochs, val_losses, label='Validation Loss')
            plt.plot(epochs, val_accuracies, label='Validation Accuracy')
            plt.plot(epochs, val_f1_scores, label='Validation F1 Score')
            plt.plot(epochs, val_precisions, label='Validation Precision')
            plt.plot(epochs, val_recalls, label='Validation Recall')
            plt.plot(epochs, val_log_losses, label='Validation Log Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            plt.title(f'Fold {fold} Metrics')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'metrics_fold_{fold}.png')
            plt.show()

def main():
    root = tk.Tk()
    app = TrainSegformerGUI(root)
    root.minsize(800, 600)
    root.mainloop()

if __name__ == "__main__":
    main()
