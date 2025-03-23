import os
import re
import csv
import numpy as np
from PIL import Image

# Pfade zu den Ordnern (bitte anpassen)
y_gen_dir = r"/Users/deniztekin/Documents/Uni/Masterarbeit/inference/inferenz/Batch16/1000/inference/y_gen"
label_dir = r"/Users/deniztekin/Documents/Uni/Masterarbeit/inference/inferenz/Batch16/1000/inference/label"
output_csv = "/Users/deniztekin/Documents/Uni/Masterarbeit/inference/inferenz/Batch16/1000/mse_mae.csv"

# Regex, um Epoch und Batch aus dem Dateinamen zu extrahieren
# Beispiel: y_gen_5_epoch_batch_123.tiff
y_gen_pattern = re.compile(r"y_gen_(\d+)_batch_(\d+)\.tiff$")

# Label-Dateinamen (nur Batch enthalten)
# Beispiel: label_1_batch_123.tiff
label_pattern = re.compile(r"label_0_batch_(\d+)\.tiff$")

# Dictionary, das pro Batch alle (epoch, pfad)-Infos sammelt
# Struktur: y_gen_dict[<batch_number>] = [ { "epoch": <epoch>, "path": <full_path>, ... }, ... ]
y_gen_dict = {}

# 1) Alle y_gen-Dateien auslesen und nach Batch gruppieren
for filename in os.listdir(y_gen_dir):
    match = y_gen_pattern.match(filename)
    if match:
        epoch_number = match.group(1)  # z.B. 5
        batch_number = match.group(2)  # z.B. 123
        full_path = os.path.join(y_gen_dir, filename)
        
        if batch_number not in y_gen_dict:
            y_gen_dict[batch_number] = []
        
        y_gen_dict[batch_number].append({
            "epoch": epoch_number,
            "path": full_path,
            "filename": filename
        })

# 2) Für jedes Label die passenden y_gen-Dateien (alle Epochen) finden und vergleichen
results = []
for label_filename in os.listdir(label_dir):
    label_match = label_pattern.match(label_filename)
    if label_match:
        batch_number = label_match.group(1)
        label_path = os.path.join(label_dir, label_filename)

        # Falls wir y_gen-Dateien für diesen Batch haben
        if batch_number in y_gen_dict:
            # Label laden
            label_img = np.array(Image.open(label_path), dtype=np.float32)
            
            # Für jede Epoche in diesem Batch berechnen wir den Fehler
            for gen_info in y_gen_dict[batch_number]:
                epoch_number = gen_info["epoch"]
                gen_path = gen_info["path"]
                gen_filename = gen_info["filename"]

                # Generiertes Bild laden
                y_gen_img = np.array(Image.open(gen_path), dtype=np.float32)

                # Fehler berechnen
                mse = np.mean((y_gen_img - label_img) ** 2)
                mae = np.mean(np.abs(y_gen_img - label_img))

                # Ergebnisse sammeln
                results.append({
                    "epoch_number": epoch_number,
                    "batch_number": batch_number,
                    "y_gen_filename": gen_filename,
                    "label_filename": label_filename,
                    "MSE": mse,
                    "MAE": mae
                })

# 3) Ergebnisse in einer CSV speichern
with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, 
        fieldnames=["epoch_number", "batch_number", "y_gen_filename", "label_filename", "MSE", "MAE"]
    )
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Fertig! Ergebnisse wurden in {output_csv} gespeichert.")
