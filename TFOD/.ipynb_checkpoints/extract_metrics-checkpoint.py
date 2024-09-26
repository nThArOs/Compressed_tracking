import re
import csv

# Lire le fichier de sortie
with open("output.txt", "r") as f:
    output = f.read()

# Regex pour extraire les métriques importantes
regex = r"(DetectionBoxes_Precision/mAP(?:@\S*)?|DetectionBoxes_Precision/mAP \((?:small|medium|large)\)|DetectionBoxes_Recall/AR@\S*|Loss/\S+):\s+(\d+\.\d+)|Performing evaluation on (\d+) images|Eval metrics at step (\d+)"

# Extraire les métriques, les valeurs correspondantes, le nombre d'images et l'étape
matches = re.finditer(regex, output)

# Utiliser un dictionnaire pour stocker les métriques uniques et leurs valeurs
unique_metrics = {}
num_images = None
step = None

for match in matches:
    metric = match.group(1)
    value = match.group(2)
    if metric is not None and value is not None:
        unique_metrics[metric] = value
    else:
        if match.group(3) is not None:
            num_images = int(match.group(3))
        if match.group(4) is not None:
            step = int(match.group(4))

# Créer un fichier CSV et enregistrer les métriques, leurs valeurs, le nombre d'images et l'étape
with open("/home/jovyan/Desktop/TFOD/models/retina50/eval/metrics.csv", "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Number of Images", num_images])
    csv_writer.writerow(["Step", step])
    csv_writer.writerow([])
    csv_writer.writerow(["Metric", "Value"])
    for metric, value in unique_metrics.items():
        csv_writer.writerow([metric, value])
