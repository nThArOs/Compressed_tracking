import os
import shutil

main_dir = os.path.join(os.getcwd(),"datasets/MOT17-reid")
dest_dir = os.path.join(os.getcwd(),"datasets/MOT17-reid-merged")

print(main_dir)
print(dest_dir)
os.makedirs(dest_dir, exist_ok=True)

# Parcourir tous les répertoires dans le répertoire principal
for dir_name in os.listdir(main_dir):
    dir_path = os.path.join(main_dir, dir_name)

    # Vérifier si c'est un répertoire
    if os.path.isdir(dir_path):
        # Extraire le numéro de séquence du nom du répertoire
        sequence_number = dir_name.split('-')[1]

        # Parcourir tous les sous-répertoires dans le répertoire courant
        for subdir_name in os.listdir(dir_path):
            subdir_path = os.path.join(dir_path, subdir_name)

            # Vérifier si c'est un répertoire
            if os.path.isdir(subdir_path):
                # Créer un nouveau nom de sous-répertoire préfixé par le numéro de séquence
                new_subdir_name = f"{sequence_number}_{subdir_name}"
                new_subdir_path = os.path.join(dest_dir, new_subdir_name)

                # Créer le nouveau sous-répertoire dans le répertoire de destination
                os.makedirs(new_subdir_path, exist_ok=True)

                # Parcourir tous les fichiers dans le sous-répertoire courant
                for file_name in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file_name)

                    # Ignore if it's a directory
                    if os.path.isdir(file_path):
                        continue

                    # Créer le chemin vers le nouveau fichier
                    new_file_path = os.path.join(new_subdir_path, file_name)

                    # Copier le fichier dans le nouveau sous-répertoire
                    shutil.copy(file_path, new_file_path)
