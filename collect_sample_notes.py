import os
import shutil

# Chemin du dossier racine à parcourir
root_dir = 'dataset/archive/Guitar Dataset'  # Remplace par le chemin réel
# Dossier de destination pour les fichiers déplacés
destination_dir = 'samples_notes'  # Remplace par le chemin réel

# Créer le dossier de destination s'il n'existe pas
os.makedirs(destination_dir, exist_ok=True)

# Parcourir les sous-dossiers
for subdir, dirs, files in os.walk(root_dir):
    if files:  # Vérifier s'il y a des fichiers dans le sous-dossier
        first_file = files[0]  # Obtenir le premier fichier
        src_file_path = os.path.join(subdir, first_file)  # Chemin complet du fichier source
        
        # Créer un dossier pour le sous-dossier dans le dossier de destination
        subdir_name = os.path.basename(subdir)  # Nom du sous-dossier
        new_subdir_path = os.path.join(destination_dir, subdir_name)
        os.makedirs(new_subdir_path, exist_ok=True)  # Créer le sous-dossier s'il n'existe pas

        # Déplacer le fichier vers le nouveau dossier
        shutil.move(src_file_path, new_subdir_path)
        print(f'Déplacé : {src_file_path} vers {new_subdir_path}')

print("Tous les fichiers ont été déplacés avec succès.")
