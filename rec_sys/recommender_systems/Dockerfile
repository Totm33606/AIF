# Utiliser une image de base pytorch
FROM pytorch/pytorch

# Créer un répertoire de travail
WORKDIR /usr/src/app

# Copier tous les fichiers dans le workdir
COPY requirements.txt /usr/src/app/

# Installation des dépendances manquantes
# La commande 'rm' permet de limiter le poids de
# l'image Docker
RUN apt-get update && apt-get install -y build-essential unzip wget && rm -rf /var/lib/apt/lists/* 

# Installer les dépendances manquantes
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copier le fichier kaggle.json dans le conteneur
COPY kaggle.json /root/.kaggle/kaggle.json

# Modifier les permissions de kaggle.json pour qu'il soit sécurisé
RUN chmod 600 /root/.kaggle/kaggle.json

# Exposer le port pour le notebook
EXPOSE 8888