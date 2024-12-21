# Définir l'image de base
FROM python:3.9.12

# Définir le mainteneur (optionnel)
LABEL maintainer="fotiosor@yahoo.fr"

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires dans l'image
COPY requirements.txt /app/
COPY . /app/

# Installer les dépendances système et Python
RUN apt-get update && apt-get install -y libgomp1 \
    && pip install --no-cache-dir -r requirements.txt

# Exposer le port pour Streamlit
EXPOSE 8501

# Commande par défaut pour exécuter l'application
CMD ["streamlit", "run", "streamlit_fotio.py", "--server.port", "8501"]
