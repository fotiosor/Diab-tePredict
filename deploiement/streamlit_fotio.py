# -*- coding: utf-8 -*-

# Librairies
import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
import numpy as np

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Déploiement du Modèle DiabètePredict 📌",
    page_icon="🌟",
    layout="wide",
)

# Titre principal avec style
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>Déploiement du Modèle de Classification 📊</h1>
    <p style='text-align: center; color: #6c757d;'>Entrez les caractéristiques pour prédire la classe</p>
    <hr style="border-top: 2px solid #4CAF50;">
    """,
    unsafe_allow_html=True,
)

# Chargement du modèle avec mise en cache
@st.cache_resource
def chargement_modele():
    return load_model("modele_fotio")

# Chargement du modèle
modele = chargement_modele()

# Organisation en colonnes
col1, col2 = st.columns([2, 3])

# Texte dans la première colonne (entrée des caractéristiques)
with col1:
    st.subheader("Entrez les caractéristiques")
    diastolic = st.text_input("Diastolic Blood Pressure (mmHg) :", "72", max_chars=3)
    bodymass = st.text_input("Body Mass Index (BMI) :", "33", max_chars=3)
    age = st.text_input("Age (years) :", "50", max_chars=3)
    plasma = st.text_input("Plasma Glucose Concentration (mg/dL) :", "120", max_chars=3)

# Fonction pour convertir les saisies utilisateur
def try_parse(value):
    try:
        return float(value)
    except ValueError:
        return np.nan

# Conversion des saisies en dictionnaire
la_Data = {
    'diastolic': try_parse(diastolic),
    'bodymass': try_parse(bodymass),
    'age': try_parse(age),
    'plasma': try_parse(plasma),
}

# Bouton pour prédire
if st.button("🧪 Prédire la Classe"):
    try:
        # Prédiction via le modèle
        prediction = predict_model(modele, data=pd.DataFrame([la_Data]))
        classe = prediction['prediction_label'][0]
        score = prediction['prediction_score'][0]

        # Résultats affichés dans la deuxième colonne
        with col2:
            st.subheader("Résultats de la Prédiction")
            st.markdown(
                f"""
                <div style='padding: 15px; border: 2px solid #4CAF50; border-radius: 10px;'>
                    <h3 style='color: #4CAF50;'>Classe Prédite : <b>{classe}</b></h3>
                    <p style='font-size: 18px; color: #555;'>Score d'Appartenance : <b>{score:.2f}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception as e:
        # Gestion des erreurs
        st.error("Erreur dans la prédiction. Vérifiez vos saisies.")
else:
    with col2:
        st.info("Appuyez sur le bouton **Prédire la Classe** après avoir saisi les valeurs.")

# Footer esthétique
st.markdown(
    """
    <hr style="border-top: 2px solid #eee;">
    <p style='text-align: center; color: #888;'>Créé avec ❤️ et Streamlit par FOTIO Fabrice</p>
    """,
    unsafe_allow_html=True,
)
