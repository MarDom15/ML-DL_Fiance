import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Charger le modèle enregistré avec le chemin complet
model_path = r'C:\Users\marti\Desktop\Nouveau dossier\ML-DL_Fiance\BTC_USD_yahoofinace\best_model.pkl'
model = joblib.load(model_path)

# Initialisation de la mise en page de l'application
st.title("Prédiction du prix du Bitcoin")
st.write("Entrez les valeurs des caractéristiques pour prédire le prochain prix le plus élevé du Bitcoin.")

# Création des champs pour l'utilisateur afin d'entrer les valeurs des caractéristiques
adj_close = st.number_input('Prix ajusté', min_value=0.0, value=50000.0, step=0.1)
close = st.number_input('Prix de clôture', min_value=0.0, value=50000.0, step=0.1)
low = st.number_input('Prix le plus bas', min_value=0.0, value=48000.0, step=0.1)
open_price = st.number_input('Prix d\'ouverture', min_value=0.0, value=49000.0, step=0.1)
volume = st.number_input('Volume', min_value=0, value=100000000, step=100000)

# Créer un bouton pour effectuer la prédiction
if st.button('Prédire le prochain prix élevé'):
    # Créer une entrée de données avec les valeurs saisies
    input_data = np.array([adj_close, close, low, open_price, volume]).reshape(1, -1)
    
    # Normaliser les données avant de faire la prédiction
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Prédiction avec le modèle
    prediction = model.predict(input_data_scaled)
    
    # Afficher la prédiction
    st.write(f"Le prix prédit pour le prochain jour est : ${prediction[0]:,.2f}")
