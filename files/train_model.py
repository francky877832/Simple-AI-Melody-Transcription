import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Charger le CSV
csv_path = 'dataset_mfcc_features.csv'
df = pd.read_csv(csv_path)

print("Aperçu des données :")
print(df.head())

print("\nInformations sur le DataFrame :")
print(df.info())

print("\nDistribution des classes :")
print(df['label'].value_counts())

# 2. Séparer les caractéristiques et les labels
X = df.drop('label', axis=1).values
y = df['label'].values

# 3. Encodage des labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nClasses encodées :")
print(list(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

# Sauvegarder le LabelEncoder
joblib.dump(label_encoder, 'datas/label_encoder.joblib')
print("\nLabelEncoder sauvegardé sous 'label_encoder.joblib'.")

# 4. Normalisation des caractéristiques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'datas/scaler.joblib')  # Sauvegarder le scaler
print("\nCaractéristiques normalisées et scaler sauvegardé sous 'scaler.joblib'.")

# 5. Division des données
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTaille de l'ensemble d'entraînement : {X_train.shape[0]}")
print(f"Taille de l'ensemble de test : {X_test.shape[0]}")

# 6. Définir le modèle
input_dim = X_train.shape[1]
num_classes = len(label_encoder.classes_)

model = Sequential([
    Dense(256, input_dim=input_dim, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 7. Définir les callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 8. Entraîner le modèle
history = model.fit(
    X_train,
    y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 9. Évaluer le modèle
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrécision sur l'ensemble de test : {test_accuracy*100:.2f}%")

# 10. Rapport de classification
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 11. Matrice de confusion
conf_mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.ylabel('Classe réelle')
plt.xlabel('Classe prédite')
plt.title('Matrice de confusion')
plt.show()

# 12. Courbes d'apprentissage
def plot_history(history):
    plt.figure(figsize=(14, 5))
    
    # Perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perte Entraînement')
    plt.plot(history.history['val_loss'], label='Perte Validation')
    plt.title('Courbe de perte')
    plt.xlabel('Épochs')
    plt.ylabel('Perte')
    plt.legend()
    
    # Précision
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Précision Entraînement')
    plt.plot(history.history['val_accuracy'], label='Précision Validation')
    plt.title('Courbe de précision')
    plt.xlabel('Épochs')
    plt.ylabel('Précision')
    plt.legend()
    
    plt.show()

plot_history(history)

# 13. Sauvegarder le modèle
model.save('datas/note_classification_model.keras')
print("\nModèle sauvegardé sous 'note_classification_model.keras'.")
