import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Cargar datos y entrenar el modelo base
df = pd.read_csv('dataset_tito_v1.csv', encoding='utf-8')
X = df['texto']
y = df['categoria']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

modelo = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clasificador', MultinomialNB())
])
modelo.fit(X_train, y_train)

# Paso 1 (Clase 2): Generar predicciones sobre el conjunto de prueba
y_pred = modelo.predict(X_test)

# Paso 2: Calcular y mostrar Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(accuracy, 3)}\n")

# Paso 3: Reporte de Clasificación (Precision, Recall, F1-Score)
print("--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred, zero_division=0))

# Paso 4: Matriz de Confusión
print("--- Matriz de Confusión ---")
categorias = sorted(df['categoria'].unique())
matriz = confusion_matrix(y_test, y_pred, labels=categorias)
print("Categorías:", categorias)
print(matriz)

# Paso 5: Revisar errores uno por uno
print("\n--- Analisis de Errores Reales ---")
resultados = pd.DataFrame({
    'pregunta': X_test,
    'real': y_test,
    'predicha': y_pred
})
errores = resultados[resultados['real'] != resultados['predicha']]
print(errores.to_string(index=False))

# Paso 6: Guardar el modelo en disco
joblib.dump(modelo, 'modelo_tito_semana8.joblib')
print("\nModelo guardado exitosamente como 'modelo_tito_semana8.joblib'")

# Reto A: Modo Interactivo
print("\n--- Modo Interactivo (Escribe 'salir' para terminar) ---")
while True:
    pregunta = input('Pregunta para Tito: ')
    if pregunta.lower() == 'salir':
        break
    categoria = modelo.predict([pregunta])[0]
    print(f'Categoría predicha: {categoria}\n')