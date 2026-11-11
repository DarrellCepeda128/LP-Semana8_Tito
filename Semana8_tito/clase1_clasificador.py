import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Paso 3: Cargar el dataset
df = pd.read_csv('dataset_tito_v1.csv', encoding='utf-8')
print("--- Primeras filas del dataset ---")
print(df.head())
print("\n--- Conteo de categorías ---")
print(df['categoria'].value_counts())

# Paso 4: Separar X (texto) e y (categorías)
X = df['texto']
y = df['categoria']

# Paso 5: Crear conjuntos de entrenamiento (75%) y prueba (25%)
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y
)

# Paso 6: Construir el pipeline TF-IDF + Clasificador MultinomialNB
modelo = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clasificador', MultinomialNB())
])

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Paso 7: Probar con preguntas nuevas
print("\n--- Pruebas individuales ---")
preguntas_prueba = [
    'que necesito para una beca',
    'a que hora abre la biblioteca',
    'como me inscribo al siguiente semestre'
]

for pregunta in preguntas_prueba:
    categoria_predicha = modelo.predict([pregunta])[0]
    print(f"'{pregunta}' -> {categoria_predicha}")