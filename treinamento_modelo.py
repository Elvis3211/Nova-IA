import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Definir e preparar os dados
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([0, 1, 1, 0])

# Construir o modelo de IA
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(4, input_shape=(2,), activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

modelo = build_model()

# Compilar o modelo
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
historico = modelo.fit(entradas, saidas, epochs=1000, verbose=0)

# Testar o modelo
testes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
previsoes = modelo.predict(testes)
previsoes_list = [(teste.tolist(), previsao[0]) for teste, previsao in zip(testes, previsoes)]

# Salvar o modelo treinado
modelo.save("modelo_IA_com_dropout.h5")

# Exibir as previsões
for entrada, previsao in previsoes_list:
    print(f"Entrada: {entrada}, Previsão: {previsao:.4f}")

# Plotar a precisão do treinamento
plt.plot(historico.history['accuracy'])
plt.title('Precisão do Modelo')
plt.xlabel('Época')
plt.ylabel('Precisão')
plt.show()

# Plotar a perda do treinamento
plt.plot(historico.history['loss'])
plt.title('Perda do Modelo')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.show()
