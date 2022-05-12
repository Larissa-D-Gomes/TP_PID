import tensorflow as tf
from matplotlib.pyplot import table
import numpy as np
import pickle
import random
from sklearn import metrics

# Treina e testa uma Rede Neural a partir de um conjunto de descritores de todas imagens
# @param conjunto de descritores de todas imagens, número de descritores, opção gravar SVM em arquivo
# @return [Rede Neural, metricas do teste]
def treinar_rede_neural(descritores_todas_imagens, numero_descritores=3, gravar_rede=False):
    
    # Descritores separados por Birad
    birad1 = []
    birad2 = []
    birad3 = []
    birad4 = []

    # Incluir em cada birad uma com os descritores de cada imagem, transformados em uma arranjo 1D e a classificação
    #[arranjo descritores, classificação]
    for instance in range(len(descritores_todas_imagens[0])):
        birad1.append([np.reshape(descritores_todas_imagens[0][instance], numero_descritores*5), 1])
        birad2.append([np.reshape(descritores_todas_imagens[1][instance], numero_descritores*5), 2])
        birad3.append([np.reshape(descritores_todas_imagens[2][instance], numero_descritores*5), 3])
        birad4.append([np.reshape(descritores_todas_imagens[3][instance], numero_descritores*5), 4])

    # Embaralhar as amostras para randomizar os conjuntos de treino e teste
    random.shuffle(birad1)
    random.shuffle(birad2)
    random.shuffle(birad3)
    random.shuffle(birad4)

    # Conjuntos de treino e teste
    training_data = []
    test_data = []

    # Separação do conjunto de treino (75 instâncias de cada birad)
    for instance in range(75):
        training_data.append(birad1[instance])
        training_data.append(birad2[instance])
        training_data.append(birad3[instance])
        training_data.append(birad4[instance])

    training_data = np.array(training_data, dtype=object)

    # Separação do conjunto de teste (25 instâncias de cada birad)
    for instance in range(75, 100):
        test_data.append(birad1[instance])
        test_data.append(birad2[instance])
        test_data.append(birad3[instance])
        test_data.append(birad4[instance])

    test_data = np.array(test_data, dtype=object)


    # Embaralhar os conjuntos para não ficar em ordem por Birad
    random.shuffle(training_data)
    random.shuffle(test_data)


    # Separar descritores das classificações - Treino
    train_X = [] #descritores
    train_y = [] #classificações

    for descritor, classe in training_data:
        train_X.append(descritor)
        train_y.append(classe)

    train_X = np.array(train_X)
    train_y = np.array(train_y)


    # Separar descritores das classificações - Test
    test_X = []
    test_y = []

    for descritor, classe in test_data:
        test_X.append(descritor)
        test_y.append(classe)

    test_X = np.array(test_X)
    test_y = np.array(test_y)

    # Criar Rede Neural
    modelo_rede = tf.keras.models.Sequential()
    modelo_rede.add(tf.keras.layers.Dense(42, activation=tf.nn.relu)) #adiciona camada oculta
    modelo_rede.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax)) #adiciona camada de saída
    modelo_rede.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])#compilar rede
    
    # Treinar Rede Neural
    modelo_rede.fit(x=train_X, y=train_y, epochs=10)

    # Testar Rede Neural
    predictions = modelo_rede.predict(test_X).argmax(1) +1
    metricas = [metrics.confusion_matrix(test_y, predictions), metrics.accuracy_score(test_y, predictions)]

    # Gravar modelo
    if (gravar_rede):
        output_rede = open('rede.pkl', 'wb')
        pickle.dump(modelo_rede, output_rede)

        output_metricas = open('metricas_rede.pkl', 'wb')
        pickle.dump(metricas, output_metricas)


    return(modelo_rede, metricas)

# Testa uma imagem em uma Rede Neural
# @param modelo da rede neural, descritores da imagem, número de descritores
# @return classificação
def classificar_rede(modelo_rede, descritores, numero_descritores=3):
    instancia = np.reshape(descritores, numero_descritores*5)
    instancia = instancia.reshape(1, -1)
    return modelo_rede.predict(instancia).argmax(1) +1