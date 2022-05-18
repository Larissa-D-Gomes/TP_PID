import time
from matplotlib.pyplot import table
import numpy as np
import pickle
import random
from sklearn import svm
from sklearn import metrics
from tkinter.messagebox import showinfo

# Treina e testa uma SVM a partir de um conjunto de descritores de todas imagens
# @param conjunto de descritores de todas imagens, número de descritores, opção gravar SVM em arquivo
# @return [SVM, metricas do teste]
def treinar_svm(descritores_todas_imagens, numero_descritores=4, gravar_svm=False):
    showinfo(message="Treino da SVM iniciado!")
    tempoInicial = time.time()
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


    # Criar SVM
    modelo_svm = svm.SVC(gamma=0.1, C=100)

    # Treinar SVM
    modelo_svm.fit(train_X, train_y)

    # Testar SVM
    predictions = modelo_svm.predict(test_X)
    metricas = [metrics.confusion_matrix(test_y, predictions), metrics.accuracy_score(test_y, predictions)]

    # Cálculo Manual da especificidade (vn /(vn+fp))
    matriz = metricas[0]
    especificidade = 0.0
    for classe in range(3):
        vn = 0.0
        fp = 0.0
        for i in range(len(matriz)):
            for j in range(len(matriz[0])):
                if (i != classe and j != classe):
                    vn += matriz[i][j]
                if (i != classe and j == classe):
                    fp += matriz[i][j]
        especificidade += vn /(vn+fp)
    especificidade /= 4
    metricas.append(especificidade)

    # Gravar modelo
    if (gravar_svm):
        output_svm = open('dados/svm.pkl', 'wb')
        pickle.dump(modelo_svm, output_svm)

        output_metricas = open('dados/metricas_svm.pkl', 'wb')
        pickle.dump(metricas, output_metricas)

    tempoFinal = (time.time() - tempoInicial)
    metricas.append(tempoFinal)

    showinfo(message="Treino da SVM finalizado!")

    return(modelo_svm, metricas)


# Testa uma imagem em uma SVM
# @param modelo da svm, descritores da imagem, número de descritores
# @return classificação
def classificar_svm(modelo_svm, descritores, numero_descritores=4):
    instancia = np.reshape(descritores, numero_descritores*5)
    instancia = instancia.reshape(1, -1)
    return modelo_svm.predict(instancia)