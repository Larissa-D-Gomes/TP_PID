from re import L
import gerador_matrizes as gm
import descritores_haralick as dh
import ia_svm
import ia_rede
import pickle
import numpy as np
from sklearn import svm
from sklearn import metrics

# Calcular set de matrizes de todas imagens / diretorio default / número de tons default (argumento -> numero_tons)
# set_matrizes_todas_imagens = gm.calcula_matrizes_varias_imagens(gravar_arquivo="True")

# Carregar set_matrizes_todas_imagens  do arquivo dataset_matrizes.pkl
input = open('./dados/dataset_matrizes.pkl', 'rb')
set_matrizes_todas_imagens = np.array(pickle.load(input))
input.close()
print("Arquivo dataset_matrizes.pkl carregado.")

# Calcular conjunto de matrizes de uma imagem / número de tons default (argumento -> numero_tons)
set_matrizes_uma_imagem = gm.calcula_matrizes_uma_imagem(path="./imagens/1/p_d_left_cc(12).png")
print("Conjunto de matrizes de uma imagem calculado.")

# Calcular descritores para todas matrizes de todas imagens
descritores_todas_imagens = dh.calcula_descritores_varias_imagens(set_matrizes_todas_imagens)
print("Descritores de haralick de todas imagens calculado.")

# Calcular descritores para matrizes de uma única imagem
descritores_uma_imagem = dh.calcula_descritores_uma_imagem(set_matrizes_uma_imagem)
print("Descritores de haralick de uma imagem calculado.")

# Treinar uma SVM (argumento numero_descritores Default)
(modelo_svm, metricas) = ia_svm.treinar_svm(descritores_todas_imagens=descritores_todas_imagens, numero_descritores=4, gravar_svm=False)
print ("SVM treinada e testada.")

# Classificar uma imagem com a SVM(a partir de seus descritores)
classe = ia_svm.classificar_svm(modelo_svm, descritores_uma_imagem, numero_descritores=4)
print("Uma imagem classificada (SVM)")

"""
# Treinar uma Rede Neural (argumento numero_descritores Default)
(modelo_rede, metricas) = ia_rede.treinar_rede_neural(descritores_todas_imagens=descritores_todas_imagens, numero_descritores=4, gravar_rede=False)
print("Rede Neural treinada e testada.")

# Classificar uma imagem com a Rede Neural(a partir de seus descritores)
classe = ia_rede.classificar_rede(modelo_rede, descritores_uma_imagem, numero_descritores=4)
print("Uma imagem classificada (Rede Neural)")

"""
print(metricas)
print(classe)
