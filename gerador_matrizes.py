import numpy as np
import os
import cv2
import pickle
import matriz_circular as mc


# Calcula todas as matrizes de co-ocorrência de todas as imagens de um diretório com 4 pastas de birads
# @param path do diretório, opção de salvar em arquivo, número de tons de cinza para reamostragem
# @return lista de listas de birads cada um com a lista das matrizes de cada imagem no diretório
def calcula_matrizes_varias_imagens(diretorio="./imagens", gravar_arquivo=False, numero_tons=32):
    # Declarações
    BIRADS = ["1", "2", "3", "4"]
    dataset = [[], [], [], []] #uma lista de matrizes para cada imagem em cada birad

    count = 1 #contador para mostrar terminal

    # Passar pela pasta de cada Birad
    for birad in BIRADS:
        path = os.path.join(diretorio,birad)

        # Passar por cada imagem na pasta da Birad
        for image_name in os.listdir(path):

            # Testar tipo do arquivo
            if(not(image_name.endswith(".png") or image_name.endswith(".jpg" ))):
                continue

            #Abrir Imagem
            image = cv2.imread(os.path.join(path,image_name), 0)

            # Selecionar tom mais claro da imagem
            max = np.max(image)
            # Reamostrar imagem para numero_tons tons de cinza
            for i in range(len(image)):
                for j in range(len(image[0])):
                    image[i][j] = int(image[i][j]/max * numero_tons-1)

            # Calcular descritores a partir das matrizes de coocorrência circulares C1, C2, C4, C8 e C16
            c1 = mc.c1(np.array(image), numero_tons)
            c2 = mc.c2(np.array(image), numero_tons)
            c4 = mc.c4(np.array(image), numero_tons)
            c8 = mc.c8(np.array(image), numero_tons)
            c16 = mc.c16(np.array(image), numero_tons)

            # Inserir matrizes no dataset
            dataset[int(birad)-1].append([c1, c2, c4, c8, c16])
            print(count)
            count+=1

    # Gravar datatset
    if (gravar_arquivo):
        output = open('dados/dataset_matrizes.pkl', 'wb')
        pickle.dump(dataset, output)

    return dataset

# Calcula todas as matrizes de co-ocorrência de uma imagem
# @param path da imagem, número de tons para reamostragme
# @return lista das matrizes da imagem 
def calcula_matrizes_uma_imagem(path, numero_tons=32):
    
    # Testar tipo do arquivo
    if(not(path.endswith(".png") or path.endswith(".jpg" ))):
        return None

    # Abrir Imagem
    image = cv2.imread(path, 0)

    # Reamostrar imagem para 32 tons de cinza
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j] = int(image[i][j]/255 * numero_tons-1)  

    # Calcular descritores a partir das matrizes de coocorrência circulares C1, C2, C4, C8 e C16
    c1 = mc.c1(np.array(image), numero_tons)
    c2 = mc.c2(np.array(image), numero_tons)
    c4 = mc.c4(np.array(image), numero_tons)
    c8 = mc.c8(np.array(image), numero_tons)
    c16 = mc.c16(np.array(image), numero_tons)

    return[c1, c2, c4, c8, c16]

    