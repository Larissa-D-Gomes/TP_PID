# Biblioteca: skimage
# https://scikit-image.org/docs/0.19.x/

from skimage.feature import greycomatrix, greycoprops
from skimage import io
from math import log
import numpy

image = io.imread('p_d_left_cc(12).png')

# Função para calcular entropia de uma matriz de co-ocorrência de tons de cinza
# Parâmetros: GCHistogram[li][lj][ld][la], li, lj, ld, la
# Retorna matriz de entropia calculada [Ângulo m, Distância n]
def entropy(GCHistogram, li, lj, ld, la):
    entropyMatrix = [[0 for x in range(ld)] for y in range(la)]
    for i in range(li):
        for j in range(lj):
            for d in range(ld):
                for a in range(la):
                    if(GCHistogram[i][j][d][a] > 0.0):
                        entropyMatrix[a][d] -= GCHistogram[i][j][d][a] * log(GCHistogram[i][j][d][a], 2)
    return entropyMatrix

def main():
    # github.com/scikit-image/scikit-image/blob/00177e14097237ef20ed3141ed454bc81b308f82/skimage/feature/texture.py#L15
    # Parâmetros: imagem, distâncias [], ângulos []
    # Retorna Histograma de co-ocorrência de tons de cinza

    # Distâncias: 1, 2, 4, 8, 16
    # Ângulos (Graus): 0, 45, 90, 135
    GCHistogram = greycomatrix(image, [1, 2, 4, 8, 16], [0, numpy.pi/4, numpy.pi/2, numpy.pi, 3*numpy.pi/4], None, False, True)

    # github.com/scikit-image/scikit-image/blob/00177e14097237ef20ed3141ed454bc81b308f82/skimage/feature/texture.py#L159
    # Parâmetros: Matriz de co-ocorrência, Propriedade calculada
    # Retorna matriz de propriedade calculada [Ângulo m, Distância n]

    # Cálculo de homogeneidade
    Homogeneity = greycoprops(GCHistogram, 'homogeneity')

    # Cálculo de energia
    Energy = greycoprops(GCHistogram, 'energy')

    # Cálculo de entropia
    Entropy = entropy(GCHistogram, len(GCHistogram), len(GCHistogram[0]), len(GCHistogram[0][0]), len(GCHistogram[0][0][0]))


if __name__ == '__main__':
    main()