import numpy as np
from PIL import Image
from scipy.linalg import hadamard


class Watermark:
    def __init__(self, image_path: str, embedded_image: str):
        self.image = self.get_cover_image(image_path)
        self.image_matrix = self.image_to_matrix(self.image)
        self.block_array = self.crop_matrix(self.image_matrix)
        self.entropies = self.get_entropies()
        self.embedded_image = self.get_embedded_image(embedded_image)
        self.embedded_image_bin = self.image_to_bin(self.embedded_image)
        self.embedded_image_size = 64 * 64
        self.candidate_blocks_count = self.embedded_image_size // 2
        self.candidate_blocks = self.entropies[:self.candidate_blocks_count]
        self.ddfa()

    @staticmethod
    def get_cover_image(image_path):
        im = Image.open(image_path)
        assert im.size == (512, 512)
        assert im.mode == "L"

        return im

    @staticmethod
    def get_embedded_image(embedded_image_path):
        im = Image.open(embedded_image_path)
        assert im.size == (64, 64)
        assert im.mode == "L"

        return im

    @staticmethod
    def image_to_matrix(image: Image):
        """
        Функция принимает на вход изображение,
        и возвращает матрицу
        """
        pix = np.array(image)

        return pix

    @staticmethod
    def image_to_bin(image: Image):
        """
        Функция принимает на вход изображение,
        и возвращает двоичную матрицу
        """
        pix = Watermark.image_to_matrix(image)
        threshold = 128

        return pix < threshold

    @staticmethod
    def crop_matrix(matrix):
        """
        Дробит матрицу размером 512х512 на блоки 8х8
        и возвращает в массив размерностью (4096, 8, 8)
        """
        cropped = np.zeros(shape=(64 * 64, 8, 8), dtype=int)
        for i in range(64):
            for j in range(64):
                cropped[i * 64 + j] = matrix[8 * i:8 * (i + 1), 8 * j:8 * (j + 1)]
        return cropped

    @staticmethod
    def reconstruct_matrix(cropped):
        """
        Воссоздает и возвращает матрицу из массива блоков
        """
        matrix = np.zeros(shape=(512, 512))

        for i in range(64):
            for j in range(64):
                matrix[8 * i:8 * (i + 1), 8 * j:8 * (j + 1)] = cropped[i * 64 + j]

        return matrix

    @staticmethod
    def get_shannon_entropy(array):
        """ Calculate Shannon's entropy of array """
        items = []
        for i in array:
            items.append(i * np.log(i))

        return -np.sum(items)

    @staticmethod
    def get_pal_entropy(array):
        """ Calculate Pal's & Pal's entropy """
        items = []
        for i in array:
            items.append(i * np.exp(1 - i))

        return np.sum(items)

    def get_entropies(self):
        """ Calculate complex entropy """
        entropies = []
        for i in range(len(self.block_array)):
            array = self.block_array[i].flatten()
            array_norm = array / np.sum(array)

            shannon_entropy = self.get_shannon_entropy(array_norm)
            pal_entropy = self.get_pal_entropy(array_norm)

            entropies.append({
                'index': i,
                'entropy': shannon_entropy + pal_entropy,
                'block': self.block_array[i]
            })

        return sorted(entropies, key=lambda d: d['entropy'])

    @staticmethod
    def get_hadamard(matrix):
        """
        Returns Hadamard coefficients of matrix with shape 8 x 8
        """
        assert matrix.shape == (8,8)
        return np.multiply(matrix, hadamard(8))

    @staticmethod
    def get_coefficients(hadamard_matrix):
        """ Returns coefficients of changing elements """
        return hadamard_matrix[2][1], hadamard_matrix[2][5], hadamard_matrix[6][1], hadamard_matrix[6][5]

    def ddfa(self):
        """
        Distinct discrete firefly algorithm implementation.
        Returns two matrices with x and y coordinates of
        selected blocks
        """
        for candidate in self.candidate_blocks:
            hadamard_matrix = self.get_hadamard(candidate['block'])
            candidate['hadamard'] = hadamard_matrix
            candidate['coefficients'] = self.get_coefficients(hadamard_matrix)