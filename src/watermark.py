from PIL import Image
import numpy as np


class Watermark:
    def __init__(self, image_path: str):
        self.image = self.validate_image(image_path)
        self.matrix = self.image_to_matrix(self.image)
        self.block_array = self.crop_matrix(self.matrix)
        self.entropies = self.get_entropies()
        self.embedded_image_size = 64 * 64
        self.candidate_blocks_num = self.embedded_image_size // 2
        self.candidate_blocks = self.entropies[:self.candidate_blocks_num]
        self.selected_blocks = self.ddfa()

    @staticmethod
    def validate_image(image_path):
        im = Image.open(image_path)
        assert im.size == (512, 512)
        assert im.mode == "L"

        return im

    @staticmethod
    def image_to_matrix(image: Image):
        """
        Функция принимает на вход изображение,
        и возвращает матрицу размерностью 512х512
        """
        pix = np.array(image)

        return pix

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
                'entropy': shannon_entropy + pal_entropy
            })

        return sorted(entropies, key=lambda d: d['entropy'])

    def ddfa(self):
        """
        Distinct discrete firefly algorithm implementation.
        Returns two matrices with x and y coordinates of
        selected blocks
        """
        return True
