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
        self.embedding_threshold = 5
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
        return np.multiply(np.multiply(hadamard(8), matrix), hadamard(8))

    @staticmethod
    def get_coefficients(hadamard_matrix):
        """ Returns coefficients of changing elements """
        return hadamard_matrix[2][1], hadamard_matrix[2][5], hadamard_matrix[6][1], hadamard_matrix[6][5]

    @staticmethod
    def calculate_avgs(hadamard_matrix):
        """
        Function calculates avg value of pixels around target coefficie
        :param hadamard_matrix:
        :return:
        """
        def get_avg(i, j):
            return (hadamard_matrix[i-1][j-1] +
                    hadamard_matrix[i-1][j] +
                    hadamard_matrix[i-1][j+1] +
                    hadamard_matrix[i][j-1] +
                    hadamard_matrix[i][j+1] +
                    hadamard_matrix[i+1][j-1] +
                    hadamard_matrix[i+1][j] +
                    hadamard_matrix[i+1][j+1]) / 8

        return get_avg(2,1), get_avg(2,5), get_avg(6,1), get_avg(6,5)

    @staticmethod
    def get_avg(block, bit_num):
         return block['averages'][bit_num]

    @staticmethod
    def embedding_new_pixels(block):
        try:
            block['hadamard_embedded'][2][1] = block['new_coefficients'][0]
            block['hadamard_embedded'][2][5] = block['new_coefficients'][1]
            block['hadamard_embedded'][6][1] = block['new_coefficients'][2]
            block['hadamard_embedded'][6][5] = block['new_coefficients'][3]
        except IndexError:
            pass

    @staticmethod
    def calculate_new_pixels(block, pixel):
        if 'new_coefficients' in block:
            block['new_coefficients'].append(pixel)
        else:
            block['new_coefficients'] = [pixel]

    def embedding_image(self, block_array):
        block_count = 0
        assert len(block_array) == 1024

        for i in range(self.embedded_image_bin.shape[0]):
            for j in range(self.embedded_image_bin.shape[1]):
                bit_count = i * 64 + j + 1
                block_count = (bit_count - 1) // 4
                bit_num = bit_count % 4
                block = self.candidate_blocks[block_count]
                new_pix = self.get_avg(block, bit_num - 1)

                if self.embedded_image_bin[i][j]:
                    new_pix += self.embedding_threshold
                else:
                    new_pix -= self.embedding_threshold

                self.calculate_new_pixels(block, new_pix)

        for block in block_array:
            self.embedding_new_pixels(block)
            block['block_embedded'] = self.get_hadamard(block['hadamard_embedded'])

        return block_count

    def ddfa(self):
        """
        Distinct discrete firefly algorithm implementation.
        Returns two matrices with x and y coordinates of
        selected blocks
        """
        for candidate in self.candidate_blocks:
            hadamard_matrix = self.get_hadamard(candidate['block'])

            candidate['hadamard'] = hadamard_matrix
            candidate['hadamard_embedded'] = np.array(hadamard_matrix, copy=True)
            candidate['coefficients'] = self.get_coefficients(hadamard_matrix)
            candidate['averages'] = self.calculate_avgs(hadamard_matrix)

        self.embedding_image(self.candidate_blocks[:1024])
