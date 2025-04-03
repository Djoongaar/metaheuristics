import numpy as np
from PIL import Image
from scipy.linalg import hadamard
from .utilities import Utilities
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr
)


class Watermark:
    def __init__(self, image_path: str, embedded_image: str):
        self.image = Utilities.get_image(image_path)
        self.image_matrix = Utilities.image_to_matrix(self.image)
        self.embedded_matrix = np.zeros(shape=(512, 512), dtype=int)
        self.block_array = Utilities.crop_matrix(self.image_matrix)
        self.entropies = self.get_entropies()
        self.embedded_image = Utilities.get_image(embedded_image)
        self.embedded_image_bin = Utilities.image_to_bin(self.embedded_image)
        self.embedded_image_size = 64 * 64
        self.embedding_threshold = 2
        self.candidate_blocks_count = self.embedded_image_size // 2
        self.candidate_blocks = self.entropies[:self.candidate_blocks_count]
        self.secret_key = np.zeros(shape=(64, 64), dtype=float)
        self.watermark_matrix = self.embedding(self.candidate_blocks[:1024])
        self.watermark = Utilities.matrix_to_image(self.watermark_matrix)
        self.ssim = Utilities.get_ssim(self.image_matrix, self.watermark_matrix)
        self.psnr = Utilities.get_psnr(self.image_matrix, self.watermark_matrix)
        self.extracted = self.extracting()
        self.nc = Utilities.get_normal_correlation(self.embedded_image_bin, self.extracted)

    def build_matrix(self):
        """ Воссоздает матрицу размером 512 х 512 из массива блоков (4096, 8, 8) """
        matrix = np.array(self.image_matrix, copy=True)

        for block in self.candidate_blocks:
            if 'block_embedded' in block:
                for i in range(8):
                    for j in range(8):
                        i_offset = block['index'] // 64
                        j_offset = block['index'] % 64
                        matrix[8 * i_offset + i][8 * j_offset + j] = block['block_embedded'][i][j]

        return matrix

    def get_entropies(self):
        """ Рассчитывает комплексную энтропию """
        entropies = []
        for i in range(len(self.block_array)):
            array = self.block_array[i].flatten()
            array_norm = array / np.sum(array)

            shannon_entropy = Utilities.get_shannon_entropy(array_norm)
            pal_entropy = Utilities.get_pal_entropy(array_norm)

            entropies.append({
                'index': i,
                'entropy': shannon_entropy + pal_entropy,
                'origin': self.block_array[i]
            })

        return sorted(entropies, key=lambda d: d['entropy'])

    def save_secret_key(self, block_index, i, j):
        self.secret_key[i][j] = block_index

    def encoding(self):

        for i in range(64):
            for j in range(64):
                bit_count = i * 64 + j
                block_count = bit_count  // 4
                bit_num = bit_count % 4
                block = self.candidate_blocks[block_count]
                new_pix = Utilities.get_avg(block['hadamard'], bit_num)

                if self.embedded_image_bin[i][j]:
                    new_pix += self.embedding_threshold
                else:
                    new_pix -= self.embedding_threshold

                Utilities.insert_new_pixel(block, new_pix, bit_num)
                self.save_secret_key(block['index'], i, j)

    def embedding(self, candidate_blocks):
        """
        Погружает ЦВЗ (бинарное изображение размерностью 64 х 64)
        в покрывающее изображение размером 512 х 512
        """

        # Шаг 1. Сначала для всех блоков-кандидатов считаем функцию Адамара
        for candidate_block in candidate_blocks:
            hadamard_matrix = Utilities.get_hadamard(candidate_block['origin'])

            candidate_block['hadamard'] = hadamard_matrix
            candidate_block['hadamard_embedded'] = np.array(hadamard_matrix, copy=True)

        # Шаг 2. Затем для каждого бита встраиваемого изображения:
        # считаем новое значение коэффициента и встраиваем его в матрицу Адамара
        self.encoding()

        # Шаг 3. Выполняем обратное преобразование Адамара
        for candidate in candidate_blocks:
            candidate['block_embedded'] = Utilities.get_hadamard(candidate['hadamard_embedded'], round=True)

        # Шаг 4. По массиву размерностью (4096, 8, 8) восстанавливаем матрицу размерностью 512 х 512
        self.embedded_matrix = self.build_matrix()

        # Шаг 5. Из матрицы размерностью 512 х 512 восстанавливаем и возвращаем изображение PNG
        return self.embedded_matrix

    def extracting(self):
        """
        Получает ЦВЗ из покрывающего объекта
        """
        result = np.zeros(shape=(64, 64), dtype=float)

        # Шаг 1. Разбивает матрицу (со встроенным ЦВЗ) размером 512 х 512 на массив размерностью (4096, 8, 8)
        block_array = Utilities.crop_matrix(self.embedded_matrix)

        # Шаг 2. Использую секретный ключ получаем адреса блоков в которых хранятся ЦВЗ
        for i in range(64):
            for j in range(64):
                bit_count = i * 64 + j
                block_ind = int(self.secret_key[i][j])
                bit_num = bit_count % 4

                # Шаг 3. Применить преобразование Адамара к полученным блокам
                if bit_num == 0:
                    hadamard = Utilities.get_hadamard(block_array[block_ind])
                    result[i][j] = 1 if hadamard[2][1] >= Utilities.get_avg(hadamard, 0) else 0
                    result[i][j+1] = 1 if hadamard[2][5] >= Utilities.get_avg(hadamard, 1) else 0
                    result[i][j+2] = 1 if hadamard[6][1] >= Utilities.get_avg(hadamard, 2) else 0
                    result[i][j+3] = 1 if hadamard[6][5] >= Utilities.get_avg(hadamard, 3) else 0

        return result