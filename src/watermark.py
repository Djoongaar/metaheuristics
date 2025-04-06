import numpy as np
from PIL import Image
from scipy.linalg import hadamard
from .utilities import Utilities, Attack
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
        self.embedded_image = Utilities.get_image(embedded_image)
        self.embedded_image_bin = Utilities.image_to_bin(self.embedded_image)
        self.embedding_threshold = 2
        self.candidate_blocks = Utilities.get_block_candidates(self.block_array)
        self.secret_key = np.zeros(shape=(64, 64), dtype=float)
        self.watermark_matrix = self.embedding(self.candidate_blocks[:1024])
        self.watermark = Utilities.matrix_to_image(self.watermark_matrix)
        self.ssim = Utilities.get_ssim(self.image_matrix, self.watermark_matrix)
        self.psnr = Utilities.get_psnr(self.image_matrix, self.watermark_matrix)
        self.extracted_image_bin = Utilities.extracting(self.watermark, self.secret_key)
        self.nc = self.attack_and_extract()

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

    def attack_and_extract(self):
        return Utilities.matrix_to_image( Attack(self.watermark).spr70)
