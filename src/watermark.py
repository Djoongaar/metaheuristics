import numpy as np
from PIL import Image
from scipy.linalg import hadamard


class Watermark:
    def __init__(self, image_path: str, embedded_image: str):
        self.image = self.get_cover_image(image_path)
        self.image_matrix = self.image_to_matrix(self.image)
        self.embedded_matrix = np.zeros(shape=(512, 512), dtype=int)
        self.block_array = self.crop_matrix(self.image_matrix)
        self.entropies = self.get_entropies()
        self.embedded_image = self.get_embedded_image(embedded_image)
        self.embedded_image_bin = self.image_to_bin(self.embedded_image)
        self.embedded_image_size = 64 * 64
        self.embedding_threshold = 2
        self.candidate_blocks_count = self.embedded_image_size // 2
        self.candidate_blocks = self.entropies[:self.candidate_blocks_count]
        self.secret_key = np.zeros(shape=(64, 64), dtype=float)
        self.watermark = self.embedding(self.candidate_blocks[:1024])
        self.result = self.extracting()

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
    def matrix_to_image(matrix):

        return Image.fromarray(matrix, mode='L')

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
        cropped = np.zeros(shape=(64 * 64, 8, 8), dtype=float)
        for i in range(64):
            for j in range(64):
                cropped[i * 64 + j] = matrix[8 * i:8 * (i + 1), 8 * j:8 * (j + 1)]
        return cropped

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

    @staticmethod
    def get_shannon_entropy(array):
        """ Рассчитывает энтропию Шеннона """
        items = []
        for i in array:
            items.append(i * np.log(i))

        return -np.sum(items)

    @staticmethod
    def get_pal_entropy(array):
        """ Рассчитывает энтропию Пал """
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
                'origin': self.block_array[i]
            })

        return sorted(entropies, key=lambda d: d['entropy'])

    @staticmethod
    def get_hadamard(matrix, round=False):
        """
        Выполняет преобразование Адамара.
        Возвращает матрицу размерностью 8 x 8
        """
        assert matrix.shape == (8,8)

        h = hadamard(8) / 1
        if round:
            return np.round(np.matmul(h, matrix, h) / 3)

        return np.matmul(h, matrix, h) / 3

    @staticmethod
    def get_avg(matrix, bit_num):
        assert matrix.shape == (8, 8)

        if bit_num == 0:
            i, j = 2, 1
        elif bit_num == 1:
            i, j = 2, 5
        elif bit_num == 2:
            i, j = 6, 1
        else:
            i, j = 6, 5

        return (matrix[i-1][j-1] +
            matrix[i-1][j] +
            matrix[i-1][j+1] +
            matrix[i][j-1] +
            matrix[i][j+1] +
            matrix[i+1][j-1] +
            matrix[i+1][j] +
            matrix[i+1][j+1]) / 8

    @staticmethod
    def insert_new_pixel(block, new_pix, bit_num):
        if bit_num == 0:
            block['hadamard_embedded'][2][1] = new_pix
        elif bit_num == 1:
            block['hadamard_embedded'][2][5] = new_pix
        elif bit_num == 2:
            block['hadamard_embedded'][6][1] = new_pix
        else:
            block['hadamard_embedded'][6][5] = new_pix

    def save_secret_key(self, block_index, i, j):
        self.secret_key[i][j] = block_index

    def encoding(self):

        for i in range(64):
            for j in range(64):
                bit_count = i * 64 + j
                block_count = bit_count  // 4
                bit_num = bit_count % 4
                block = self.candidate_blocks[block_count]
                new_pix = self.get_avg(block['hadamard'], bit_num)

                if self.embedded_image_bin[i][j]:
                    new_pix += self.embedding_threshold
                else:
                    new_pix -= self.embedding_threshold

                self.insert_new_pixel(block, new_pix, bit_num)
                self.save_secret_key(block['index'], i, j)

    def embedding(self, candidate_blocks):
        """
        Погружает ЦВЗ (бинарное изображение размерностью 64 х 64)
        в покрывающее изображение размером 512 х 512
        """

        # Шаг 1. Сначала для всех блоков-кандидатов считаем функцию Адамара
        for candidate_block in candidate_blocks:
            hadamard_matrix = self.get_hadamard(candidate_block['origin'])

            candidate_block['hadamard'] = hadamard_matrix
            candidate_block['hadamard_embedded'] = np.array(hadamard_matrix, copy=True)

        # Шаг 2. Затем для каждого бита встраиваемого изображения:
        # считаем новое значение коэффициента и встраиваем его в матрицу Адамара
        self.encoding()

        # Шаг 3. Выполняем обратное преобразование Адамара
        for candidate in candidate_blocks:
            candidate['block_embedded'] = self.get_hadamard(candidate['hadamard_embedded'], round=True)

        # Шаг 4. По массиву размерностью (4096, 8, 8) восстанавливаем матрицу размерностью 512 х 512
        self.embedded_matrix = self.build_matrix()

        # Шаг 5. Из матрицы размерностью 512 х 512 восстанавливаем и возвращаем изображение PNG
        return self.matrix_to_image(self.embedded_matrix)

    def extracting(self):
        """
        Получает ЦВЗ из покрывающего объекта
        """
        result = np.zeros(shape=(64, 64), dtype=float)
        # Шаг 1. Получаем из ЦВЗ матрицу размерностью 512 х 512
        matrix = self.image_to_matrix(self.watermark)

        # Шаг 2. Разбивает матрицу (со встроенным ЦВЗ) размером 512 х 512 на массив размерностью (4096, 8, 8)
        block_array = self.crop_matrix(matrix)

        # Шаг 3. Использую секретный ключ получаем адреса блоков в которых хранятся ЦВЗ
        for i in range(64):
            for j in range(64):
                bit_count = i * 64 + j
                block_ind = int(self.secret_key[i][j])
                bit_num = bit_count % 4

                # Шаг 4. Применить преобразование Адамара к полученным блокам
                if bit_num == 0:
                    hadamard = self.get_hadamard(block_array[block_ind])
                    result[i][j] = 0 if hadamard[2][1] <= self.get_avg(hadamard, 0) else 1
                    result[i][j] = 0 if hadamard[2][5] <= self.get_avg(hadamard, 1) else 1
                    result[i][j] = 0 if hadamard[6][1] <= self.get_avg(hadamard, 2) else 1
                    result[i][j] = 0 if hadamard[6][5] <= self.get_avg(hadamard, 3) else 1

        return self.matrix_to_image(result * 255)
