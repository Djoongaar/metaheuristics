import numpy as np
from PIL import Image
from scipy.linalg import hadamard
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr
)


class Utilities:

    def __init__(self):
        pass

    @staticmethod
    def get_ssim(im1, im2):
        return ssim(im1, im2)

    @staticmethod
    def get_psnr(im1, im2):
        return psnr(im1, im2)

    @staticmethod
    def get_normal_correlation(im1, im2):
        return np.sum(np.multiply(im1, im2)) / np.sum(np.multiply(im1, im1))

    @staticmethod
    def compute_quality(im1: np.ndarray, im2: np.ndarray):
        assert im1.shape == im2.shape == (512, 512)

        gamma = 10
        ssim = Utilities.get_ssim(im1, im2)
        psnr = Utilities.get_psnr(im1, im2)
        robust = Utilities.get_normal_correlation(im1, im2)

        return gamma / psnr + 1 / ssim + 1 / robust

    @staticmethod
    def get_image(image_path):
        im = Image.open(image_path)
        assert im.size[0] == im.size[1]
        assert im.size[0] % 8 == 0
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
        result = np.zeros(shape=(64, 64), dtype=int)
        pix = Utilities.image_to_matrix(image)

        for i in range(64):
            for j in range(64):
                if pix[i][j] < 128:
                    result[i][j] = 0
                else:
                    result[i][j] = 1
        return result

    @staticmethod
    def matrix_to_image(matrix):
        return Image.fromarray(matrix, mode='L')

    @staticmethod
    def bin_to_image(bin_matrix):
        assert bin_matrix.shape == (64, 64)

        result = np.zeros(shape=(64, 64), dtype=int)
        for i in range(64):
            for j in range(64):
                if bin_matrix[i][j] == 0:
                    result[i][j] = 100
                else:
                    result[i][j] = 255
        return result

    @staticmethod
    def crop_matrix(matrix, block_size=8):
        """
        Дробит матрицу размером 512х512 на блоки 8х8
        и возвращает в массив размерностью (4096, 8, 8)
        """
        assert matrix.shape[0] == matrix.shape[1]
        n = matrix.shape[0] // block_size
        block_count = int(n ** 2)
        cropped = np.zeros(shape=(block_count, block_size, block_size), dtype=float)
        for i in range(n):
            for j in range(n):
                cropped[i * n + j] = matrix[block_size * i:block_size * (i + 1), block_size * j:block_size * (j + 1)]
        return cropped

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
