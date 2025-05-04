import io
import re
import cv2
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from scipy.linalg import hadamard
from scipy.ndimage import rotate
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr
)


class Utilities:

    def __init__(self):
        pass

    @staticmethod
    def get_ssim(im1, im2):
        return ssim(im1, im2, data_range=255)

    @staticmethod
    def get_psnr(im1, im2):
        return psnr(im1, im2, data_range=255)

    @staticmethod
    def get_normal_correlation(im1, im2):
        return np.sum(np.multiply(im1, im2)) / np.sum(np.multiply(im1, im1))

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
        assert matrix.shape == (8, 8)

        h = hadamard(8) / 1
        if round:
            return np.round(np.matmul(np.matmul(h, matrix), h) / 8)

        return np.matmul(np.matmul(h, matrix), h) / 8

    @staticmethod
    def get_avg(matrix, bit_num):
        assert matrix.shape == (8, 8)

        if bit_num == 0:
            m = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ])
        elif bit_num == 1:
            m = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ])
        elif bit_num == 2:
            m = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0]
            ])
        else:
            m = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 1, 1, 1, 0]
            ])

        return np.sum(np.multiply(matrix, m)) / 8

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

    @staticmethod
    def get_entropies(block_array):
        """ Рассчитывает комплексную энтропию """
        entropies = []
        for i in range(len(block_array)):
            array = block_array[i].flatten()
            array_norm = array / np.sum(array)

            shannon_entropy = Utilities.get_shannon_entropy(array_norm)
            pal_entropy = Utilities.get_pal_entropy(array_norm)

            entropies.append({
                'index': i,
                'entropy': shannon_entropy + pal_entropy,
                'origin': block_array[i]
            })

        sorted_entropies = sorted(entropies, key=lambda d: d['entropy'])
        for num, i in enumerate(sorted_entropies):
            i['sorted_i'] = num

        assert len(sorted_entropies) == 4096

        return sorted_entropies

    @staticmethod
    def get_all_candidates(block_array):
        entropies = Utilities.get_entropies(block_array)
        return entropies  # entropies[:2048]

    @staticmethod
    def extracting(watermark, secret_key):
        """
        Получает ЦВЗ из покрывающего объекта
        """
        result = np.zeros(shape=(64, 64), dtype=float)

        # Шаг 1. Разбивает матрицу (со встроенным ЦВЗ) размером 512 х 512 на массив размерностью (4096, 8, 8)
        block_array = Utilities.crop_matrix(Utilities.image_to_matrix(watermark))

        # Шаг 2. Использую секретный ключ получаем адреса блоков в которых хранятся ЦВЗ
        for i in range(64):
            for j in range(64):
                bit_count = i * 64 + j
                block_ind = int(secret_key[i][j])
                bit_num = bit_count % 4

                # Шаг 3. Применить преобразование Адамара к полученным блокам
                if bit_num == 0:
                    hadamard = Utilities.get_hadamard(block_array[block_ind])

                    # Шаг 4. Посчитать неравенство
                    result[i][j] = 1 if hadamard[2][1] >= Utilities.get_avg(hadamard, 0) else 0
                    result[i][j + 1] = 1 if hadamard[2][5] >= Utilities.get_avg(hadamard, 1) else 0
                    result[i][j + 2] = 1 if hadamard[6][1] >= Utilities.get_avg(hadamard, 2) else 0
                    result[i][j + 3] = 1 if hadamard[6][5] >= Utilities.get_avg(hadamard, 3) else 0

        return result

    @staticmethod
    def mutate(arr):
        assert len(arr) == 4096
        while True:
            ind = random.randrange(0, 4096)
            if np.sum(arr) > 1024:
                arr[ind] = 0
            elif np.sum(arr) < 1024:
                arr[ind] = 1
            else:
                assert len(arr) == 4096
                assert np.sum(arr) == 1024
                return arr

    @staticmethod
    def get_data(file_path, name="value", size=10):
        """ Функция возвращает датасет готовый к визуализации """

        def preprocess_data(x, ind=0):
            x = x.replace("'", '"')
            x = x.replace("None", "null")
            x = re.sub(r"array\(\[[ 0-9,.]*\]\)", "null", x)
            return json.loads(f'{x}')[ind][name]

        dataframe = pd.read_csv(file_path, names=("epoch", "data"))

        for i in range(size):
            dataframe[i] = dataframe["data"].map(lambda x: preprocess_data(x, ind=i))

        dataframe = dataframe.drop(columns=["data"])
        dataframe = dataframe.drop(columns=["epoch"])

        return dataframe

    @staticmethod
    def draw_scatter(data, figure):
        """
        Отрисовывает полет светлячка, а вернее его
        значения с течением шагов алгоритма и итераций
        """
        for i in range(data.shape[1]):
            figure.add_trace(go.Scatter(x=data.index, y=data[i],
                                        mode='lines+markers',
                                        name=i))
        return figure

    @staticmethod
    def draw_lines(data, figure):
        """
        Отрисовывает полет светлячка, а вернее его
        значения с течением шагов алгоритма и итераций
        """
        for i in range(data.shape[1]):
            figure.add_trace(go.Scatter(x=data.index, y=data[i],
                                        mode='lines',
                                        name=i))
        return figure

    @staticmethod
    def draw_candles(data, figure):
        """
        Отрисовывает полет светлячка, а вернее его
        значения с течением шагов алгоритма и итераций
        """
        data = data.T
        for i in range(data.shape[1]):
            figure.add_trace(go.Box(y=data[i],
                                    quartilemethod="linear",
                                    name=i))
        return figure


class Attack:
    def __init__(self, image):
        self.image = image
        self.image_matrix = Utilities.image_to_matrix(image)
        self.mf = self.median_filter()
        self.gs3 = self.gaussian_filter()
        self.gs5 = self.gaussian_filter(ksize=5)
        self.avr = self.average_filter(ksize=3)
        self.shr = self.sharpening_filter()
        self.his = self.histogram_equalization()
        self.gc2 = self.gamma_correction()
        self.gc4 = self.gamma_correction(gamma=0.4)
        self.gn1 = self.gaussian_noise()
        self.gn5 = self.gaussian_noise(n=0.005)
        self.gn9 = self.gaussian_noise(n=0.009)
        self.sp1 = self.salt_and_pepper_noise()
        self.sp2 = self.salt_and_pepper_noise(n=0.02)
        self.sp3 = self.salt_and_pepper_noise(n=0.03)
        self.rt5 = self.rotation()
        self.rt45 = self.rotation(angle=45)
        self.rt90 = self.rotation(angle=90)
        self.com70 = self.compression()
        self.com80 = self.compression(quality=80)
        self.com90 = self.compression(quality=90)
        self.crp_ct = self.cropping_quarter()
        self.crp_tl = self.cropping_quarter("top-left")
        self.crp_br = self.cropping_quarter("bottom-right")
        self.scl_1024 = self.scaling()
        self.scl_256 = self.scaling(size=256)

    def median_filter(self, ksize=3):
        """
        Применяет медианный фильтр на матрицах с заданным размером
        """
        return cv2.medianBlur(self.image_matrix, ksize)

    def gaussian_filter(self, ksize=3):
        """
        Применяет фильтр Гаусса на матрицах с заданным размером
        :param ksize: Possible values: 3, 5
        """
        return cv2.GaussianBlur(self.image_matrix, ksize=(ksize, ksize), sigmaX=0.4, sigmaY=0.4)

    def average_filter(self, ksize=3):
        """
        Применяет усредненную фильтрацию на матрицах с заданным размером
        :param ksize: 3
        """
        return cv2.blur(self.image_matrix, (ksize, ksize))

    def sharpening_filter(self):
        """
        Фильтр увеличивает контраст соседних пикселей
        """
        gaussian_blur = cv2.GaussianBlur(self.image_matrix, (7, 7), sigmaX=2)
        return cv2.addWeighted(self.image_matrix, 1.5, gaussian_blur, -0.5, 0)

    def histogram_equalization(self):
        """
        Изменение контрастности изображения с использованием гистограмм
        """
        return cv2.equalizeHist(self.image_matrix)

    def gamma_correction(self, gamma=0.2):
        """
        Искажения яркости пикселей с использованием степенной функции с заданным параметром
        :param gamma: Возможные параметры: 0.2, 0.4
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(self.image_matrix, table)

    def gaussian_noise(self, n=0.001):
        """
        Нанесение статистического шума, имеющий плотность вероятности,
        равную плотности вероятности нормального распределения
        :param n: Возможные параметры: 0.001, 0.005, 0.009
        """
        noise = np.random.normal(0, 25, (512, 512)).astype(np.uint8)
        binary = np.random.choice([0, 1], size=(512, 512), p=[1 - n, n])

        return np.array(self.image_matrix + np.multiply(noise, binary), dtype=np.uint8)

    def salt_and_pepper_noise(self, n=0.01):
        """
        С заданной вероятностью затемняет или засвечивает пиксели изображения
        :param n: Possible values: 0.01, 0.02, 0.03
        """
        # Pepper noise
        binary = np.random.choice([0, 1], size=(512, 512), p=[1 - n / 2, n / 2])
        image_matrix = np.array(self.image_matrix, copy=True)

        image_matrix = image_matrix - np.multiply(image_matrix, binary)

        # Salt noise
        binary = np.random.choice([0, 1], size=(512, 512), p=[1 - n / 2, n / 2])
        salt = np.multiply(binary, 255)
        sub = np.multiply(image_matrix, binary)
        image_matrix = image_matrix - sub + salt

        return np.array(image_matrix, dtype=np.uint8)

    def cropping_quarter(self, place="center"):
        """
        В заданной области изображения затемняет или засвечивает 1/4 всех пикселей
        :param place: Возможные параметры: center, top-left, bottom-right
        """
        displace = 128
        if place == "top-left":
            displace = 0
        elif place == "bottom-right":
            displace = 256
        noise = np.ones(shape=(512, 512), dtype=int)
        for i in range(displace, displace + 256):
            for j in range(displace, displace + 256):
                noise[i][j] = 0

        return np.array(np.multiply(self.image_matrix, noise), dtype=np.uint8)

    def scaling(self, size=1024):
        """
        Растягивает / сжимает изображения и возвращает к изначальным размерам с потерей качества
        :param size: Возможные параметры: 1024 (512 -> 1024 -> 512), 256 (512 -> 245 -> 512)
        """
        result = cv2.resize(self.image_matrix, (size, size), 2, 2, cv2.INTER_LINEAR)
        return cv2.resize(result, (512, 512), 2, 2, cv2.INTER_LINEAR)

    def rotation(self, angle=5):
        """
        Вращает изображение против часовой стрелки на заданный угол
        :param angle: Возможные параметры: 5, 45, 90
        """
        return np.array(rotate(self.image_matrix, angle=angle), dtype=np.uint8)

    def compression(self, quality=70):
        """
        Применяет сжатие JPEG с заданным параметром качества
        :param quality: Возможные параметры: 70, 80, 90
        """
        buffered = io.BytesIO()
        self.image.save(buffered, format="PNG", quality=quality)

        return Utilities.image_to_matrix(Image.open(buffered))
