import numpy as np
import random
import bisect
from PIL import Image
from scipy.linalg import hadamard
from src import Utilities, Attack
from tqdm.notebook import tqdm, trange
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr
)


class Algorithm:
    def __init__(self, image_path: str, embedded_image_path: str):
        self.image = Utilities.get_image(image_path)
        self.image_matrix = Utilities.image_to_matrix(self.image)
        self.block_array = Utilities.crop_matrix(self.image_matrix)
        self.embedded_image = Utilities.get_image(embedded_image_path)
        self.embedded_image_bin = Utilities.image_to_bin(self.embedded_image)
        self.all_candidates = Utilities.get_all_candidates(self.block_array)


class Genetic(Algorithm):

    def __init__(self, image_path: str, embedded_image_path: str):
        super().__init__(image_path, embedded_image_path)
        self.best_candidate = None
        self.elite_candidates = []
        self.population_size = 100
        self.elite_size = 20
        self.max_iterations = 20
        self.chromosome_length = 2048
        self.population_bin = [self.random_candidates() for _ in range(self.population_size)]
        self.population = self.bin_to_index()
        self.evaluation = self.evaluate_population()
        self.breeding()

    def random_candidates(self):
        candidate = np.random.choice([0, 1], size=(self.chromosome_length,), p=[1 / 2, 1 / 2])
        return Utilities.mutate(candidate)

    def bin_to_index(self):
        result = []
        for chromosome_bin in self.population_bin:
            chromosome = []
            for i in range(self.chromosome_length):
                if chromosome_bin[i] == 1:
                    chromosome.append(self.all_candidates[i])
            result.append(chromosome)
        return result

    def evaluate_population(self):
        results = []
        for num, candidate in enumerate(self.population):
            watermark = Watermark(candidate, self.embedded_image_bin, self.image_matrix)
            watermark_data = {
                'index': num,
                'evaluation': watermark.evaluation,
                'ssim': watermark.ssim,
                'psnr': watermark.psnr,
                'value': self.population_bin[num]
            }
            results.append(watermark_data)

            if (len(self.elite_candidates) < self.elite_size or
                    self.elite_candidates[-1]['evaluation'] > watermark_data['evaluation']):
                bisect.insort(self.elite_candidates, watermark_data, key=lambda x: x['evaluation'])

        self.elite_candidates = self.elite_candidates[:self.elite_size]
        results = sorted(results, key=lambda x: x['evaluation'])
        self.best_candidate = self.elite_candidates[0]

        print('Best score: ', self.best_candidate)
        return results[:self.population_size - self.elite_size]

    def breeding(self):
        for _ in range(self.max_iterations):
            new_population = []
            new_population.extend([i['value'] for i in self.evaluation])
            new_population.extend([i['value'] for i in self.elite_candidates])
            new_population = random.sample(new_population, len(new_population))

            self.population_bin = []

            for i in range(0, len(new_population), 2):
                child_1 = np.concatenate((
                    new_population[i][:self.chromosome_length // 4],  # [0:1/4]
                    new_population[i + 1][self.chromosome_length // 4: self.chromosome_length * 3 // 4],  # [1/4:3/4]
                    new_population[i][self.chromosome_length * 3 // 4:]  # [3/4:-1]
                ))
                child_2 = np.concatenate((
                    new_population[i + 1][:self.chromosome_length // 4],  # [0:1/4]
                    new_population[i][self.chromosome_length // 4: self.chromosome_length * 3 // 4],  # [1/4:3/4]
                    new_population[i + 1][self.chromosome_length * 3 // 4:]  # [3/4:-1]
                ))
                self.population_bin.append(Utilities.mutate(child_1))
                self.population_bin.append(Utilities.mutate(child_2))

            self.population = self.bin_to_index()
            self.evaluation = self.evaluate_population()


class Firefly(Algorithm):

    def __init__(self, image_path: str, embedded_image_path: str):
        super().__init__(image_path, embedded_image_path)
        self.firefly_length = 1024
        self.population_size = 10
        self.max_iterations = 100
        self.population = [self.random_candidates() for _ in range(self.population_size)]

    def random_candidates(self):
        return random.sample(self.all_candidates, self.firefly_length)


class Watermark:

    def __init__(self, candidate_blocks, embedded_image_bin, image_matrix):
        self.image_matrix = image_matrix
        self.embedded_image_bin = embedded_image_bin
        self.candidate_blocks = candidate_blocks
        self.secret_key = np.zeros(shape=(64, 64), dtype=float)
        # TODO: Подобрать параметр с помощью алгоритма светлячков (гибридная мета-эвристика)
        self.embedding_threshold = 2
        self.embedded_matrix = np.zeros(shape=(512, 512), dtype=int)
        self.watermark_matrix = self.embedding(self.candidate_blocks)
        self.watermark = Utilities.matrix_to_image(self.watermark_matrix)
        self.ssim = Utilities.get_ssim(self.image_matrix, self.watermark_matrix)
        self.psnr = Utilities.get_psnr(self.image_matrix, self.watermark_matrix)
        self.extracted_image_bin = Utilities.extracting(self.watermark, self.secret_key)
        self.evaluation = self.evaluate()

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
                block_count = bit_count // 4
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

    def evaluate(self):
        # Согласно статье принимаем параметр гамма за 10
        gamma = 10

        # Произвожу разные атаки на стеганограмму и пытаюсь получить ЦВЗ
        attacked = Attack(self.watermark)

        # mf = Utilities.extracting(Utilities.matrix_to_image(attacked.mf), self.secret_key)
        gs3 = Utilities.extracting(Utilities.matrix_to_image(attacked.gs3), self.secret_key)
        gs5 = Utilities.extracting(Utilities.matrix_to_image(attacked.gs5), self.secret_key)
        # avr = Utilities.extracting(Utilities.matrix_to_image(attacked.avr), self.secret_key)
        # shr = Utilities.extracting(Utilities.matrix_to_image(attacked.shr), self.secret_key)
        # his = Utilities.extracting(Utilities.matrix_to_image(attacked.his), self.secret_key)
        # gc2 = Utilities.extracting(Utilities.matrix_to_image(attacked.gc2), self.secret_key)
        # gc4 = Utilities.extracting(Utilities.matrix_to_image(attacked.gc4), self.secret_key)
        # gn1 = Utilities.extracting(Utilities.matrix_to_image(attacked.gn1), self.secret_key)
        # gn5 = Utilities.extracting(Utilities.matrix_to_image(attacked.gn5), self.secret_key)
        # gn9 = Utilities.extracting(Utilities.matrix_to_image(attacked.gn9), self.secret_key)
        # sp1 = Utilities.extracting(Utilities.matrix_to_image(attacked.sp1), self.secret_key)
        # sp2 = Utilities.extracting(Utilities.matrix_to_image(attacked.sp2), self.secret_key)
        # sp3 = Utilities.extracting(Utilities.matrix_to_image(attacked.sp3), self.secret_key)
        # rt5 = Utilities.extracting(Utilities.matrix_to_image(attacked.rt5), self.secret_key)
        # rt45 = Utilities.extracting(Utilities.matrix_to_image(attacked.rt45), self.secret_key)
        # rt90 = Utilities.extracting(Utilities.matrix_to_image(attacked.rt90), self.secret_key)
        # com70 = Utilities.extracting(Utilities.matrix_to_image(attacked.com70), self.secret_key)
        # com80 = Utilities.extracting(Utilities.matrix_to_image(attacked.com80), self.secret_key)
        # com90 = Utilities.extracting(Utilities.matrix_to_image(attacked.com90), self.secret_key)
        # crp_ct = Utilities.extracting(Utilities.matrix_to_image(attacked.crp_ct), self.secret_key)
        # crp_tl = Utilities.extracting(Utilities.matrix_to_image(attacked.crp_tl), self.secret_key)
        # crp_br = Utilities.extracting(Utilities.matrix_to_image(attacked.crp_br), self.secret_key)
        # scl_1024 = Utilities.extracting(Utilities.matrix_to_image(attacked.scl_1024), self.secret_key)
        # scl_256 = Utilities.extracting(Utilities.matrix_to_image(attacked.scl_256), self.secret_key)

        # Считаю параметр нормальной корреляции для каждого полученного ЦВЗ
        # nc = [
        #     Utilities.get_normal_correlation(i, self.embedded_image_bin) for i in (
        #         mf, gs3, gs5, avr, shr, his, gc2, gc4, gn1, gn5, gn9, sp1,
        #         sp2, sp3, rt5, rt45, rt90, com70, com80, com90, crp_ct,
        #         crp_tl, crp_br, scl_1024, scl_256
        #     )
        # ]
        nc = [
            Utilities.get_normal_correlation(i, self.embedded_image_bin) for i in (
                gs3, gs5
            )
        ]
        # Вывожу среднее значение нормальной корреляции
        avg_nc = sum(nc) / len(nc)

        # Возвращаю значение целевой функции
        return gamma / self.psnr + 1 / self.ssim + 1 / avg_nc
