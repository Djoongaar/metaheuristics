import numpy as np
import random
import bisect
from PIL import Image
from scipy.linalg import hadamard
from src import Utilities, Attack
from tqdm import tqdm
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr
)


class Base:
    def __init__(self, image_path: str, embedded_image_path: str):
        self.image = Utilities.get_image(image_path)
        self.image_matrix = Utilities.image_to_matrix(self.image)
        self.block_array = Utilities.crop_matrix(self.image_matrix)
        self.embedded_image = Utilities.get_image(embedded_image_path)
        self.embedded_image_bin = Utilities.image_to_bin(self.embedded_image)
        self.all_candidates = Utilities.get_all_candidates(self.block_array)


class Firefly:

    def __init__(self):
        self.alpha = 0.01
        self.beta0 = 1.0
        self.gamma = 0.01
        self.theta = 0.97
        self.firefly_min = 0
        self.firefly_max = 20
        self.firefly_stop_iterations = 0
        self.firefly_iteration_max = 3
        self.firefly_stop = False
        self.best_firefly = None
        self.best_firefly_score = None
        self.firefly_elite_size = 3
        self.firefly_population = []

    def init_fireflies(self):
        return [
            {
                'value': np.random.uniform(self.firefly_min, self.firefly_max),
                'score': None,
                'attract': self.beta0
            } for _ in range(self.firefly_min, self.firefly_max)
        ]

    def fireflies(self):
        candidate = self.generation[self.best_candidate['index']]

        # Инициализация популяции светлячков
        self.firefly_population = self.init_fireflies()

        # Если были предыдущие итерации, то добавляем в популяцию лучшее решение (элитные особи)
        if self.best_firefly is not None:
            for _ in range(self.firefly_elite_size):
                self.firefly_population[random.randint(0, len(self.firefly_population) - 1)]['value'] = \
                    self.best_firefly['value']

        # Запускаем алгоритм только если не достигнут предел счетчика self.firefly_stop_iterations
        if self.firefly_stop_iterations >= self.firefly_iteration_max:
            return

        for i in range(len(self.firefly_population)):
            for j in range(len(self.firefly_population)):
                val_i = self.firefly_population[i]['value']
                val_j = self.firefly_population[j]['value']


                watermark_i = Watermark(candidate, self.embedded_image_bin, self.image_matrix, val_i)
                watermark_j = Watermark(candidate, self.embedded_image_bin, self.image_matrix, val_j)

                self.firefly_population[i]['score'] = watermark_i.score
                self.firefly_population[j]['score'] = watermark_j.score

                # Выполняем шаг полета светлячка


                step = self.alpha * self.firefly_max

                if watermark_j.score > watermark_i.score:
                    r = val_i - val_j
                    attract = self.firefly_population[i]['attract']
                    attract = attract * np.exp(-self.gamma * r ** 2)

                    val_j_new = val_j + attract * (val_i - val_j) + step
                    self.firefly_population[j]['value'] = val_j_new
                    self.firefly_population[i]['attract'] = attract

                    # Если оценка улучшилась - обновляем self.best_firefly,
                    # а иначе увеличиваем счетчик self.firefly_stop_iterations
                    if self.best_firefly['score'] is None or self.best_firefly['score'] > watermark_i.score:
                        self.best_firefly = self.firefly_population[i]

                        # Сохраняю эволюцию светлячков для дальнейшей визуализации
                        print('Firefly:', self.best_firefly)

                elif watermark_j.score < watermark_i.score:
                    r = val_j - val_i
                    attract = self.firefly_population[j]['attract']
                    attract = attract * np.exp(-self.gamma * r ** 2)

                    val_i_new = val_i + attract * (val_j - val_i) + step
                    self.firefly_population[i]['value'] = val_i_new
                    self.firefly_population[j]['attract'] = attract

                    # Если оценка улучшилась - обновляем self.best_firefly,
                    # а иначе увеличиваем счетчик self.firefly_stop_iterations
                    if self.best_firefly['score'] is None or self.best_firefly['score'] > watermark_j.score:
                        self.best_firefly = self.firefly_population[j]

                        # Сохраняю эволюцию светлячков для дальнейшей визуализации
                        print('Firefly:', self.best_firefly)

        # Увеличиваем счетчик self.firefly_stop_iterations
        self.firefly_stop_iterations += 1


class Genetic:

    def __init__(self):
        self.best_candidate = None
        self.best_candidate_indexes = None
        self.elite_candidates = []
        self.generation_size = 100
        self.elite_size = 20
        self.chromosome_length = 2048
        self.generation_bin = [self.random_candidates() for _ in range(self.generation_size)]
        self.generation = self.bin_to_index()

    def random_candidates(self):
        candidate = np.random.choice([0, 1], size=(self.chromosome_length,), p=[1 / 2, 1 / 2])
        return Utilities.mutate(candidate)

    def bin_to_index(self):
        result = []
        for chromosome_bin in self.generation_bin:
            chromosome = []
            for i in range(self.chromosome_length):
                if chromosome_bin[i] == 1:
                    chromosome.append(self.all_candidates[i])
            result.append(chromosome)
        return result

    def crossing(self):
        new_generation = []
        new_generation.extend([i['value'] for i in self.last_score])
        new_generation.extend([i['value'] for i in self.elite_candidates])
        new_generation = random.sample(new_generation, len(new_generation))
        self.generation_bin = []
        for i in range(0, len(new_generation), 2):
            child_1 = np.concatenate((
                new_generation[i][:self.chromosome_length // 4],  # [0:1/4]
                new_generation[i + 1][self.chromosome_length // 4: self.chromosome_length * 3 // 4],  # [1/4:3/4]
                new_generation[i][self.chromosome_length * 3 // 4:]  # [3/4:-1]
            ))
            child_2 = np.concatenate((
                new_generation[i + 1][:self.chromosome_length // 4],  # [0:1/4]
                new_generation[i][self.chromosome_length // 4: self.chromosome_length * 3 // 4],  # [1/4:3/4]
                new_generation[i + 1][self.chromosome_length * 3 // 4:]  # [3/4:-1]
            ))
            self.generation_bin.append(Utilities.mutate(child_1))
            self.generation_bin.append(Utilities.mutate(child_2))

        self.generation = self.bin_to_index()


class HybridMetaheuristic(Base, Genetic, Firefly):

    def __init__(self, image_path, embedded_image_path):
        Base.__init__(self, image_path, embedded_image_path)
        Genetic.__init__(self)
        Firefly.__init__(self)
        self.max_generations = 20
        self.genetic_evolution = []
        self.last_score = None

    def evaluate(self):
        results = []
        for num, candidate in enumerate(self.generation):
            if self.best_firefly is None:
                self.best_firefly = {'value': (self.firefly_min + self.firefly_max) / 2, 'score': None, 'attract': 0}
            watermark = Watermark(candidate, self.embedded_image_bin, self.image_matrix, self.best_firefly['value'])
            watermark_data = {
                'index': num,
                'score': watermark.score,
                'ssim': watermark.ssim,
                'psnr': watermark.psnr,
                'value': self.generation_bin[num],
                'nc': watermark.avg_nc
            }
            results.append(watermark_data)

            if (len(self.elite_candidates) < self.elite_size or
                    self.elite_candidates[-1]['score'] > watermark_data['score']):
                bisect.insort(self.elite_candidates, watermark_data, key=lambda x: x['score'])

        self.elite_candidates = self.elite_candidates[:self.elite_size]
        results = sorted(results, key=lambda x: x['score'])

        # Переопределяем лучшего кандидата и записываем индексы блоков
        self.best_candidate = self.elite_candidates[0]
        self.best_candidate_indexes = []

        for num, i in enumerate(self.best_candidate['value']):
            if i:
                self.best_candidate_indexes.append(self.all_candidates[num])

        best_candidate = {
            'score': round(self.best_candidate['score'], 4),
            'ssim': round(self.best_candidate['ssim'], 4),
            'psnr': round(self.best_candidate['psnr'], 2),
            'nc': round(self.best_candidate['nc'], 4),
            'th': round(self.best_firefly['value'], 2)
        }
        ### Printing intermediate results during iterations
        self.genetic_evolution.append(best_candidate)
        print('Genetic:', best_candidate)

        self.last_score = results[:self.generation_size - self.elite_size]

    def evolution(self):
        for _ in range(self.max_generations):
            # for _ in tqdm(range(self.max_generations)):
            self.evaluate()
            self.crossing()
            self.fireflies()


class Watermark:

    def __init__(self, candidate_blocks, embedded_image_bin, image_matrix, threshold):
        self.image_matrix = image_matrix
        self.embedded_image_bin = embedded_image_bin
        self.candidate_blocks = candidate_blocks
        self.secret_key = np.zeros(shape=(64, 64), dtype=float)
        self.embedding_threshold = threshold
        self.embedded_matrix = np.zeros(shape=(512, 512), dtype=int)
        self.watermark_matrix = self.embedding(self.candidate_blocks)
        self.watermark = Utilities.matrix_to_image(self.watermark_matrix)
        self.avg_nc = None
        self.ssim = Utilities.get_ssim(self.image_matrix, self.watermark_matrix)
        self.psnr = Utilities.get_psnr(self.image_matrix, self.watermark_matrix)
        self.extracted_image_bin = Utilities.extracting(self.watermark, self.secret_key)
        self.score = self.get_score()

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

    def get_score(self):
        # Согласно статье принимаем параметр гамма за 10
        gamma = 10

        # Произвожу разные атаки на стеганограмму и пытаюсь получить ЦВЗ
        attacked = Attack(self.watermark)

        mf = Utilities.extracting(Utilities.matrix_to_image(attacked.mf), self.secret_key)
        gs3 = Utilities.extracting(Utilities.matrix_to_image(attacked.gs3), self.secret_key)
        gs5 = Utilities.extracting(Utilities.matrix_to_image(attacked.gs5), self.secret_key)
        avr = Utilities.extracting(Utilities.matrix_to_image(attacked.avr), self.secret_key)
        shr = Utilities.extracting(Utilities.matrix_to_image(attacked.shr), self.secret_key)
        his = Utilities.extracting(Utilities.matrix_to_image(attacked.his), self.secret_key)
        gc2 = Utilities.extracting(Utilities.matrix_to_image(attacked.gc2), self.secret_key)
        gc4 = Utilities.extracting(Utilities.matrix_to_image(attacked.gc4), self.secret_key)
        gn1 = Utilities.extracting(Utilities.matrix_to_image(attacked.gn1), self.secret_key)
        gn5 = Utilities.extracting(Utilities.matrix_to_image(attacked.gn5), self.secret_key)
        gn9 = Utilities.extracting(Utilities.matrix_to_image(attacked.gn9), self.secret_key)
        sp1 = Utilities.extracting(Utilities.matrix_to_image(attacked.sp1), self.secret_key)
        sp2 = Utilities.extracting(Utilities.matrix_to_image(attacked.sp2), self.secret_key)
        sp3 = Utilities.extracting(Utilities.matrix_to_image(attacked.sp3), self.secret_key)
        crp_ct = Utilities.extracting(Utilities.matrix_to_image(attacked.crp_ct), self.secret_key)
        crp_tl = Utilities.extracting(Utilities.matrix_to_image(attacked.crp_tl), self.secret_key)
        crp_br = Utilities.extracting(Utilities.matrix_to_image(attacked.crp_br), self.secret_key)
        scl_1024 = Utilities.extracting(Utilities.matrix_to_image(attacked.scl_1024), self.secret_key)
        scl_256 = Utilities.extracting(Utilities.matrix_to_image(attacked.scl_256), self.secret_key)
        rt5 = Utilities.extracting(Utilities.matrix_to_image(attacked.rt5), self.secret_key)
        rt45 = Utilities.extracting(Utilities.matrix_to_image(attacked.rt45), self.secret_key)
        rt90 = Utilities.extracting(Utilities.matrix_to_image(attacked.rt90), self.secret_key)
        com70 = Utilities.extracting(Utilities.matrix_to_image(attacked.com70), self.secret_key)
        com80 = Utilities.extracting(Utilities.matrix_to_image(attacked.com80), self.secret_key)
        com90 = Utilities.extracting(Utilities.matrix_to_image(attacked.com90), self.secret_key)

        # Считаю параметр нормальной корреляции для каждого полученного ЦВЗ
        nc = [
            Utilities.get_normal_correlation(i, self.embedded_image_bin) for i in (
                mf, gs3, gs5, avr, shr, his, gc2, gc4, gn1, gn5, gn9, sp1,
                sp2, sp3, rt5, rt45, rt90, com70, com80, com90, crp_ct,
                crp_tl, crp_br, scl_1024, scl_256
            )
        ]

        # Вывожу среднее значение нормальной корреляции
        self.avg_nc = sum(nc) / len(nc)

        # Возвращаю значение целевой функции
        return gamma / self.psnr + 1 / self.ssim + 1 / self.avg_nc
