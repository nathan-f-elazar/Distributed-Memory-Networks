import csv
import random
import numpy as np
import __settings__


class StudentData():
    def __init__(self, students, meta):
        self.meta = meta
        self.index = 0
        self.labels = []
        target_correctness = [[students[i][-2][j] for j in range(1, len(students[i][-2]))] for i in range(len(students))]
        num_steps = [len(s[1]) for s in students]
        self.all_labels = target_correctness
        self.num_steps = num_steps
        self.raw_students = students
        self.students = meta.get_features(students)

    def get_batch(self):
        if self.index + self.meta.batch_size > len(self.students):
            s = np.s_[self.index:]
            self.index = 0
            batch = self.students[s]
            np.random.shuffle(self.students)
        else:
            s = np.s_[self.index:self.index + self.meta.batch_size]
            self.index += self.meta.batch_size
            batch = self.students[s]
        target_correctness = self.all_labels[s]
        target_correctness = [int(x) for l in target_correctness for x in l]
        num_steps = self.num_steps[s]
        self.labels.extend(target_correctness)
        return batch, target_correctness, num_steps

    def reset(self):
        self.labels = []
        self.index = 0


class Dataset():
    def __init__(self, filename, test_file=None, batch_size=32):
        self.batch_size = batch_size
        self.max_num_problems = 0
        self.test_portion = __settings__.test_portion
        self.valid_portion = __settings__.validate_portion
        self.train_portion = 1 - (self.test_portion + self.valid_portion)
        self.num_features = 0
        with open(filename, 'r') as f:
            self.data = self.load_data(f)
        if test_file is not None:
            with open(test_file, 'r') as f:
                test_data = self.load_data(f)
                self.test_portion = len(test_data) / (len(test_data) + len(self.data))
                self.data += test_data
        self.students = []
        for r in self.data[1:]:
            if len(r) == 1:
                break
            self.num_features += 1
        self.num_features -= 1

        id = 0
        self.item_counts = [0] * self.num_features
        for i in range(0, (len(self.data) - 1), self.num_features+2):
            if int(self.data[i][0]) <= __settings__.min_num_responses:
                continue
            for j in range(self.num_features):
                max_i = max(map(int, self.data[i + j + 1])) + 1
                if max_i > self.item_counts[j]:
                    self.item_counts[j] = max_i

            num_problems = int(self.data[i][0])
            if num_problems > self.max_num_problems:
                self.max_num_problems = num_problems

            self.students.append([self.data[j] for j in range(i,i+self.num_features+2)] + [id])
            id += 1
        self.num_skills = self.item_counts[0]
        self.max_num_problems += 1
        test_size = int(len(self.students) * self.test_portion)
        train_students = self.students[:-test_size]
        test_students = self.students[-test_size:]

        random.shuffle(train_students)

        valid_size = int(self.valid_portion * len(train_students))
        self.train_students = StudentData(train_students[valid_size:], self)
        self.valid_students = StudentData(train_students[:valid_size], self)
        self.test_students = StudentData(test_students, self)

    def reset(self):
        self.train_students.reset()
        self.valid_students.reset()
        self.test_students.reset()

    def get_data(self, is_training):
        return self.train_students if is_training == 'train' else self.valid_students if is_training == 'valid' else self.test_students

    def load_data(self, file):
        reader = csv.reader(file, delimiter=',')
        return [row for row in reader]

    def get_features(self, data):
        label_index = [[int(f_id) + int(c) * self.num_skills for (f_id, c) in zip(b[1], b[-2])] for b in data]
        label_index = [x for l in label_index for x in l]
        problem_index = [[j for j in range(len(data[i][1]))] for i in range(len(data))]
        problem_index = [x for l in problem_index for x in l]
        problem_index2 = [[i] * (len(data[i][1])) for i in range(len(data))]
        problem_index2 = [x for l in problem_index2 for x in l]
        X = np.ndarray([len(data), self.max_num_problems, self.num_features])
        X[problem_index2, problem_index, 0] = label_index
        if self.num_features == 1:
            return X

        label_index2 = [[int(f_id) for (f_id, c) in zip(b[2], b[-2])] for b in data]
        label_index2 = [x for l in label_index2 for x in l]
        X[problem_index2, problem_index, 1] = label_index2
        return X
