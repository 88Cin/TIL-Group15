import numpy as np
from sko.GA import GA
import IDM
import torch
import matplotlib.pyplot as plt
import pandas as pd

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fitness(param_set):
    global CARS, test_scenes

    param = {
                "v0": param_set[0],
                "s0": param_set[1] / 100,
                "T": param_set[2] / 100,
                "a": param_set[3] / 100,
                "b": param_set[4] / 100,
                "sigma": 4
                }

    errors = []

    for i in range(NUM_TEST):
        test_scene = test_scenes[i]
        errors.append(single_test(test_scene, param))

    return np.mean(errors)

def single_test(test_scene, param):

    car_front = IDM.Car(4.5, test_scene[:, 1] + 2.25, test_scene[:, 3], test_scene[:, 3])
    car_follow = IDM.Car(5, [test_scene[0, 4] + 2.5], [test_scene[0, 5]], [test_scene[0, 6]])
    car_follow_real = IDM.Car(5, [test_scene[:, 4] + 2.5], [test_scene[:, 5]], [test_scene[:, 6]])

    LENGTH = len(car_front.xs)

    idm = IDM.IntelDriverModel(param)
    idm.update(car_front, car_follow, 0.1)
    return np.sqrt(1 / LENGTH * np.sum(np.square([x1 - x2 for x1, x2 in zip(car_follow_real.xs[0], car_follow.xs)])))


