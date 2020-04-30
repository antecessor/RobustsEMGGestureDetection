from unittest import TestCase

import numpy as np
from pycm import ConfusionMatrix

from python.utils.reportUtils import KFoldEvaluate


class TestKFoldEvaluate(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.kfoldEvaluate = KFoldEvaluate()

    def test_calculate_parameters(self):
        for i in range(3):
            cm = ConfusionMatrix(actual_vector=np.random.randint(5, size=200), predict_vector=np.random.randint(5, size=200))
            self.kfoldEvaluate.addCM(cm)
        keys, mean, std = self.kfoldEvaluate.calculateParameters()
        TestCase.assertIsNotNone(self, mean)

    def test_saveResultsInExcel(self):
        for i in range(3):
            cm = ConfusionMatrix(actual_vector=np.random.randint(5, size=200), predict_vector=np.random.randint(5, size=200))
            self.kfoldEvaluate.addCM(cm)
        keys, mean, std = self.kfoldEvaluate.calculateParameters()
        self.kfoldEvaluate.addColumnToResults("keys", keys, 0)
        self.kfoldEvaluate.addColumnToResults("mean_sub1", mean)
        self.kfoldEvaluate.addColumnToResults("std_sub1", std)
        self.kfoldEvaluate.writeResOnExcel()
        TestCase.assertIsNotNone(self, mean)
