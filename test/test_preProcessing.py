from unittest import TestCase

from python.utils.DLUtils import convertWindowsToBeUsedByDeepLearning
from python.utils.dataUtils import getTrainTestKFoldBasedOnADay
from python.utils.preProcessing import extractSoundFeatures


class Test(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.filePath = "../../Data/"

    def test_extractFeature(self):
        subject = 0
        day = 0
        dataTrainTestKFold = getTrainTestKFoldBasedOnADay(subject, day, 30, self.filePath)
        for dataTrainEachFold in dataTrainTestKFold:
            X_train, Y_train, _, _ = dataTrainEachFold
            X_train, Y_train = convertWindowsToBeUsedByDeepLearning(X_train, Y_train)
            X_trainFeatures = extractSoundFeatures(X_train)
            TestCase.assertIsNotNone(self, X_trainFeatures)
