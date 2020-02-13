from unittest import TestCase

from python.deepLearning.lstmBasedDL import lstmTrain
from python.utils.dataUtils import getTrainTestKFoldBasedOnSession, getTrainTestKFoldBasedOnDayToDay


class Test(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.filePath = "../../Data/"

    def test_lstmBasedOnOneSession(self):
        kFold = 4
        subject = 0
        day = 0
        session = 0
        dataTrainTestKFold = getTrainTestKFoldBasedOnSession(subject, day, session, 30, self.filePath, kFold)
        for dataTrainEachFold in dataTrainTestKFold:
            X_train, Y_train, X_test, Y_test = dataTrainEachFold
            lstmTrain(X_train, Y_train, X_test, Y_test)
            TestCase.assertIsNotNone(self, dataTrainEachFold)

    def test_lstmBasedOnDayToDay(self):
        subject = 0
        dataTrainTestKFold = getTrainTestKFoldBasedOnDayToDay(subject, 30, self.filePath)
        for dataTrainEachFold in dataTrainTestKFold:
            X_train, Y_train, X_test, Y_test = dataTrainEachFold
            lstmTrain(X_train, Y_train, X_test, Y_test)
            TestCase.assertIsNotNone(self, dataTrainEachFold)
