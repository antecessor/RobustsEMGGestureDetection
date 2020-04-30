from unittest import TestCase

from python.utils.dataUtils import getSignal, getSignalForADay, getTrainTestKFoldBasedOnSession, getTrainTestKFoldBasedOnADay, getTrainTestKFoldBasedOnDayToDay
from python.utils.preProcessing import windowingSignalWithOverLap


class Test(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.filePath = "../../Data/"

    def test_get_signal(self):
        dataLoaded = getSignal(1, 1, 1, self.filePath)
        TestCase.assertIsNotNone(self, dataLoaded)

    def test_getAllSignalForSubjectBetweenDay(self):
        data = getSignalForADay(0, 1, self.filePath)
        TestCase.assertIs(self, len(data), 3)
        data = getSignalForADay(13, 1, self.filePath)
        TestCase.assertIs(self, len(data), 2)

    def test_windowingSignalWithOverlap(self):
        data = getSignalForADay(0, 1, self.filePath)
        windows = windowingSignalWithOverLap(data[0][0]["data"], 100, 10)
        TestCase.assertIsNotNone(self, windows)

    def test_trainTestSplitBasedOnSession(self):
        kFold = 4
        subject = 0
        day = 0
        session = 0
        dataTrainTestKFold = getTrainTestKFoldBasedOnSession(subject, day, session, 30, self.filePath, kFold)
        for dataTrainEachFold in dataTrainTestKFold:
            TestCase.assertIsNotNone(self, dataTrainEachFold)

    def test_trainTestSplitBasedOnADay(self):
        subject = 0
        day = 0
        dataTrainTestKFold = getTrainTestKFoldBasedOnADay(subject, day, 30, self.filePath)
        for dataTrainEachFold in dataTrainTestKFold:
            TestCase.assertIsNotNone(self, dataTrainEachFold)

    def test_trainTestSplitBasedDayToDay(self):
        subject = 11
        dataTrainTestKFold = getTrainTestKFoldBasedOnDayToDay(subject, 'all', self.filePath)
        for dataTrainEachFold in dataTrainTestKFold:
            TestCase.assertIsNotNone(self, dataTrainEachFold)
