from unittest import TestCase

from python.deepLearning.lstmBasedDL import lstmTrain, ldaTrain
from python.utils.dataUtils import getTrainTestKFoldBasedOnSession, getTrainTestKFoldBasedOnDayToDay, getTrainTestKFoldBasedOnADay, getTrainTestKFoldBasedOnDayToDayWithNoShift
from python.utils.reportUtils import KFoldEvaluate


class Test(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.filePath = "../../Data/"

    def test_lstmBasedOnOneSession(self):
        kFold = 4
        for subject in range(15):
            for day in range(5):
                if subject < 11:
                    sessionTotal = 3
                else:
                    sessionTotal = 2

                for session in range(sessionTotal):
                    kfoldEvaluate = KFoldEvaluate()
                    dataTrainTestKFold = getTrainTestKFoldBasedOnSession(subject, day, session, 'all', self.filePath, kFold)
                    fold = 0
                    for dataTrainEachFold in dataTrainTestKFold:
                        X_train, Y_train, X_test, Y_test = dataTrainEachFold
                        model, cm = lstmTrain(X_train, Y_train, X_test, Y_test)
                        kfoldEvaluate.addCM(cm)
                        cm.save_csv("WithinSessionSub{}_day{}_session{}_fold{}".format(subject, day, session, fold))
                        fold = fold + 1
                    labels, mean, std = kfoldEvaluate.calculateParameters()
                    kfoldEvaluate.addColumnToResults("keys", labels, 0)
                    kfoldEvaluate.addColumnToResults("mean_sub{}_day{}_session{}".format(subject, day, session), mean)
                    kfoldEvaluate.addColumnToResults("std_sub{}_day{}_session{}".format(subject, day, session), std)
                    kfoldEvaluate.writeResOnExcel("WithinSession_sub{}_day{}_session{}.xlsx".format(subject, day, session))
        TestCase.assertTrue(self, True)

    def test_lstmBasedOnDayToDay(self):
        for subject in range(0, 15):
            kfoldEvaluate = KFoldEvaluate()
            print("start for subject {}".format(subject))
            dataTrainTestKFold = getTrainTestKFoldBasedOnDayToDay(subject, 'all', self.filePath)
            fold = 0
            for dataTrainEachFold in dataTrainTestKFold:
                X_train, Y_train, X_test, Y_test = dataTrainEachFold
                model, cm = ldaTrain(X_train, Y_train, X_test, Y_test)
                cm.save_csv("DayToDaySub{}_fold{}".format(subject, fold))
                fold = fold + 1
                kfoldEvaluate.addCM(cm)
            labels, mean, std = kfoldEvaluate.calculateParameters()

            kfoldEvaluate.addColumnToResults("keys", labels, 0)
            kfoldEvaluate.addColumnToResults("mean_sub{}".format(subject), mean)
            kfoldEvaluate.addColumnToResults("std_sub{}".format(subject), std)
            kfoldEvaluate.writeResOnExcel("DayToDaySub{}.xlsx".format(subject))
        TestCase.assertTrue(self, True)

    def test_lstmBasedOnDayToDayWithShift(self):
        for subject in range(0, 15):
            kfoldEvaluate = KFoldEvaluate()
            print("start for subject {}".format(subject))
            dataTrainTestKFold = getTrainTestKFoldBasedOnDayToDayWithNoShift(subject, 'all', self.filePath)
            fold = 0
            for dataTrainEachFold in dataTrainTestKFold:
                X_train, Y_train, X_test, Y_test = dataTrainEachFold
                model, cm = lstmTrain(X_train, Y_train, X_test, Y_test)
                cm.save_csv("DayToDaySub{}_fold{}".format(subject, fold))
                fold = fold + 1
                kfoldEvaluate.addCM(cm)
            labels, mean, std = kfoldEvaluate.calculateParameters()

            kfoldEvaluate.addColumnToResults("keys", labels, 0)
            kfoldEvaluate.addColumnToResults("mean_sub{}".format(subject), mean)
            kfoldEvaluate.addColumnToResults("std_sub{}".format(subject), std)
            kfoldEvaluate.writeResOnExcel("DayToDaySub{}.xlsx".format(subject))
        TestCase.assertTrue(self, True)

    def test_lstmBasedOnOneDay(self):
        kFold = 4
        for subject in range(0, 15):
            for day in range(5):
                # if subject < 11:
                #     sessionTotal = 3
                # else:
                #     sessionTotal = 2

                kfoldEvaluate = KFoldEvaluate()
                dataTrainTestKFold = getTrainTestKFoldBasedOnADay(subject, day, 'all', self.filePath)
                fold = 0
                for dataTrainEachFold in dataTrainTestKFold:
                    X_train, Y_train, X_test, Y_test = dataTrainEachFold
                    model, cm = lstmTrain(X_train, Y_train, X_test, Y_test)
                    kfoldEvaluate.addCM(cm)
                    cm.save_csv("WithinDay{}_day{}_fold{}".format(subject, day, fold))
                    fold = fold + 1
                labels, mean, std = kfoldEvaluate.calculateParameters()
                kfoldEvaluate.addColumnToResults("keys", labels, 0)
                kfoldEvaluate.addColumnToResults("mean_sub{}_day{}".format(subject, day), mean)
                kfoldEvaluate.addColumnToResults("std_sub{}_day{}".format(subject, day), std)
                kfoldEvaluate.writeResOnExcel("WithinDay_sub{}_day{}.xlsx".format(subject, day))
        TestCase.assertTrue(self, True)
