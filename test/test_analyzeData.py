import os
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# 0 = No Movement
# 1 = Wrist Supination
# 2 = Wrist Pronation
# 3 = Wrist Flexion
# 4 = Wrist Extension
# 5 = Hand Open
# 6 = Key Grip
# 7 = Fine Pinch
class test_analyzeData(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.labels = ["No Movement", "Wrist Supination", "Wrist Pronation", "Wrist Flexion", "Wrist Extension", "Hand Open", "Key Grip", "Fine Pinch"]

    def test_electrodeShift(self):
        path = "E:\Workspaces\phD project\\reports\WithinDayLSTM\Folds"
        files = os.listdir(path)
        ylabel = ["Accuracy", "F1-Score", "AUC"]
        for indparam, parm in enumerate(["ACC", "F1", "AUC"]):
            Data = []
            dataForSub = []
            interestedParams = [parm]
            for ind, file in enumerate(files):
                if not file.__contains__("matrix"):
                    sub = int(file.split("_")[0].replace("WithinDay", ''))
                    day = int(file.split("_")[1].replace("day", ""))

                    session = int(file.split("_")[2].replace("fold", "").replace(".csv", ""))
                    if session == 0:
                        if day == 0:
                            if ind != 0:
                                Data.append(dataForSub)
                            dataForSub = []
                        data = pd.read_csv(path + "\\" + file, index_col="Class")
                        data = data.loc[interestedParams].astype(float).values
                        dataForSub.append(data)
            Data.append(dataForSub)
            df = self.makeDataFrameFromResultForElectrodeShift(Data, interestedParams)
            # plt = sns.catplot(x="sub", y="vals",
            #                 hue="param", col="motion",
            #                 data=df, kind="box",
            #                 height=4, aspect=2, col_wrap=3)
            plt.clf()
            ax = sns.boxplot(x="motion", y="vals", data=df)
            ax.set(xlabel='subject', ylabel=ylabel[indparam])
            # ax = sns.swarmplot(x="sub", y="vals", data=df, color=".25")
            # plt.set_ylabels(ylabel[indparam])
            # plt.set_ylabels(ylabel[indparam])
            plt.savefig("histogram_lectrodeShift{}.png".format(parm))

        pass

    def test_Day2DayWithNoElectrodeShift(self):
        path = "E:\Workspaces\phD project\\reports\Day2DayWithNoShiftLSTM\\folds"
        files = os.listdir(path)
        ylabel = ["Accuracy", "F1-Score", "AUC"]
        for indparam, parm in enumerate(["ACC", "F1", "AUC"]):
            Data = []
            dataForSub = []
            previousSub = 0
            interestedParams = [parm]
            for ind, file in enumerate(files):
                if not file.__contains__("matrix"):
                    sub = int(file.split("_")[0].replace("DayToDaySub", ''))
                    day = int(file.split("_")[1].replace("fold", "").replace(".csv", ""))
                    if day == 0:
                        if ind != 0:
                            Data.append(dataForSub)
                        dataForSub = []

                    data = pd.read_csv(path + "\\" + file, index_col="Class")
                    data = data.loc[interestedParams].astype(float).values
                    dataForSub.append(data)
            Data.append(dataForSub)
            df = self.makeDataFrameFromResultForDay2Day(Data, interestedParams)
            # plt = sns.catplot(x="sub", y="vals",
            #                 hue="param", col="motion",
            #                 data=df, kind="box",
            #                 height=4, aspect=2, col_wrap=3)
            plt.clf()
            ax = sns.boxplot(x="sub", y="vals", data=df)
            ax.set(xlabel='subject', ylabel=ylabel[indparam])
            # ax = sns.swarmplot(x="sub", y="vals", data=df, color=".25")
            # plt.set_ylabels(ylabel[indparam])
            # plt.set_ylabels(ylabel[indparam])
            plt.savefig("histogram_Day2DayWithNoShift{}_HueParam.png".format(parm))

        pass

    def test_Day2DayWithShift(self):
        path = "E:\Workspaces\phD project\\reports\Day2DayLSTM\Folds"
        files = os.listdir(path)
        ylabel = ["Accuracy", "F1-Score", "AUC"]
        for indparam, parm in enumerate(["ACC", "F1", "AUC"]):
            Data = []
            dataForSub = []
            previousSub = 0
            interestedParams = [parm]
            for ind, file in enumerate(files):
                if not file.__contains__("matrix"):
                    sub = int(file.split("_")[0].replace("DayToDaySub", ''))
                    day = int(file.split("_")[1].replace("fold", "").replace(".csv", ""))
                    if day == 0:
                        if ind != 0:
                            Data.append(dataForSub)
                        dataForSub = []

                    data = pd.read_csv(path + "\\" + file, index_col="Class")
                    data = data.loc[interestedParams].astype(float).values
                    dataForSub.append(data)
            Data.append(dataForSub)
            df = self.makeDataFrameFromResultForDay2Day(Data, interestedParams)
            # plt = sns.catplot(x="sub", y="vals",
            #                 hue="param", col="motion",
            #                 data=df, kind="box",
            #                 height=4, aspect=2, col_wrap=3)
            plt.clf()
            ax = sns.boxplot(x="sub", y="vals", data=df)
            ax.set(xlabel='subject', ylabel=ylabel[indparam])
            # ax = sns.swarmplot(x="sub", y="vals", data=df, color=".25")
            # plt.set_ylabels(ylabel[indparam])
            plt.savefig("histogram_Day2DayWithNoShift{}_HueParam.png".format(parm))

        pass

    def makeDataFrameFromResultForDay2Day(self, Data, interestedParams):
        orderedData = []
        for subind, subData in enumerate(Data):
            for indDay, dayDataSub in enumerate(subData):

                for indInterestedParam, interestedParamItem in enumerate(interestedParams):
                    for indMotion, val in enumerate(dayDataSub[indInterestedParam, :]):
                        orderedData.append([val, interestedParamItem, subind + 1, indDay + 1, self.labels[indMotion]])
                # dayDataSub = np.append(dayDataSub, np.reshape(np.array(interestedParams), [len(interestedParams), 1]), axis=1)
                # dayDataSub = np.append(dayDataSub, (np.ones([3, 1]) * subind).astype(int), axis=1)
                # orderedData.extend(dayDataSub)

        orderedData = np.array(orderedData)
        column = ["vals", "param", "sub", "day", "motion"]
        # column.extend(["Param", "Sub"])
        df = pd.DataFrame(orderedData, columns=column)
        # for i in range(len(self.labels)):
        #     df[self.labels[i]] = df[self.labels[i]].astype(float)
        df["param"] = df["param"].astype(str)
        df["sub"] = df["sub"].astype(int)
        df["vals"] = df["vals"].astype(float)
        df["motion"] = df["motion"].astype(str)
        df["day"] = df["day"].astype(int)

        return df

    def makeDataFrameFromResultForElectrodeShift(self, Data, interestedParams):
        orderedData = []
        for subind, subData in enumerate(Data):
            for indDay, dayDataSub in enumerate(subData):

                for indInterestedParam, interestedParamItem in enumerate(interestedParams):
                    for indMotion, val in enumerate(dayDataSub[indInterestedParam, :]):
                        orderedData.append([val, interestedParamItem, subind + 1, indDay + 1, self.labels[indMotion]])
                # dayDataSub = np.append(dayDataSub, np.reshape(np.array(interestedParams), [len(interestedParams), 1]), axis=1)
                # dayDataSub = np.append(dayDataSub, (np.ones([3, 1]) * subind).astype(int), axis=1)
                # orderedData.extend(dayDataSub)

        orderedData = np.array(orderedData)
        column = ["vals", "param", "sub", "day", "motion"]
        # column.extend(["Param", "Sub"])
        df = pd.DataFrame(orderedData, columns=column)
        # for i in range(len(self.labels)):
        #     df[self.labels[i]] = df[self.labels[i]].astype(float)
        df["param"] = df["param"].astype(str)
        df["sub"] = df["sub"].astype(int)
        df["vals"] = df["vals"].astype(float)
        df["motion"] = df["motion"].astype(str)
        df["day"] = df["day"].astype(int)

        return df
