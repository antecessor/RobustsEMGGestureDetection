import numpy as np
import pandas as pd


class KFoldEvaluate:
    def __init__(self) -> None:
        super().__init__()
        self.confusionMats = []
        self.results = pd.DataFrame()

    def addCM(self, cm):
        self.confusionMats.append(cm)

    def calculateParameters(self):
        keys = ['Overall ACC', 'Kappa', 'Overall RACC', 'TPR Macro', 'ACC Macro', 'F1 Macro', 'TPR Micro', 'PPV Micro', 'F1 Micro', 'Scott PI', 'Gwet AC1', 'Bennett S', 'Kappa Standard Error',
                'Standard Error', 'Response Entropy', 'Reference Entropy', 'Cross Entropy', 'Joint Entropy', 'Conditional Entropy', 'Lambda A', 'Kappa Unbiased', 'Overall RACCU',
                'Kappa No Prevalence', 'Mutual Information', 'Hamming Loss', 'NIR', 'Overall CEN', 'Overall MCEN', 'RR', 'CBA', 'AUNU', 'AUNP', 'RCI']
        params = []
        for index, cm in enumerate(self.confusionMats):
            paramsForOneCM = []
            for key in keys:
                paramsForOneCM.append(cm.overall_stat[key])
            params.append(paramsForOneCM)
        params = np.array(params).transpose()
        self.confusionMats.clear()
        return keys, np.mean(params, axis=1), np.std(params, axis=1)

    def addColumnToResults(self, colName, vals, index=1):
        self.results.insert(index, colName, vals, allow_duplicates=True)

    def clearResults(self):
        self.results = pd.DataFrame()

    def writeResOnExcel(self, name="results.xlsx"):
        self.results.to_excel(name)
