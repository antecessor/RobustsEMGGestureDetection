from unittest import TestCase

import numpy as np

from python.utils.VisualizationUtils import plotSignal
from python.utils.dataUtils import getSignal


class Test(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.filePath = "../../Data/"

    def test_plot_signal(self):
        dataLoaded = getSignal(1, 1, 1, self.filePath)
        signal = np.asarray(dataLoaded[100]['data'])
        plotSignal(signal[0, :])
