import unittest

import numpy as np
import scipy.sparse as sp

import mathutil
from mathutil import nonzero as nzt


class MyTestCase(unittest.TestCase):
    """
    Unit tests for some parts of the classifier.
    """

    def test_sum_rows(self):
        """
        Test mathutil.sum_rows
        """
        A = np.array([
            [1., 2., 3.],
            [0., -1., 1.],
            [3., 4., 5.]])
        result = np.zeros(shape=3)

        mathutil.sum_rows(A, result, A.shape[0], A.shape[1])
        t = np.max(np.abs(result - np.sum(A, axis=0)))

        self.assertEqual(t, 0)

    def test_concat(self):
        """
        Test helpers.concatenate_csc_matrices_by_columns
        """
        from helpers import concatenate_csr_matrices_by_columns
        A = np.array([
            [1., 2., 3.],
            [0., -1., 1.],
            [3., 4., 5.]])

        B = np.array([
            [1., 2., 3.]
        ], ndmin=2)

        t1 = sp.csr_matrix(A.T).A
        t2 = sp.csr_matrix(A).T.A
        C = np.array([
            [1., 2., 3., 1.],
            [0., -1., 1., 2.],
            [3., 4., 5., 3.]])
        D = concatenate_csr_matrices_by_columns(
            sp.csr_matrix(A),
            sp.csr_matrix(B.T)
        )
        t = -1
        try:
            t = np.max(np.abs(C - D))
        finally:
            self.assertEqual(t, 0)

    def test_cc_score(self):
        """
        Test MlcScore.score_accuracy
        :return:
        """
        from MlcScore import score_accuracy
        y = sp.csr_matrix([
            [1, 1, 1],
            [1, 0, 0]
        ])
        ypred = sp.csr_matrix([
            [1, 1, 1],
            [1, 1, 0]
        ])
        # A AND B / A OR B
        # 1 1 1 1 0 0 / 1 1 1 1 1 0 = 4/5 = 0.8
        self.assertAlmostEqual(score_accuracy(ypred, y), 0.8)
        y = sp.csr_matrix([
            [1, 1, 1]
        ])
        ypred = sp.csr_matrix([
            [1, 1, 1]
        ])
        self.assertAlmostEqual(score_accuracy(ypred, y), 1)
        y = sp.csr_matrix([
            [1, 1, 1]
        ])
        ypred = sp.csr_matrix([
            [0, 0, 0]
        ])
        self.assertAlmostEqual(score_accuracy(ypred, y), 0)
        y = sp.csr_matrix([
            [1, 1, 1, 0, 0]
        ])
        ypred = sp.csr_matrix([
            [0, 0, 0, 0 ,0]
        ])

        #self.assertAlmostEqual(score_accuracy(ypred, y), 0.4)

    def test_non_zero_csc(self):
        y = np.zeros(12000, dtype=float)
        y[2] = 1.
        y[3] = 1
        y[1000] = 1

        y2 = sp.csc_matrix(y)
        # t = np.max(np.abs(y.nonzero()-nz(y)))
        A = np.asarray(y2.nonzero())
        B = np.asarray(nzt(y2))
        t1 = np.max(np.abs(A - B))
        self.assertEqual(t1, 0)

    def test_non_zero_csr(self):

        y = np.zeros(12000, dtype=float)
        y[2] = 1.
        y[3] = 1
        y[1000] = 1

        y1 = sp.csr_matrix(y)
        # t = np.max(np.abs(y.nonzero()-nz(y)))
        A = np.asarray(y1.nonzero())
        B = np.asarray(nzt(y1))
        t = np.max(np.abs(A - B))
        self.assertEqual(t, 0)

    def test_gradient(self):
        DATASET_FILENAME = "data\delicious_data.txt"
        DATASET_TRAIN_SET_FILENAME = "data\delicious_trSplit.txt"
        DATASET_TEST_SET_FILENAME = "data\delicious_tstSplit.txt"

        from helpers import load_mlc_dataset
        X, y, header_info = load_mlc_dataset(DATASET_FILENAME,
                                             header=True,
                                             concatbias=True)
        DATASET_SIZE = int(header_info[0])
        FEATURE_NUMBER = int(header_info[1])
        LABEL_NUMBER = int(header_info[2])

        f1 = open(DATASET_TRAIN_SET_FILENAME)
        train_ind = np.loadtxt(fname=f1, delimiter=" ", dtype=int)
        f1.close()

        f2 = open(DATASET_TEST_SET_FILENAME)
        test_ind = np.loadtxt(fname=f2, delimiter=" ", dtype=int)
        f2.close()

        # Normalize train and test indexes
        train_ind = train_ind - 1
        test_ind = test_ind - 1

        X_train = X[train_ind[:, 0]]
        X_test = X[test_ind[:, 0]]

        y_train = y[train_ind[:, 0]]
        y_test = y[test_ind[:, 0]]

        from MlcClassifierChains import MlcClassifierChains
        from MlcLinReg import MlcLinReg
        mlc = MlcLinReg(grad_check=True)

        abs_max = mlc.fit(X_train, y_train[:, 0])
        self.assertLessEqual(abs_max, 1e-4)


if __name__ == '__main__':
    unittest.main()
