import unittest
import numpy as np
import mathutil
import scipy.sparse as sp


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
        t = np.max(np.abs(result-np.sum(A, axis=0)))

        self.assertEqual(t, 0)

    def test_concat(self):
        """
        Test helpers.concatenate_csc_matrices_by_columns
        """
        from helpers import concatenate_csc_matrices_by_columns
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
        D = concatenate_csc_matrices_by_columns(
            sp.csr_matrix(A),
            sp.csr_matrix(B.T)
        )

        t = np.max(np.abs(C-D))

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

    def test_non_zero_csc(self):
        from mathutil import nonzero as nz
        from scipy import sparse as sp

        y = np.zeros(12000, dtype=float)
        y[2] = 1.
        y[3] = 1
        y[1000] = 1

        y2 = sp.csc_matrix(y)
        # t = np.max(np.abs(y.nonzero()-nz(y)))
        A = np.asarray(y2.nonzero())
        B = np.asarray(nz(y2))
        t1 = np.max(np.abs(A - B))
        self.assertEqual(t1, 0)

    def test_non_zero_csr(self):
        from mathutil import nonzero as nz
        from scipy import sparse as sp

        y = np.zeros(12000, dtype=float)
        y[2] = 1.
        y[3] = 1
        y[1000] = 1

        y1 = sp.csr_matrix(y)
        # t = np.max(np.abs(y.nonzero()-nz(y)))
        A = np.asarray(y1.nonzero())
        B = np.asarray(nz(y1))
        t = np.max(np.abs(A - B))
        self.assertEqual(t, 0)

if __name__ == '__main__':
    unittest.main()
