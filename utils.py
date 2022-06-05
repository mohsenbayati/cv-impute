from typing import List, NamedTuple

import numpy as np
import numpy.random as npr
import numpy.linalg as npl

from impute import SVD
from impute import Dataset
from impute import LagrangianImpute


# container classes
class Problem(NamedTuple):
    imputer: LagrangianImpute
    train: Dataset
    test: Dataset

    def fit(self, alphas=None, max_iters = 100):
        return self.imputer.fit(self.train, alphas=alphas, max_iters = max_iters)

    def eval_prediction(self, yh):
        ys = self.test.ys
        return rss(ys, yh)

    def eval_estimate(self, mat):
        yh = self.test.op(mat)

        return self.eval_prediction(yh)

    def eval(self, svds):
        mats = [svd.to_matrix() for svd in svds]
        errs = np.array([self.eval_estimate(mat) for mat in mats])

        return Solution(svds, mats, errs)

    def fit_and_eval(self, alphas, max_iters = 100):
        svds = self.fit(alphas, max_iters)

        return self.eval(svds)

    def alpha_max(self):
        return self.imputer.alpha_max(self.train)

    def get_alpha_seq(self, alpha: float, eta: float = 0.5):
        return self.imputer.get_alpha_seq(self.train, alpha_min=alpha, eta=eta)


class Solution(NamedTuple):
    svds: List[SVD]
    mats: List[np.ndarray]
    errs: np.ndarray


# evaluation helpers
def rss(y, yh) -> float:
    return np.sum((y - yh) ** 2).item()


def relative_error(b, bh):
    return rss(b, bh) / rss(b, 0)


# split and append helper functions
def generate_foldids(n, k, shuffle=True):
    indices = np.arange(n)
    if shuffle:
        npr.shuffle(indices)

    l = n // k
    c = n % k

    starts = [f * l + min(f, c) for f in range(k)]
    ends = [(f + 1) * l + min(f + 1, c) for f in range(k)]

    folds = [indices[s:e] for s, e in zip(starts, ends)]
    return folds


def add_to_subproblems(probs, folds):
    assert len(probs) == len(folds)

    for i, prob in enumerate(probs):
        for j, fold in enumerate(folds):
            if j == i:
                prob.test.extend(*fold)
            else:
                prob.train.extend(*fold)


def split_and_add_to_subproblems(probs, xs, ys, foldids=None, shuffle=True):
    assert len(xs) == len(ys)

    n = len(xs)
    k = len(probs)
    if not foldids:
        foldids = generate_foldids(n, k, shuffle)

    folds = [(subarray(xs, fold), subarray(ys, fold)) for fold in foldids]

    add_to_subproblems(probs, folds)

def cv_alpha_selection(probs, alpha_min_ratio, eta = 0.5):
    alpha_max = max(prob.alpha_max() for prob in probs)
    alpha_min = alpha_max * alpha_min_ratio

    alphas = []
    alpha = alpha_max
    while alpha > alpha_min:
        alphas.append(alpha)
        alpha *= eta
    
    alphas.append(alpha_min)
    #alphas = np.geomspace(alpha_max, alpha_max*.0001, 100)
    #alphas = np.geomspace(alpha_max, alpha_min, n_alpha)

    return(alphas)



def cv_impute(probs, refit_prob, n_alpha=100, alpha_min_ratio=0.01, alphas=None, verbose=1):
    # compute alpha sequence if not provided
    if not alphas:
        alphas = cv_alpha_selection(probs, alpha_min_ratio)

    # computing the test error for each fitted matrix
    sols = []
    for i, prob in enumerate(probs):
        if verbose == 2:
            print(f'fitting problem {i}')

        sol = prob.fit_and_eval(alphas)
        sols += [sol]

        if verbose == 2:
            print(f'optimum[{i}] = {np.argmin(sol.errs)}')

    # computing the total performance of each alpha
    errors = sum(sol.errs for sol in sols)

    opt = np.argmin(errors).item()
    est = sum(sol.mats[opt] for sol in sols) / len(sols)

    if (verbose==2):
        print(f'optimum = {opt}')
        print(f'optimum alpha = {alphas[opt]}')
        print(f'maximum alpha = {alphas[0]}')
        print(f'minimum alpha = {alphas[-1]}')
    
    alpha_stat = [alphas[-1], alphas[opt], alphas[0]]

    #------------ CV Single Output
    opt_single = np.argmin(sols[0].errs).item()
    est_single = sols[0].mats[opt_single]
    alpha_stat_single = [alphas[-1], alphas[opt_single], alphas[0]]

    #------------ CV Refit
    refit_alphas = refit_prob.get_alpha_seq(alphas[opt])

    svd = refit_prob.fit(refit_alphas)[-1]
    est_refit = svd.to_matrix()

    return est, est_single, est_refit, alpha_stat, alpha_stat_single


def oracle_impute(prob, alpha, verbose = 1):
    alphas = prob.get_alpha_seq(alpha)

    # computing the test error for each fitted matrix
    sol = prob.fit_and_eval(alphas)

    opt = np.argmin(sol.errs).item()
    est = sol.mats[opt]

    if (verbose==2):
        print(f'oracle alpha = {alphas[opt]}, smallest alpha = {alpha}')

    alpha_stat = [alphas[-1], alphas[opt], alphas[0]]

    return est, alpha_stat, sol

def overfit_impute(prob, alpha, verbose = 1):
    alphas = prob.get_alpha_seq(alpha)

    # computing the test error for each fitted matrix
    sol = prob.fit_and_eval(alphas)

    opt = np.argmin(sol.errs).item()
    est = sol.mats[opt]

    if (verbose==2):
        print(f'overfit alpha = {alphas[opt]}, smallest alpha = {alpha}')

    alpha_stat = [alphas[-1], alphas[opt], alphas[0]]

    return est, alpha_stat, sol

def regular_impute(prob, alpha: float):
    alphas = prob.get_alpha_seq(alpha)

    # computing the test error for each fitted matrix
    svd = prob.fit(alphas)[-1]
    est = svd.to_matrix()

    return est, svd


# miscellaneous
def subarray(arr, ind):
    return [arr[i] for i in ind]


def compute_op_norm_thresh(config, n_sample, level, repetition):
    norms = []
    for _ in range(repetition):
        rs = npr.randint(config.n_row, size=n_sample)
        cs = npr.randint(config.n_col, size=n_sample)
        ys = npr.randn(n_sample) * config.sd

        e = np.zeros((config.n_row, config.n_col))
        np.add.at(e, (rs, cs), ys)

        norms.append(npl.norm(e, 2))

    norms.sort()
    index = int(level * (repetition - 1))

    return norms[index]


def moments_to_confidence_band(m0, m1, m2):
    ys = m1 / m0
    vs = (m2 - m1 ** 2 / m0) / (m0 - 1)    
    #ss = (vs ** 0.5)
    ss = 2 * (vs ** 0.5) / np.sqrt(m0)  # Updated to reflect 2xSE

    return ys, ss


def plot_data(data, plt, alpha=0.2):
    for name, (xs, ys, ss) in data.items():
        plt.fill_between(xs, ys - ss, ys + ss, alpha=alpha)
        plt.plot(xs, ys, label=name)

    plt.xlabel('number of samples')
    plt.ylabel('relative error')
    plt.legend()
