"""

User-Artist-Song Poisson matrix factorization with Batch inference

CREATED: 2014-03-25 02:06:52 by Dawen Liang <dliang@ee.columbia.edu>

MODIFIED: 2015-03-25 13:06:12 by Jaan Altosaar <altosaar@princeton.edu>

"""

import sys
import numpy as np
from scipy import sparse, special, weave
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class PoissonMF(BaseEstimator, TransformerMixin):
    ''' Poisson matrix factorization with batch inference '''
    def __init__(self, n_components=100, max_iter=100, tol=0.0001,
                 smoothness=100, random_state=None, verbose=False,
                 **kwargs):

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.a = float(kwargs.get('a', 0.1))
        self.b = float(kwargs.get('b', 0.1))
        self.c = float(kwargs.get('c', 0.1))
        self.d = float(kwargs.get('d', 0.1))
        self.f = float(kwargs.get('f', 0.1))
        self.g = float(kwargs.get('g', 0.1))
        self.song2artist = np.array(kwargs.get('s2a', None))
        self.artist2songs = dict()
        self.n_artists = len(np.unique(self.song2artist))
        for artist in range(0,self.n_artists):
            self.artist2songs[artist]=np.where(self.song2artist==artist)[0]
        self.n_songs_by_artist = np.reshape(np.array([self.artist2songs[artist].size for artist in range(self.n_artists)]).astype(np.float32),(self.n_artists,1))
        self.artist_indicator = pd.get_dummies(self.song2artist).T
        self.artist_indicator = sparse.csr_matrix(self.artist_indicator.values)

    def _init_users(self, n_users):
        # variational parameters for theta
        self.gamma_t = self.smoothness * \
            np.random.gamma(self.smoothness, 1. / self.smoothness,
                            size=(self.n_components, n_users)
                            ).astype(np.float32)
        self.rho_t = self.smoothness * \
            np.random.gamma(self.smoothness, 1. / self.smoothness,
                            size=(self.n_components, n_users)
                            ).astype(np.float32)
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)

    def _init_items(self, n_items):
        # variational parameters for beta_songs (beta_s)
        self.gamma_bs = 0.01 * self.smoothness * \
            np.random.gamma(self.smoothness, 1. / self.smoothness,
                            size=(n_items, self.n_components)
                            ).astype(np.float32)
        self.rho_bs = 0.01 * self.smoothness * \
            np.random.gamma(self.smoothness, 1. / self.smoothness,
                            size=(n_items, self.n_components)
                            ).astype(np.float32)
        self.Ebs, self.Elogbs = _compute_expectations(self.gamma_bs, self.rho_bs)

    def _init_artists(self, n_artists):
        # variational parameters for beta_artist (beta_a or beta_a(s))
        self.gamma_ba = self.smoothness * \
            np.random.gamma(self.smoothness, 1. / self.smoothness,
                            size=(n_artists, self.n_components)
                            ).astype(np.float32)
        self.rho_ba = self.smoothness * \
            np.random.gamma(self.smoothness, 1. / self.smoothness,
                            size=(n_artists, self.n_components)
                            ).astype(np.float32)
        self.Eba, self.Elogba = _compute_expectations(self.gamma_ba, self.rho_ba)

    def fit(self, X, rows, cols, vad):
        '''Fit the model to the data in X.

        Parameters
        ----------
        X : array-like, shape (n_songs, n_users)
            Training data.

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        n_items, n_users = X.shape
        self._init_items(n_items)
        self._init_users(n_users)
        self._init_artists(self.n_artists)
        self._update(X, rows, cols, vad)
        return self

    def _update(self, X, rows, cols, vad):
        # alternating between update latent components and weights
        old_pll = -np.inf
        for i in xrange(self.max_iter):
            self._update_users(X, rows, cols)
            self._update_items(X, rows, cols)
            self._update_artists(X,rows,cols)
            pred_ll = self.pred_loglikeli(**vad)
            improvement = (pred_ll - old_pll) / abs(old_pll)
            if self.verbose:
                txt = 'ITERATION: %d\tPred_ll: %.2f\tOld Pred_ll: %.2f\tImprovement: %.5f' % (i, pred_ll, old_pll, improvement)
                print(txt)
                #sys.stdout.flush()
            if improvement < self.tol:
                break
            old_pll = pred_ll
        pass

    def _update_users(self, X, rows, cols):
        ratioTs = sparse.csr_matrix((X.data / self._xexplog_bs(rows, cols),
                                    (rows, cols)),
                                   dtype=np.float32, shape=X.shape).transpose()
        ratioTa = sparse.csr_matrix((X.data / self._xexplog_ba(rows, cols),
                                    (rows, cols)),
                                   dtype=np.float32, shape=X.shape).transpose()
        self.gamma_t = self.a + np.exp(self.Elogt) * \
            ratioTs.dot(np.exp(self.Elogbs)).T + \
            np.exp(self.Elogt) * \
            ratioTa.dot(np.exp(self.Elogba)[self.song2artist]).T
        self.rho_t = self.b + np.sum(self.Eba[self.song2artist], axis=0, keepdims=True).T + np.sum(self.Ebs, axis=0, keepdims=True).T
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)

    def _update_items(self, X, rows, cols):
        ratio = sparse.csr_matrix((X.data / self._xexplog_bs(rows, cols),
                                   (rows, cols)),
                                  dtype=np.float32, shape=X.shape)
        self.gamma_bs = self.f + np.exp(self.Elogbs) * \
            ratio.dot(np.exp(self.Elogt.T))
        self.rho_bs = self.g + np.sum(self.Et, axis=1)
        self.Ebs, self.Elogbs = _compute_expectations(self.gamma_bs, self.rho_bs)

    def _update_artists(self, X, rows, cols):
        ratio = sparse.csr_matrix((X.data / self._xexplog_ba(rows, cols),
                                   (rows, cols)),
                                  dtype=np.float32, shape=X.shape)

        summed_over_artists = self.artist_indicator.dot(np.exp(self.Elogba[self.song2artist]) * \
            ratio.dot(np.exp(self.Elogt.T)))

        self.gamma_ba = self.c + summed_over_artists
        self.rho_ba = self.d + self.n_songs_by_artist * np.sum(self.Et, axis=1)
        self.Eba, self.Elogba = _compute_expectations(self.gamma_ba, self.rho_ba)


    def _xexplog_bs(self, rows, cols):
        '''
        sum_k exp(E[log theta_{ik} * beta_s_{kd}])
        '''
        data = _inner(np.exp(self.Elogbs), np.exp(self.Elogt), rows, cols)
        return data

    def _xexplog_ba(self, rows, cols):
        '''
        user i, artist a(s) = s_num2a_num[cols]
        sum_k exp(E[log theta_{ik} * beta_a_{ka}])
        '''
        rows_artists = np.array([self.song2artist[song] for song in rows], dtype=np.int32)
        data = _inner(np.exp(self.Elogba), np.exp(self.Elogt), rows_artists, cols)
        return data

    def pred_loglikeli(self, X_new, rows_new, cols_new):
        X_pred_bs = _inner(self.Ebs, self.Et, rows_new, cols_new)
        rows_artists_new = np.array([self.song2artist[song] for song in rows_new], dtype=np.int32)
        X_pred_ba = _inner(self.Eba, self.Et, rows_artists_new, cols_new)
        X_pred = X_pred_bs + X_pred_ba
        pred_ll = np.mean(X_new * np.log(X_pred) - X_pred)
        return pred_ll

def _inner(beta, theta, rows, cols):
    n_ratings = rows.size
    n_components, n_users = theta.shape
    data = np.empty(n_ratings, dtype=np.float32)
    code = r"""
    for (int i = 0; i < n_ratings; i++) {
       data[i] = 0.0;
       for (int j = 0; j < n_components; j++) {
           data[i] += beta[rows[i] * n_components + j] * theta[j * n_users + cols[i]];
       }
    }
    """
    weave.inline(code, ['data', 'theta', 'beta', 'rows', 'cols',
                        'n_ratings', 'n_components', 'n_users'])
    return data

def _compute_expectations(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute E[x] and E[log x]
    '''
    return (alpha / beta, special.psi(alpha) - np.log(beta))