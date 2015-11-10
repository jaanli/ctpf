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
import logging

class PoissonMF(BaseEstimator, TransformerMixin):
    ''' Poisson matrix factorization with batch inference '''
    def __init__(self, n_components=100, max_iter=100, min_iter=1, tol=0.0001,
                 smoothness=100, random_state=None, verbose=False,
                 **kwargs):

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.min_iter = min_iter
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose
        self.max_iter_fixed = 4

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_args(**kwargs)
        self.logger = logging.getLogger(__name__)


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
            #self.artist2songs[artist]=np.where(self.song2artist==artist)[0]
            self.artist2songs[artist]=np.array([artist])
        self.n_songs_by_artist = np.reshape(np.array([self.artist2songs[artist].size for artist in range(self.n_artists)]).astype(np.float32),(self.n_artists,1))
        #self.artist_indicator = pd.get_dummies(self.song2artist).T
        #self.artist_indicator = sparse.csr_matrix(self.artist_indicator.values)
        self.artist_indicator = sparse.identity(self.n_artists, format='csr')

    def _init_users(self, n_users, theta=False):
        # if we pass in observed thetas:
        if type(theta) == np.ndarray:
            self.logger.info('initializing theta to be the observed one')
            self.Et = theta
            self.Elogt = None
            self.gamma_t = None
            self.rho_t = None
        else: # proceed normally
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

    def _init_items(self, n_items, beta=None):
        # if we pass in observed betas:
        if type(beta) == np.ndarray:
            self.logger.info('initializing beta to be the observed one')
            self.Ebs = beta
            self.Elogbs = None
            self.gamma_bs = None
            self.rho_bs = None
        else: # proceed normally
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

    def fit(self, X, rows, cols, vad, beta=False, theta=False,
        categorywise=False,
        item_fit_type='default',
        user_fit_type='default',
        zero_untrained_components=False):
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
        self.n_users = n_users
        self._init_items(n_items, beta=beta)
        self._init_users(n_users, theta=theta)
        self._init_artists(self.n_artists)
        if user_fit_type != 'default':
            best_validation_ll = -np.inf
            for switch_idx in xrange(self.max_iter_fixed):
                if user_fit_type == 'converge_separately':
                    if switch_idx % 2 == 0:
                        only_update = 'items'
                    else:
                        only_update = 'users'

                    if switch_idx == 1:
                        self.logger.info('set init users')
                        initialize_users = 'default'
                    else:
                        initialize_users = 'none'
                    self.logger.info('=> only updating {}, switch number {}'
                        .format(only_update, switch_idx))
                    validation_ll, best_pll_dict = self._update(X, rows, cols, vad, beta=beta,
                        theta=theta,
                        categorywise=categorywise,
                        user_fit_type=user_fit_type,
                        item_fit_type=item_fit_type,
                        initialize_users = initialize_users,
                        zero_untrained_components=zero_untrained_components,
                        observed_user_preferences=True,
                        only_update=only_update)
                    new_validation_ll = best_pll_dict['pred_ll']
                    self.logger.info('set params to best pll {}, old one was {}'
                        .format(new_validation_ll, validation_ll))
                    validation_ll = new_validation_ll
                    self.Eba = best_pll_dict['best_Eba']
                    self.Ebs = best_pll_dict['best_Ebs']
                    self.Et = best_pll_dict['best_Et']
                    self.Elogba = best_pll_dict['best_Elogba']
                    self.Elogbs = best_pll_dict['best_Elogbs']
                    self.Elogt = best_pll_dict['best_Elogt']
                    if validation_ll > best_validation_ll:
                        best_Eba = self.Eba
                        best_Ebs = self.Ebs
                        # print best_Eb
                        # print '^Eb'
                        best_Et = self.Et
                        # print best_Et
                        # print '^Et'
                        # best_self = self
                        best_validation_ll = validation_ll
                self.logger.info('best validation ll was {}'.format(best_validation_ll))
                self.Eba = best_Eba
                self.Ebs = best_Ebs
                self.Et = best_Et
        else:
            self._update(X, rows, cols, vad, beta=beta, theta=theta,
            categorywise=categorywise,
            item_fit_type=item_fit_type,
            zero_untrained_components=zero_untrained_components)
        return self

    def _update(self, X, rows, cols, vad, beta=False, theta=False,
        categorywise=False,
        item_fit_type='default',
        user_fit_type='default',
        zero_untrained_components=False,
        initialize_users = 'none',
        only_update=None,
        observed_user_preferences=False,
        update='default'):
        # alternating between update latent components and weights
        old_pll = -np.inf
        best_pll_dict = dict(pred_ll = -np.inf)
        for i in xrange(self.max_iter):
            if (only_update == 'items' or
                observed_user_preferences and
                update != 'default'):
                pass
            elif (only_update == 'users'):
                if initialize_users == 'default':
                    if i == 0:
                        self.logger.info('init default users')
                        self._init_users(self.n_users)
                    self._update_users(X, rows, cols, beta=beta,
                        observed_user_preferences=False,
                        only_update=only_update)
                elif initialize_users == 'none':
                    self._update_users(X, rows, cols, beta=beta,
                        observed_user_preferences=False,
                        only_update=only_update)
            else:
                self._update_users(X, rows, cols)

            if type(beta) == np.ndarray:
                pass
            else:
                self._update_items(X, rows, cols)

            if item_fit_type != 'default':
                if zero_untrained_components and i == 0 and update == 'default':
                    # store the initial values somewhere, then zero them out,
                    # then load them back in once they've been fit
                    beta_bool = beta.astype(bool)
                    beta_bool_not = np.logical_not(beta_bool)
                    small_num = 1e-5
                    if item_fit_type == 'converge_in_category_first':
                        # zero out out_category components
                        gamma_ba_out_category = self.gamma_ba[beta_bool_not]
                        rho_ba_out_category = self.rho_ba[beta_bool_not]
                        self.gamma_ba[beta_bool_not] = small_num
                        self.rho_ba[beta_bool_not] = small_num
                    elif item_fit_type == 'converge_out_category_first':
                        # zero out in_category components
                        gamma_ba_in_category = self.gamma_ba[beta_bool]
                        rho_ba_in_category = self.rho_ba[beta_bool]
                        self.gamma_ba[beta_bool] = small_num
                        self.rho_ba[beta_bool] = small_num

            if not only_update == 'users':
                if (type(beta) == np.ndarray and categorywise and
                    item_fit_type == 'converge_in_category_first'):

                    if update == 'default':
                        self._update_artists(X,rows,cols, theta=theta, beta=beta,
                            categorywise=categorywise, update='in_category')
                    else:
                        self._update_artists(X,rows,cols, theta=theta, beta=beta,
                            categorywise=categorywise, update=update)
                elif (type(beta) == np.ndarray and categorywise and
                    item_fit_type == 'converge_out_category_first'):

                    if update == 'default':

                        self._update_artists(X,rows,cols, theta=theta, beta=beta,
                            categorywise=categorywise, update='out_category')
                    else:
                        self._update_artists(X,rows,cols, theta=theta, beta=beta,
                            categorywise=categorywise, update=update)
                else:
                    self._update_artists(X,rows,cols, theta=theta)

            pred_ll = self.pred_loglikeli(**vad)
            if np.isnan(pred_ll):
                self.logger.error('got nan in predictive ll')
                raise Exception('nan in predictive ll')
            else:
                if pred_ll > best_pll_dict['pred_ll']:
                    best_pll_dict['pred_ll'] = pred_ll
                    self.logger.info('logged new best pred_ll as {}'
                        .format(pred_ll))
                    best_pll_dict['best_Eba'] = self.Eba
                    best_pll_dict['best_Elogba'] = self.Elogba
                    best_pll_dict['best_Ebs'] = self.Ebs
                    best_pll_dict['best_Elogbs'] = self.Elogbs
                    best_pll_dict['best_Et'] = self.Et
                    best_pll_dict['best_Elogt'] = self.Elogt
            improvement = (pred_ll - old_pll) / abs(old_pll)
            if self.verbose:
                string = 'ITERATION: %d\tPred_ll: %.2f\tOld Pred_ll: %.2f\tImprovement: %.5f' % (i, pred_ll, old_pll, improvement)
                self.logger.info(string)
                #sys.stdout.flush()
            if improvement < self.tol and i > self.min_iter:
                if update == 'default' and item_fit_type != 'default':
                    if item_fit_type == 'converge_in_category_first':
                        # we converged in-category. now converge out_category
                        if zero_untrained_components:
                            self.logger.info(
                                're-load initial values for out_category')
                            self.gamma_ba[beta_bool_not] = gamma_ba_out_category
                            self.rho_ba[beta_bool_not] = rho_ba_out_category
                        self._update(X, rows, cols, vad, beta=beta, theta=theta,
                            categorywise=categorywise,
                            observed_user_preferences=observed_user_preferences,
                            item_fit_type=item_fit_type,
                            update='out_category')
                    if item_fit_type == 'converge_out_category_first':
                        # we converged out-category. now converge in_category
                        if zero_untrained_components:
                            self.logger.info(
                                're-load initial values for in_category')
                            self.gamma_ba[beta_bool] = gamma_ba_in_category
                            self.rho_ba[beta_bool] = rho_ba_in_category
                        self._update(X, rows, cols, vad, beta=beta, theta=theta,
                            categorywise=categorywise, item_fit_type=item_fit_type,
                            update='in_category')
                break
            old_pll = pred_ll
        #pass
        return pred_ll, best_pll_dict

    def _update_users(self, X, rows, cols, beta=False, theta=False, observed_user_preferences=False,
        observed_item_attributes=False,
        only_update=False):

        self.logger.info('updating users')

        xexplog_bs = self._xexplog_bs(rows, cols, beta=beta)

        if only_update == 'users':
            expElogbs = self.Ebs
        else:
            expElogbs = np.exp(self.Elogbs)


        ratioTs = sparse.csr_matrix((X.data / xexplog_bs,
                                    (rows, cols)),
                                   dtype=np.float32, shape=X.shape).transpose()
        ratioTa = sparse.csr_matrix((X.data / self._xexplog_ba(rows, cols),
                                    (rows, cols)),
                                   dtype=np.float32, shape=X.shape).transpose()
        self.gamma_t = self.a + np.exp(self.Elogt) * \
            ratioTs.dot(expElogbs).T + \
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

    def _update_artists(self, X, rows, cols, theta=False, beta=False,
        categorywise=False,
        update='default'):

        self.logger.info('updating epsilons / artists')

        ratio = sparse.csr_matrix((X.data / self._xexplog_ba(rows, cols, theta=theta),
                                   (rows, cols)),
                                  dtype=np.float32, shape=X.shape)

        if type(theta) == np.ndarray:
            expElogt = self.Et.T
        else:
            expElogt = np.exp(self.Elogt.T)

        if (type(beta) == np.ndarray and
                categorywise and
                update != 'default'):

            beta_bool = beta.astype(bool)

            summed_over_artists = self.artist_indicator.dot(np.exp(self.Elogba[self.song2artist]) * \
                ratio.dot(expElogt))

            gamma_ba_updated = self.c + summed_over_artists
            rho_ba_updated = self.d + self.n_songs_by_artist * np.sum(self.Et, axis=1)

            if update == 'in_category':
                    self.logger.info('updating *only* in-category parameters')
                    self.gamma_ba[beta_bool] = gamma_ba_updated[beta_bool]
                    self.rho_ba[beta_bool] = rho_ba_updated[beta_bool]
            elif update == 'out_category':
                    beta_bool_not = np.logical_not(beta_bool)
                    self.logger.info('updating *only* out-category parameters')
                    self.gamma_ba[beta_bool_not] = gamma_ba_updated[beta_bool_not]
                    self.rho_ba[beta_bool_not] = \
                        rho_ba_updated[beta_bool_not]
        else:

            summed_over_artists = self.artist_indicator.dot(np.exp(self.Elogba[self.song2artist]) * \
                ratio.dot(expElogt))

            self.gamma_ba = self.c + summed_over_artists
            self.rho_ba = self.d + self.n_songs_by_artist * np.sum(self.Et, axis=1)

        self.Eba, self.Elogba = _compute_expectations(self.gamma_ba, self.rho_ba)


    def _xexplog_bs(self, rows, cols, theta=None, beta=None):
        '''
        sum_k exp(E[log theta_{ik} * beta_s_{kd}])
        '''
        if type(beta) == np.ndarray and type(theta) == np.ndarray:
            data = _inner(self.Ebs, self.Et, rows, cols)
        elif type(beta) == np.ndarray:
            data = _inner(self.Ebs, np.exp(self.Elogt), rows, cols)
        else:
            data = _inner(np.exp(self.Elogbs), np.exp(self.Elogt), rows, cols)
        return data

    def _xexplog_ba(self, rows, cols, theta=False):
        '''
        user i, artist a(s) = s_num2a_num[cols]
        sum_k exp(E[log theta_{ik} * beta_a_{ka}])
        '''
        if type(theta) == np.ndarray:
            rows_artists = np.array([song for song in rows], dtype=np.int32)
            data = _inner(np.exp(self.Elogba), self.Et, rows_artists, cols)
        else:
            rows_artists = np.array([self.song2artist[song] for song in rows], dtype=np.int32)
            data = _inner(np.exp(self.Elogba), np.exp(self.Elogt), rows_artists, cols)
        return data

    def pred_loglikeli(self, X_new, rows_new, cols_new):
        X_pred_bs = _inner(self.Ebs, self.Et, rows_new, cols_new)
        #rows_artists_new = np.array([self.song2artist[song] for song in rows_new], dtype=np.int32)
        rows_artists_new = np.array([song for song in rows_new], dtype=np.int32)
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