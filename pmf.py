"""

Poisson matrix factorization with Batch inference

CREATED: 2014-03-25 02:06:52 by Dawen Liang <dliang@ee.columbia.edu>

"""
import logging
import numpy as np
from scipy import sparse, special, weave

from sklearn.base import BaseEstimator, TransformerMixin


class PoissonMF(BaseEstimator, TransformerMixin):
    ''' Poisson matrix factorization with batch inference '''
    def __init__(self, n_components=100, max_iter=100, tol=0.0001,
                 smoothness=100, random_state=None, verbose=False,
                 **kwargs):
        ''' Poisson matrix factorization

        Arguments
        ---------
        n_components : int
            Number of latent components

        max_iter : int
            Maximal number of iterations to perform

        tol : float
            The threshold on the increase of the objective to stop the
            iteration

        smoothness : int
            Smoothness on the initialization variational parameters

        random_state : int or RandomState
            Pseudo random number generator used for sampling

        verbose : bool
            Whether to show progress during model fitting

        **kwargs: dict
            Model hyperparameters
        '''
        self.logger = logging.getLogger(__name__)

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

        # # create logger
        # self.logger = logging.getLogger('pmf')
        # self.logger.setLevel(logging.DEBUG)
        # # create file handler which logs even debug messages
        # fh = logging.FileHandler(self.out_dir + 'pmf.log')
        # fh.setLevel(logging.DEBUG)
        # # create console handler with a higher log level
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.DEBUG)
        # # create formatter and add it to the handlers
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
        # fh.setFormatter(formatter)
        # ch.setFormatter(formatter)
        # # add the handlers to the logger
        # self.logger.addHandler(fh)
        # self.logger.addHandler(ch)
        # # example
        # #self.logger.info('test log creating an instance of auxiliary_module.Auxiliary')


    def _parse_args(self, **kwargs):
        self.a = float(kwargs.get('a', 0.1))
        self.b = float(kwargs.get('b', 0.1))
        self.c = float(kwargs.get('c', 0.1))
        self.d = float(kwargs.get('d', 0.1))

    def _init_users(self, n_users, theta=False):
        if type(theta) == np.ndarray:
            self.logger.info('initializing theta to be the observed one')
            self.Et = theta
            self.Elogt = None
            self.gamma_t = None
            self.rho_t = None
        else:
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

    def _init_items(self, n_items, beta=False, categorywise=False):
        # if we pass in observed betas:
        if type(beta) == np.ndarray and not categorywise:
            self.logger.info('initializing beta to be the observed one')
            self.Eb = beta
            self.Elogb = None
            self.gamma_b = None
            self.rho_b = None
        else: # proceed normally
            self.logger.info('initializing normal variational params')
            # variational parameters for beta
            self.gamma_b = self.smoothness * \
                np.random.gamma(self.smoothness, 1. / self.smoothness,
                                size=(n_items, self.n_components)
                                ).astype(np.float32)
            self.rho_b = self.smoothness * \
                np.random.gamma(self.smoothness, 1. / self.smoothness,
                                size=(n_items, self.n_components)
                                ).astype(np.float32)
            self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def fit(self, X, rows, cols, vad,
        beta=False, theta=False, categorywise=False):
        '''Fit the model to the data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)
            Training data.

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        n_items, n_users = X.shape
        self._init_items(n_items, beta=beta, categorywise=categorywise)
        self._init_users(n_users, theta=theta)
        self._update(X, rows, cols, vad, beta=beta, categorywise=categorywise)
        return self

    #def transform(self, X, attr=None):
    #    '''Encode the data as a linear combination of the latent components.

    #    Parameters
    #    ----------
    #    X : array-like, shape (n_samples, n_feats)

    #    attr: string
    #        The name of attribute, default 'Eb'. Can be changed to Elogb to
    #        obtain E_q[log beta] as transformed data.

    #    Returns
    #    -------
    #    X_new : array-like, shape(n_samples, n_filters)
    #        Transformed data, as specified by attr.
    #    '''

    #    if not hasattr(self, 'Eb'):
    #        raise ValueError('There are no pre-trained components.')
    #    n_samples, n_feats = X.shape
    #    if n_feats != self.Eb.shape[1]:
    #        raise ValueError('The dimension of the transformed data '
    #                         'does not match with the existing components.')
    #    if attr is None:
    #        attr = 'Et'
    #    self._init_weights(n_samples)
    #    self._update(X, update_beta=False)
    #    return getattr(self, attr)

    def _update(self, X, rows, cols, vad, beta=False, categorywise=False):
        # alternating between update latent components and weights
        old_pll = -np.inf
        for i in xrange(self.max_iter):
            self._update_users(X, rows, cols, beta=beta)
            if type(beta) == np.ndarray and not categorywise:
                # do nothing if we have observed betas.
                pass
            elif type(beta) == np.ndarray and categorywise:
                self._update_items(X, rows, cols, beta=beta,
                    categorywise=categorywise, iteration=i)
            else:
                self._update_items(X, rows, cols)
            pred_ll = self.pred_loglikeli(**vad)
            if np.isnan(pred_ll):
                self.logger.error('got nan in predictive ll')
                raise Exception('nan in predictive ll')
            improvement = (pred_ll - old_pll) / abs(old_pll)
            if self.verbose:
                string = 'ITERATION: %d\tPred_ll: %.2f\tOld Pred_ll: %.2f\t Improvement: %.5f' % (i, pred_ll, old_pll, improvement)
                self.logger.info(string)
            if improvement < self.tol:
                break
            old_pll = pred_ll
        pass

    def _update_users(self, X, rows, cols, beta=False):
        xexplog = self._xexplog(rows, cols, beta=beta)
        ratioT = sparse.csr_matrix(( X.data / xexplog,
                                    (rows, cols)),
                                   dtype=np.float32, shape=X.shape).transpose()
        if type(beta) == np.ndarray:
            # for n in range(0, self.Eb.shape[0]+1):
            #     dot = ratioT[0,0:n].dot(self.Eb[0:n,0])
            #     if np.isnan(dot):
            #         print 'got nan at'
            #         print n
            # for n in range(ratioT[0].shape[0]+1):
            #     prod = ratioT[0,n] * (self.Eb[n,0])
            #     if np.isnan(prod):
            #         print n
            #     if n % 10000 == 0:
            #         print 'iter'
            #         print n
            # print 'ratioT[0,:] dot Eb[:,0] at 157141'
            # print ratioT[0,157141] # = inf!
            # print self.Eb[157141, 0]
            # print ratioT[0,157141] * self.Eb[157141, 0]
            #dotprod = 0
            # for n, el in enumerate(self.Eb[:,0]):
            #     prod = el * ratioT[0, n]
            #     dotprod += prod
            #     if np.isnan(dotprod):
            #         print 'got dot product nan at:' + str(n)
            #     if np.isnan(prod):
            #         print 'got prod nan at ' + str(n)
            #print ratioT[0,:].dot(self.Eb[:,0]) #was =nan with dropped docs!
            # we don't have Elogb and don't need it.
            # add trick for logsumexp overflow prevention
            # self.gamma_t = self.a + np.exp(self.Elogt - self.Elogt.max()) * \
            #     ratioT.dot(self.Eb).T
            self.gamma_t = self.a + np.exp(self.Elogt) * \
                ratioT.dot(self.Eb).T
        else:
            self.gamma_t = self.a + np.exp(self.Elogt) * \
                ratioT.dot(np.exp(self.Elogb)).T
        self.rho_t = self.b + np.sum(self.Eb, axis=0, keepdims=True).T
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)

    def _update_items(self, X, rows, cols, beta=False, categorywise=False,
        iteration=None):
        ratio = sparse.csr_matrix((X.data / self._xexplog(rows, cols),
                                   (rows, cols)),
                                  dtype=np.float32, shape=X.shape)
        if type(beta) == np.ndarray and categorywise:
            beta_bool = beta.astype(bool)
            gamma_b_updated = self.c + np.exp(self.Elogb) * \
                ratio.dot(np.exp(self.Elogt.T))
            rho_b_updated = self.d + np.sum(self.Et, axis=1)
            rho_b_updated_reshaped = np.reshape(np.repeat(rho_b_updated,
                self.rho_b.shape[0], axis=0), self.rho_b.shape)
            if iteration % 2 == 0:
                self.logger.info('updating *only* in-category parameters')
                self.gamma_b[beta_bool] = gamma_b_updated[beta_bool]
                self.rho_b[beta_bool] = rho_b_updated_reshaped[beta_bool]
            else:
                beta_bool_not = np.logical_not(beta_bool)
                self.logger.info('updating *only* out-category parameters')
                self.gamma_b[beta_bool_not] = gamma_b_updated[beta_bool_not]
                self.rho_b[beta_bool_not] = \
                    rho_b_updated_reshaped[beta_bool_not]
        else:
            self.gamma_b = self.c + np.exp(self.Elogb) * \
                ratio.dot(np.exp(self.Elogt.T))
            self.rho_b = self.d + np.sum(self.Et, axis=1)
            print self.rho_b.shape
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def _xexplog(self, rows, cols, beta=False):
        '''
        sum_k exp(E[log theta_{ik} * beta_{kd}])
        '''
        if type(beta) == np.ndarray:
            # add trick for log sum exp overflow prevention
            #data = _inner(self.Eb, np.exp(self.Elogt - self.Elogt.max()), rows, cols)
            data = _inner(self.Eb, np.exp(self.Elogt), rows, cols)
        else:
            data = _inner(np.exp(self.Elogb), np.exp(self.Elogt), rows, cols)
        return data

    def pred_loglikeli(self, X_new, rows_new, cols_new):
        #print self.Eb[0]
        #print self.Et[:,0]
        X_pred = _inner(self.Eb, self.Et, rows_new, cols_new)
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
