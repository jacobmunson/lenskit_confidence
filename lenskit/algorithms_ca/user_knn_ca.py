"""
User-based k-NN collaborative filtering.
"""

from sys import intern
import logging

import pandas as pd
import numpy as np

#import math # for nan in my edits

from numba import njit

from .. import util, matrix
from . import Predictor
from ..util.accum import kvp_minheap_insert

from astropy.stats import jackknife_resampling
from astropy.stats import jackknife_stats

_logger = logging.getLogger(__name__)


@njit
def _agg_weighted_avg(iur, item, sims, use):
    """
    Weighted-average aggregate.

    Args:
        iur(matrix._CSR): the item-user ratings matrix
        item(int): the item index in ``iur``
        sims(numpy.ndarray): the similarities for the users who have rated ``item``
        use(numpy.ndarray): positions in sims and the rating row to actually use
    """
    rates = iur.row_vs(item)
    num = 0.0
    den = 0.0
    for j in use:
        num += rates[j] * sims[j]
        den += np.abs(sims[j])
    return num / den

@njit
def _agg_avg(iur, item, sims, use):
    """
    Average aggregate.

    Args:
        iur(matrix._CSR): the item-user ratings matrix
        item(int): the item index in ``iur``
        use(numpy.ndarray): positions in sims and the rating row to actually use
    """

    rates = iur.row_vs(item)
    num = 0.0
    den = len(use) 
    for j in use:
        num += rates[j]
    return num /den # NOTE THAT PREDICTION AT THE END WILL NOT BE EXACTLY THE AVERAGE, SINCE RATINGS ARE NORMALIZED AND OFFSET BY USER AVERAGE


@njit
def _agg_sum(iur, item, sims, use):
    """
    Sum aggregate

    Args:
        iur(matrix._CSR): the item-user ratings matrix
        item(int): the item index in ``iur``
        sims(numpy.ndarray): the similarities for the users who have rated ``item``
        use(numpy.ndarray): positions in sims and the rating row to actually use
    """
    x = 0.0
    for j in use:
        x += sims[j]
    return x

@njit
def _rating_avg_std(iur, item, sims, use): # STANDARD DEVIATION FOR AVERAGE
    """
    Sum aggregate

    Args:
        iur(matrix._CSR): the item-user ratings matrix
        item(int): the item index in ``iur``
        sims(numpy.ndarray): the similarities for the users who have rated ``item``
        use(numpy.ndarray): positions in sims and the rating row to actually use
    """
    co_ratings = iur.row_vs(item)
    coratings = co_ratings.copy()
    if len(use) > 1: # ratings
        sd = 0.0
        mu = 0.0

        for k in use:
            mu += co_ratings[k]
        mu = mu / len(use) # ratings
        
        for m in use:
            coratings[m] = (co_ratings[m] - mu)**2
        sd = np.sqrt(np.sum(coratings[use]) / (len(use) - 1)) # sd now   # actually variance
    else:
        sd = 0
    return sd


@njit
def _rating_weighted_avg_std(iur, item, sims, use): # STANDARD DEVIATION FOR WEIGHTED-AVERAGE
    """
    Sum aggregate

    Args:
        iur(matrix._CSR): the item-user ratings matrix
        item(int): the item index in ``iur``
        sims(numpy.ndarray): the similarities for the users who have rated ``item``
        use(numpy.ndarray): positions in sims and the rating row to actually use
    """
    co_ratings = iur.row_vs(item)
    coratings = co_ratings.copy()
    if len(use) > 1: # ratings
        sd = 0.0
        mu = 0.0

        for k in use:
            mu += co_ratings[k]*sims[k]
        mu = mu / len(use) # ratings
        
        for m in use:
            coratings[m] = (co_ratings[m]*sims[k] - mu)**2
        sd = np.sqrt(np.sum(coratings[use]) / (len(use) - 1)) # sd now   # actually variance
    else:
        sd = 0
    return sd



###

@njit
def _rating_avg_jk_std(iur, item, sims, use): # JACKKNIFE ESTIMATE OF STANDARD DEVIATION FOR AVERAGE
    """
    Sum aggregate

    Args:
        iur(matrix._CSR): the item-user ratings matrix
        item(int): the item index in ``iur``
        sims(numpy.ndarray): the similarities for the users who have rated ``item``
        use(numpy.ndarray): positions in sims and the rating row to actually use
    """
    co_ratings = iur.row_vs(item)
    #coratings = co_ratings.copy()
    #test_statistic = np.std
    if len(use) > 1: # ratings
        
        std_values, n = [], len(use)
        index = np.arange(n)
        used_data = co_ratings[use]

        for i in range(n):
            jk_sample = np.std(used_data[index != i])
            std_values.append(jk_sample)
        
        std_values_jk = np.mean(np.array(std_values))
        sd = np.std(used_data) - (n - 1)*(std_values_jk - np.std(used_data)) # bias-corrected estimate
    else:
        sd = 0
    return sd

@njit
def _rating_weighted_avg_jk_std(iur, item, sims, use): # JACKKNIFE ESTIMATE OF STANDARD DEVIATION FOR WEIGHTED-AVERAGE
    """
    Sum aggregate

    Args:
        iur(matrix._CSR): the item-user ratings matrix
        item(int): the item index in ``iur``
        sims(numpy.ndarray): the similarities for the users who have rated ``item``
        use(numpy.ndarray): positions in sims and the rating row to actually use
    """
    co_ratings = iur.row_vs(item)
    #coratings = co_ratings.copy()
    #test_statistic = np.std
    if len(use) > 1: # ratings
        
        std_values, n = [], len(use)
        index = np.arange(n)
        used_data = co_ratings[use]*sims[use]

        for i in range(n):
            jk_sample = np.std(used_data[index != i])
            std_values.append(jk_sample)
        
        std_values_jk = np.mean(np.array(std_values))
        sd = np.std(used_data) - (n - 1)*(std_values_jk - np.std(used_data)) # bias-corrected estimate
    else:
        sd = 0
    return sd




@njit
def _rating_weighted_avg_bs_std(iur, item, sims, use): # BOOTSTRAP ESTIMATE OF STANDARD DEVIATION FOR WEIGHTED-AVERAGE
    """
    Sum aggregate

    Args:
        iur(matrix._CSR): the item-user ratings matrix
        item(int): the item index in ``iur``
        sims(numpy.ndarray): the similarities for the users who have rated ``item``
        use(numpy.ndarray): positions in sims and the rating row to actually use
    """
    co_ratings = iur.row_vs(item)
    #coratings = co_ratings.copy()
    #test_statistic = np.std
    if len(use) > 1: # ratings
        
        #std_values, n = [], len(use)
        #index = np.arange(n)
        used_data = co_ratings[use]*sims[use]
        sample_std = []

        for _ in range(100): # 1000 bootstrap samples
            sample_n = np.random.choice(used_data, size = len(used_data))
            sample_std.append(sample_n.std())
        sd = np.mean(np.array(sample_std))
    else:
        sd = 0
    return sd


@njit
def _rating_avg_bs_std(iur, item, sims, use): # BOOTSTRAP ESTIMATE OF STANDARD DEVIATION FOR AVERAGE
    """
    Sum aggregate

    Args:
        iur(matrix._CSR): the item-user ratings matrix
        item(int): the item index in ``iur``
        sims(numpy.ndarray): the similarities for the users who have rated ``item``
        use(numpy.ndarray): positions in sims and the rating row to actually use
    """
    co_ratings = iur.row_vs(item)
    #coratings = co_ratings.copy()
    #test_statistic = np.std
    if len(use) > 1: # ratings
        
        #std_values, n = [], len(use)
        #index = np.arange(n)
        used_data = co_ratings[use]
        sample_std = []

        for _ in range(100): # 1000 bootstrap samples
            sample_n = np.random.choice(used_data, size = len(used_data))
            sample_std.append(sample_n.std())
        sd = np.mean(np.array(sample_std))
    else:
        sd = 0
    return sd



###

@njit
def _score(items, results, iur, sims, nnbrs, min_sim, min_nbrs, agg, nbhr_ratings, num_nbhr_actual, var): # TYPE OF AGGREGATION IS FED INTO HERE

    h_ks = np.empty(nnbrs, dtype=np.int32)
    h_vs = np.empty(nnbrs)
    used = np.zeros(len(results), dtype=np.int32)

    
    for i in range(len(results)):
        item = items[i] # item is a specific item
        if item < 0:
            continue

        h_ep = 0

        # who has rated this item? CORATERS
        i_users = iur.row_cs(item)

        # what are their similarities to our target user? CORATER SIMILARITY
        i_sims = sims[i_users]

        # which of these neighbors do we really want to use?
        #print('selecting neighbors similar to target user')
        for j, s in enumerate(i_sims):
            if np.abs(s) < 1.0e-10:
                continue
            if min_sim is not None and s < min_sim:
                continue
            h_ep = kvp_minheap_insert(0, h_ep, nnbrs, j, s, h_ks, h_vs)

        if h_ep < min_nbrs:
            continue
        
        results[i] = agg(iur = iur, item = item, sims = i_sims, use = h_ks[:h_ep]) # AGGREGATE CALL IS HERE     
        nbhr_ratings[i] = var(iur = iur, item = item, sims = i_sims, use = h_ks[:h_ep])   
        num_nbhr_actual[i] = len(h_ks[:h_ep])
        used[i] = h_ep

        
    return used


class UserUserCA(Predictor):
    """
    User-user nearest-neighbor collaborative filtering with ratings. This user-user implementation
    is not terribly configurable; it hard-codes design decisions found to work well in the previous
    Java-based LensKit code.

    Args:
        nnbrs(int):
            the maximum number of neighbors for scoring each item (``None`` for unlimited)
        min_nbrs(int): the minimum number of neighbors for scoring each item
        min_sim(double): minimum similarity threshold for considering a neighbor
        center(bool):
            whether to normalize (mean-center) rating vectors.  Turn this off when working
            with unary data and other data types that don't respond well to centering.
        aggregate:
            the type of aggregation to do. Can be ``weighted-average`` or ``sum``.
        variance_estimator:
            the type of variance estimator to use.

    Attributes:
        user_index_(pandas.Index): User index.
        item_index_(pandas.Index): Item index.
        user_means_(numpy.ndarray): User mean ratings.
        rating_matrix_(matrix.CSR): Normalized user-item rating matrix.
        transpose_matrix_(matrix.CSR): Transposed un-normalized rating matrix.
    """
    # Aggregation functions
    AGG_SUM = intern('sum')
    AGG_WA = intern('weighted-average')
    AGG_AVG = intern('average')

    # Standard deviation functions for average and weighted-average    
    VAR_AVG_STD = intern('standard-deviation-average')
    VAR_WGT_AVG_STD = intern('standard-deviation-weighted-average')

    # JK estimate of standard deviation for average and weighted-average 
    VAR_AVG_STD_JK = intern('standard-deviation-jackknife-average')
    VAR_WGT_AVG_STD_JK = intern('standard-deviation-jackknife-weighted-average')

    # BS estimate of standard deviation for average and weighted-average 
    VAR_AVG_STD_BS = intern('standard-deviation-bootstrap-average')
    VAR_WGT_AVG_STD_BS = intern('standard-deviation-bootstrap-weighted-average')


    def __init__(self, nnbrs, min_nbrs=1, min_sim=0, center=True, aggregate='weighted-average', variance_estimator = 'standard-deviation-weighted-average'):
        self.nnbrs = nnbrs
        self.min_nbrs = min_nbrs
        self.min_sim = min_sim
        self.center = center
        self.aggregate = intern(aggregate)
        self.variance_estimator = intern(variance_estimator)

    def fit(self, ratings, **kwargs):
        """
        "Train" a user-user CF model.  This memorizes the rating data in a format that is usable
        for future computations.

        Args:
            ratings(pandas.DataFrame): (user, item, rating) data for collaborative filtering.

        Returns:
            UUModel: a memorized model for efficient user-based CF computation.
        """

        _logger.info('calling fit in user_knn')
        uir, users, items = matrix.sparse_ratings(ratings)

        # mean-center ratings
        if self.center:
            umeans = uir.normalize_rows('center')
        else:
            umeans = None

        # compute centered transpose
        iur = uir.transpose()

        # L2-normalize ratings so dot product is cosine
        if uir.values is None:
            uir.values = np.full(uir.nnz, 1.0)
        uir.normalize_rows('unit')

        mkl = matrix.mkl_ops()
        mkl_m = mkl.SparseM.from_csr(uir) if mkl else None

        self.rating_matrix_ = uir
        self.user_index_ = users
        self.user_means_ = umeans
        self.item_index_ = items
        self.transpose_matrix_ = iur
        self._mkl_m_ = mkl_m

        return self

    def predict_for_user(self, user, items, ratings=None):
        """
        Compute predictions for a user and items.

        Args:
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series):
                the user's ratings (indexed by item id); if provided, will be used to
                recompute the user's bias at prediction time.

        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """
        #_logger.warning('starting predict for user: %d', user)
        #_logger.info('calling predict_for_user in user_knn')
        #_logger.info('calling _get_user_data in predict_for_user in user_knn')
        watch = util.Stopwatch()
        items = pd.Index(items, name='item')
	
        #_logger.info('user ratings before _get_user_data')
        #_logger.info(ratings)
        #_logger.info('user: %d', user)
       
        ratings, umean = self._get_user_data(user, ratings)
        
        
        #_logger.info('user ratings')
        #_logger.info(ratings)
        #_logger.info('user mean: %s', umean)

        if ratings is None: ######
            empty_results = pd.Series(index=items, dtype='float64')
            empty_results = pd.DataFrame(empty_results)
            empty_results['prediction'] = np.full(len(items), np.nan, dtype=np.float_)
            empty_results['item'] = items 
            empty_results['user'] = user
            empty_results['var'] = np.full(len(items), np.nan, dtype=np.float_) #nbhr_ratings
            empty_results['num_nbhr'] = np.full(len(items), np.nan, dtype=np.float_)
            return empty_results
        assert len(ratings) == len(self.item_index_)  # ratings is a dense vector

        # now ratings is normalized to be a mean-centered unit vector
        # this means we can dot product to score neighbors
        # score the neighbors!
        if self._mkl_m_:
            nsims = np.zeros(len(self.user_index_))
            nsims = self._mkl_m_.mult_vec(1, ratings, 0, nsims)
        else:
            rmat = self.rating_matrix_.to_scipy()
            nsims = rmat @ ratings
        assert len(nsims) == len(self.user_index_)
        if user in self.user_index_:
            nsims[self.user_index_.get_loc(user)] = 0

        _logger.debug('computed user similarities')
        
        # EMPTY RESULTS VECTOR
        results = np.full(len(items), np.nan, dtype=np.float_)
        #_logger.info('results before score call')
        #_logger.info(results)

        # EMPTY VARIANCE VECTOR
        nbhr_ratings = np.zeros(len(results)) #[None] * len(ratings) # np.zeros(len(ratings))

        # EMPTY NEIGHBOR VECTOR
        num_nbhr_actual = np.zeros(len(results)) #[None] * len(ratings) # np.zeros(len(ratings))

        # ITEM POSITIONS
        ri_pos = self.item_index_.get_indexer(items.values)
        #_logger.info('ri_pos')
        #_logger.info(ri_pos)

        # AGGREGATION TYPE
        #_logger.info('selecting aggregation type')
        if self.aggregate == self.AGG_WA:
            agg = _agg_weighted_avg
        elif self.aggregate == self.AGG_SUM:
            agg = _agg_sum
        elif self.aggregate == self.AGG_AVG:
            agg = _agg_avg
        else:
            raise ValueError('invalid aggregate ' + self.aggregate)
        #_logger.info('aggregation type selected')

        # VARIANCE TYPE
        if self.variance_estimator == self.VAR_AVG_STD:
            var = _rating_avg_std
        elif self.variance_estimator == self.VAR_WGT_AVG_STD:
            var = _rating_weighted_avg_std
        elif self.variance_estimator == self.VAR_AVG_STD_JK:
            var = _rating_avg_jk_std
        elif self.variance_estimator == self.VAR_WGT_AVG_STD_JK:
            var = _rating_weighted_avg_jk_std
        elif self.variance_estimator == self.VAR_AVG_STD_BS:
            var = _rating_avg_bs_std
        elif self.variance_estimator == self.VAR_WGT_AVG_STD_BS:
            var = _rating_weighted_avg_bs_std
        else:
            raise ValueError('invalid variance estimator: ' + self.variance_estimator)

        

        # SCORING FUNCTION

        
        #_logger.info('calling _score in user_knn for user: %d', user) #####
        # SEEMS LIKE _score function changes 'results' even though that's not in return
        _score(ri_pos, results, self.transpose_matrix_.N, nsims,
               self.nnbrs, self.min_sim, self.min_nbrs, agg, nbhr_ratings, num_nbhr_actual, var) # _score returns 
        #_logger.info('testing _score output')
        #_logger.info(used)
        
        # NEIGHBORHOOD RATING VARIANCES
        #_logger.info('number of neighborhood rating var: %s', len(nbhr_ratings))
        #_logger.info('neighborhood rating variances')
        #_logger.info(nbhr_ratings)
        
	
        #_logger.info('results after score call, but before mean adjusted')
        #_logger.info('number of results: %s', len(results))
        #_logger.info(results)

        results += umean

        #_logger.info('results after mean adjusted')
        #_logger.info(results)
        
        results = pd.Series(results, index=items, name='prediction')

        results_df = pd.DataFrame(results)
        results_df['item'] = results.index  
        results_df['user'] = user
        results_df['var'] = nbhr_ratings
        results_df['num_nbhr'] = num_nbhr_actual

        #nbhr_ratings = pd.Series(nbhr_ratings, index=items, name='var')
        #nbhr_ratings['item'] = nbhr_ratings.index
        

        

        #_logger.info('results pd.Series')
        #_logger.info(results)

        #_logger.info('nbhr var pd.Series')
        #_logger.info(nbhr_ratings)

        
        #results = pd.concat([results, nbhr_ratings], axis = 1).reset_index()
        
        #results = pd.DataFrame(results)
        #results = pd.DataFrame({'item': items}, {'prediction': results}, {'var': nbhr_ratings})

        _logger.debug('scored %d of %d items for %s in %s',
                      results.notna().sum(), len(items), user, watch)
        return results_df #, 2 #nbhr_ratings

    def _get_user_data(self, user, ratings):
        "Get a user's data for user-user CF"
        rmat = self.rating_matrix_
	
        #_logger.info('calling _get_user_data in user_knn')
	
	#rmat = self.rating_matrix_

        if ratings is None:
            try:
                #_logger.info('no additional rating information provided on user')
                upos = self.user_index_.get_loc(user)
                ratings = rmat.row(upos)
                #_logger.info('ratings in _get_user_data in try statement')
                #_logger.info(ratings)
                umean = self.user_means_[upos] if self.user_means_ is not None else 0
            except KeyError:
                #_logger.warning('user %d has no ratings and none provided', user)
                return None, 0
        else:
            #_logger.debug('using provided ratings for user %d', user)
            if self.center:
                umean = ratings.mean()
                ratings = ratings - umean
            else:
                umean = 0
            unorm = np.linalg.norm(ratings)
            ratings = ratings / unorm
            ratings = ratings.reindex(self.item_index_, fill_value=0).values

        #_logger.info('ratings in _get_user_data at function end')
        #_logger.info(ratings)
        #_logger.info('user mean in _get_user_data at function end')
        #_logger.info(umean) 
       
        return ratings, umean

    def __getstate__(self):
        state = dict(self.__dict__)
        if '_mkl_m_' in state:
            del state['_mkl_m_']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.aggregate = intern(self.aggregate)
        mkl = matrix.mkl_ops()
        self._mkl_m_ = mkl.SparseM.from_csr(self.rating_matrix_) if mkl else None

    def __str__(self):
        return 'UserUserCA(nnbrs={}, min_sim={})'.format(self.nnbrs, self.min_sim)
