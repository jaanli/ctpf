import bottleneck as bn
import numpy as np
from math import log
from operator import div, add
from scipy import sparse
import scipy
import time, sys, json
import pandas as pd
import cPickle as pickle
import logging

logger = logging.getLogger()

def load_data(csv_file, shape, binarize):
    tp = pd.read_csv(csv_file, delimiter='\t', header=None, names=['uid', 'did', 'count'])
    rows, cols = np.array(tp['did'], dtype=np.int32), np.array(tp['uid'], dtype=np.int32)
    count = tp['count']
    matrix = scipy.sparse.csr_matrix((count,(rows, cols)), dtype=np.int16, shape=shape)
    if binarize:
        matrix.data = np.ones_like(matrix.data)
    return matrix, rows, cols

def user_idx_generator(n_users, batch_users):
    for start in xrange(0, n_users, batch_users):
        end = min(n_users, start + batch_users)
        yield slice(start, end)

def make_str(dictionary):
    dictionary = {str(k):str(v) for k,v in dictionary.items()}
    return dictionary

def load_all(dat_dir, binarize=True):
    unique_uid = list()
    with open(dat_dir+'unique_uid_sub.txt', 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
    unique_sid = list()
    with open(dat_dir+'unique_sid_sub.txt', 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    with open(dat_dir + 'song2id_sub.json', 'r') as f:
        song2numid_sub = json.load(f)

    songnum2songid = {num:str(song) for song, num in song2numid_sub.items()}

    #load maps
    map_dir = '/home/waldorf/altosaar/projects/music-recommendations/dat/mappings_from_metadata/'
    with open(map_dir + 'songid2artistid.json', 'r') as f:
        songid2artistid = json.load(f)
    with open(map_dir + 'songid2songname.json', 'r') as f:
        songid2songname = json.load(f)
    with open(map_dir + 'artistid2artistname.json', 'r') as f:
        artistid2artistname = json.load(f)

    songid2artistid = make_str(songid2artistid)
    songid2songname = make_str(songid2songname)
    artistid2artistname = make_str(artistid2artistname)


    artistid_present = set([str(songid2artistid[sid]) for sid in unique_sid])

    n_songs = len(unique_sid)
    n_users = len(unique_uid)
    n_artists = len(artistid_present)

    artistid2artistnum = {artist:num for num, artist in enumerate(artistid_present)}
    artistnum2artistid = {v:k for k,v in artistid2artistnum.items()}
    artistnum2artistname = {n:artistid2artistname[artistnum2artistid[n]] for n in artistnum2artistid.keys()}

    # Standard format! length: n_songs, corresponding to artist_num of each song!
    song2artist = np.array([artistid2artistnum[songid2artistid[songnum2songid[songnum]]] for songnum in songnum2songid.keys()])

    songnum2fullname = {songnum:artistnum2artistname[artistnum] + ' - ' + songid2songname[songnum2songid[songnum]] for songnum, artistnum in enumerate(song2artist)}
    def load_data(csv_file, shape=(n_songs, n_users)):
        tp = pd.read_csv(csv_file)
        rows, cols = np.array(tp['sid'], dtype=np.int32), np.array(tp['uid'], dtype=np.int32)
        count = tp['count']
        return sparse.csr_matrix((count,(rows, cols)), dtype=np.int16, shape=shape), rows, cols

    train_data, rows, cols = load_data(dat_dir+'in.train.num.sub.csv')

    vad_smat, rows_vad, cols_vad = load_data(dat_dir+'in.vad.num.sub.csv')
    test_smat, rows_test, cols_test = load_data(dat_dir + 'in.test.num.sub.csv')

    # binarize the data
    if binarize == True:
        train_data.data = np.ones_like(train_data.data)
        vad_smat.data = np.ones_like(vad_smat.data)
        test_smat.data = np.ones_like(test_smat.data)

    vad_data = vad_smat.data
    test_data = test_smat.data

    vad = dict(X_new=vad_data,
               rows_new=rows_vad,
               cols_new=cols_vad)

    train_t = train_data.transpose().tocsr()
    vad_t = vad_smat.transpose().tocsr()
    test_t = test_smat.transpose().tocsr()

    return dict(song2artist=song2artist,
                songnum2fullname=songnum2fullname,
                artistnum2artistname=artistnum2artistname,
                train_data=train_data,
                rows=rows,
                cols=cols,
                vad=vad,
                train_t=train_t,
                vad_t=vad_t,
                test_t=test_t,
                test_data=test_data,
                rows_test=rows_test,
                cols_test=cols_test)


# data should be in the shape of (n_users, n_items)
def precision_at_k_batch(train_data, vad_data, test_data, Et, Eb, user_idx,
                         k=20, normalize=True):
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, vad_data, Et, Eb, user_idx,
                              batch_users)
    idx = bn.argpartsort(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.tile(np.arange(batch_users), (k, 1)).T, idx[:, :k]] = True

    X_true_binary = (test_data[user_idx] > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)

    if normalize:
        precision = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    else:
        precision = tmp / k
    return precision


def mean_rank(data, Et, Eb, user_idx):
    X_pred = Et[user_idx].dot(Eb)
    # rank starts with 1
    all_rank = np.argsort(np.argsort(-X_pred, axis=1), axis=1) + 1
    X_true_binary = (data[user_idx] > 0).tocoo()
    rank = sparse.csr_matrix((all_rank[X_true_binary.row, X_true_binary.col],
                              (X_true_binary.row, X_true_binary.col)),
                             shape=all_rank.shape)
    return np.asarray(rank.sum(axis=1),
                      dtype=np.float32).ravel() / rank.getnnz(axis=1)


def mean_rrank_at_k_batch(train_data, vad_data, test_data, Et, Eb,
                          user_idx, k=5):
    '''
    mean reciprocal rank@k: For each user, make predictions and rank for
    all the items. Then calculate the mean reciprocal rank for the top K that
    are in the held-out test set.
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, vad_data, Et, Eb, user_idx,
                              batch_users)
    all_rrank = 1. / (np.argsort(np.argsort(-X_pred, axis=1), axis=1) + 1)
    X_true_binary = (test_data[user_idx] > 0).toarray()

    test_rrank = X_true_binary * all_rrank
    top_k = bn.partsort(-test_rrank, k, axis=1)
    return -top_k[:, :k].mean(axis=1)


def mean_perc_rank_batch(train_data, vad_data, test_data, Et, Eb, user_idx):
    '''
    mean percentile rank for a batch of users
    MPR of the full set is the sum of batch MPR's divided by the sum of all the
    feedbacks. (Eq. 8 in Hu et al.)
    This metric not necessarily constrains the data to be binary
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, vad_data, Et, Eb, user_idx,
                              batch_users)
    all_perc = np.argsort(np.argsort(-X_pred, axis=1), axis=1) / \
        np.isfinite(X_pred).sum(axis=1, keepdims=True).astype(np.float32)
    perc_batch = (all_perc[test_data[user_idx].nonzero()] *
                  test_data[user_idx].data).sum()
    return perc_batch

def NDCG_binary(train_data, vad_data, test_data, Et, Eb, user_idx):
    '''
    normalized discounted cumulative gain for binary relevance
    '''
    batch_users = user_idx.stop - user_idx.start
    n_items = train_data.shape[1]

    X_pred = _make_prediction(train_data, vad_data, Et, Eb, user_idx,
                              batch_users)
    all_rank = np.argsort(np.argsort(-X_pred, axis=1), axis=1)
    # build the discount template
    tp = np.hstack((1, 1. / np.log2(np.arange(2, n_items + 1))))
    all_disc = tp[all_rank]

    X_true_binary = (test_data[user_idx] > 0).tocoo()
    disc = sparse.csr_matrix((all_disc[X_true_binary.row, X_true_binary.col],
                              (X_true_binary.row, X_true_binary.col)),
                             shape=all_disc.shape)
    DCG = np.array(disc.sum(axis=1)).ravel()
    IDCG = np.array([tp[:n].sum() for n in test_data[user_idx].getnnz(axis=1)])
    return DCG / IDCG

def calc_all(train_data, validation_data, test_data, Et, Eb):
    train_t = train_data.transpose().tocsr()
    vad_t = validation_data.transpose().tocsr()
    test_t = test_data.transpose().tocsr()
    res_prec = list()
    #res_mrr = list()
    #res_mpr = list()
    res_ndcg = list()
    n_users = train_t.shape[0]
    start_t = time.time()
    for i, user_idx in enumerate(user_idx_generator(n_users, 1000), 1):
        res_prec.append(precision_at_k_batch(train_t, vad_t, test_t,
                                                     Et, Eb, user_idx))
        # res_mrr.append(mean_rrank_at_k_batch(train_t, vad_t, test_t, Et, Eb,
        #                       user_idx))
        # res_mpr.append(mean_perc_rank_batch(train_t, vad_t, test_t, Et, Eb,
        #                       user_idx))
        res_ndcg.append(NDCG_binary(train_t, vad_t, test_t, Et, Eb, user_idx))
        sys.stdout.write('\rProgress: %d/%d\t Time: %.2f sec/batch' % (user_idx.stop, n_users, (time.time() - start_t) / i))
        sys.stdout.flush()
    mn_prec = np.hstack(res_prec)
    mnprec = mn_prec[~np.isnan(mn_prec)].mean()
    # mn_mrr = np.hstack(res_mrr)
    # mnmrr = mn_mrr[~np.isnan(mn_mrr)].mean()
    # mpr = np.sum(res_mpr)/np.sum(test_t.data)
    mn_ndcg = np.hstack(res_ndcg)
    mnndcg = mn_ndcg[~np.isnan(mn_ndcg)].mean()
    # txt = '\rmean precision @ 20:\n %.5f\n mean reciprocal rank: %.5f\n mean percentile rank:\n %.5f\n mean ndcg:\n %.5f\n' % (np.around(mnprec, decimals=5), np.around(mnmrr, decimals=5), np.around(mpr, decimals=5), np.around(mnndcg, decimals=5))
    txt = '\rmean precision @ 20:\n %.5f\n mean ndcg:\n %.5f\n' % (np.around(mnprec, decimals=5), np.around(mnndcg, decimals=5))
    #sys.stdout.write(txt)
    logging.info(txt)
    # with open(out_dir + 'eval.txt', 'wb') as f:
    #     print>>f, txt

def write_latent(out_dir, theta, beta_a, beta_s, beta):
    pickle.dump(theta, open(out_dir + 'theta.pickle', 'wb'))
    pickle.dump(beta_a, open(out_dir + 'beta_a.pickle', 'wb'))
    pickle.dump(beta_s, open(out_dir + 'beta_s.pickle', 'wb'))
    pickle.dump(beta, open(out_dir + 'beta.pickle', 'wb'))
def write_latent_v(out_dir, theta, beta_s):
    # v for Vanilla, songs only vanilla pmf
    pickle.dump(theta, open(out_dir + 'theta.pickle', 'wb'))
    pickle.dump(beta_s, open(out_dir + 'beta_s.pickle', 'wb'))

def write_dict(out_dir, artistnum2artistname, songnum2fullname, song2artist):
    pickle.dump(artistnum2artistname, open(out_dir + 'artistnum2artistname.pickle', 'wb'))
    pickle.dump(songnum2fullname, open(out_dir + 'songnum2fullname.pickle', 'wb'))
    pickle.dump(song2artist, open(out_dir + 'song2artist.pickle', 'wb'))


def write_top(out_dir, beta_a, beta_s, artistnum2artistname, songnum2fullname, song2artist):
    beta_a = _normalize(beta_a)
    beta_s = _normalize(beta_s)
    _write_top_songs(out_dir, beta_s, songnum2fullname, song2artist)
    _write_top_artists(out_dir, beta_a, artistnum2artistname)

def _normalize(array):
    for i in range(array.shape[0]):
        array[i] /= array[i].sum()
    return array

def _write_top_songs(out_dir, beta_s, songnum2fullname, song2artist):
    # beta_s is shape K x S
    K = beta_s.shape[0]
    count_mean = []
    with open(out_dir + 'top_songs_k' + str(K) + '.txt', 'wb') as f:
        for k in range(0,K):
            #print>>f, 'top songs'
            mn, top = _top_songs(k, beta_s, songnum2fullname, song2artist)
            count_mean.append(mn)
            print>>f, '\n--------------------------------\n'
            print>>f, 'topic k = ' + str(k) + ', average songs per artist: ' + str(mn) + '\n'
            for line in top:
                print>>f, line
        strng = '\n average songs per artist in each top 10: ' + str(np.mean(count_mean))
        print>>f, strng

def _write_top_artists(out_dir, beta_a, artistnum2artistname):
    # beta_a is shape K x A
    K = beta_a.shape[0]
    with open(out_dir + 'top_artists_k' + str(K) + '.txt', 'wb') as f:
        for k in range(0,K):
            print>>f, '\n--------------------------------\n'
            print>>f, 'topic k = ' + str(k) + '\n'
            #print>>f, 'top songs'
            top = _top_artists(k, beta_a, artistnum2artistname)
            for line in top:
                print>>f, line

def _top_artists(k, beta, artistnum2artistname, n=10):
    top = np.argsort(beta[k,:])[::-1]
    return ["%0.3f  %s" %(beta[k, topId],artistnum2artistname[topId]) for topId in top[0:n]]

def _top_songs(k,beta,songnum2fullname, song2artist, n=10):
    top = np.argsort(beta[k,:])[::-1]
    artist_nums = [song2artist[song] for song in top[0:n]]
    counts = [artist_nums.count(artist_num) for artist_num in set(artist_nums)]
    mn = np.mean(counts)
    return mn, ["%0.3f  %s" %(beta[k, topId],songnum2fullname[topId]) for topId in top[0:n]]

def _make_prediction(train_data, vad_data, Et, Eb, user_idx, batch_users):
    n_songs = train_data.shape[1]
    # exclude examples from training and validation
    item_idx = np.zeros((batch_users, n_songs), dtype=bool)
    item_idx[train_data[user_idx].nonzero()] = True
    item_idx[vad_data[user_idx].nonzero()] = True
    X_pred = Et[user_idx].dot(Eb)
    X_pred[item_idx] = -np.inf
    return X_pred
