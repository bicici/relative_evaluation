import sys, os
from numpy import asarray, sqrt, abs, clip
from scipy import isnan
from scipy.stats import pearsonr

def calculateScores(y, y_pred):
    """ Calculates various scores between the actual target and the prediction.
        R2: R^2 coefficient of determination
        r: Pearson's correlation coefficient
        RMSE: Root mean squared error
        MAE: Mean absolute error
        RAE: Relative mean absolute error
        MAER: MAE relative
        MRAER: Mean RAE relative
    """
    evalnames = ['R2', 'r', 'RMSE', 'MAE', 'RAE', 'MAER', 'MRAER']
    N = float(len(y))
    if N != len(y_pred):
        print ('number of scores mismatch:', N, '!=', len(y_pred))
    if type(y_pred) == list:
        y_pred = asarray(y_pred)
    if type(y) == list:
        y = asarray(y)
    d_target = y - y_pred
    d_pmean = y_pred - y_pred.mean()
    d_tmean = y - y.mean()
    RegSSE = d_target.dot(d_target)
    ResidualSSE = d_tmean.dot(d_tmean)
    scoresDict = {}
    if ResidualSSE > 0:
        R2 = 1 - RegSSE / ResidualSSE
    else:
        R2 = 0.0
    scoresDict['R2'] = R2
    scoresDict['RMSE'] = sqrt(RegSSE / N)
    absd_target = abs(d_target)
    sumd_target = absd_target.sum()
    absd_tmean = abs(d_tmean)
    sumd_tmean = absd_tmean.sum()
    if sumd_tmean > 0:
        RAE = sumd_target / sumd_tmean
    else:
        RAE = -1.0
    MAE = sumd_target / N
    scoresDict['MAE'] = MAE
    scoresDict['RAE'] = RAE
    stepsize = absd_tmean.mean()
    epsilon = 0.5 * stepsize
    absy = abs(y)
    if epsilon == 0:
        epsilon = absd_tmean[absd_tmean > 0].min()
        if epsilon == 0:
            epsilon = absy[absy > 0].min()
    absy = clip(absy, epsilon, None)
    absy[absy == 0] = epsilon
    MAER = absd_target / absy
    absd_tmean = clip(absd_tmean, epsilon, None)
    MRAER = absd_target / absd_tmean
    scoresDict['MAER'] = MAER.sum() / N
    scoresDict['MRAER'] = MRAER.sum() / N
    std_y_pred = y_pred.std()
    std_y = y.std()
    d_pmean = d_pmean / std_y_pred
    d_tmean = d_tmean / std_y
    scoresDict['r'] = pearsonr(y, y_pred)[0]
    if isnan(scoresDict['RAE']):
        scoresDict['RAE'] = 0.0
    if isnan(scoresDict['r']):
        scoresDict['r'] = 0.0
    scores = [scoresDict[x] for x in evalnames]
    print ('               \t'.join(evalnames))
    print ('\t'.join([str(x) for x in scores]))
    return scores

if __name__ == '__main__':
    y, yhat = sys.argv[1], sys.argv[2]
    print ('y:', y, '\ty_hat:', yhat)
    if not os.path.exists(y):
        print (y, 'not found.')
        sys.exit()
    if not os.path.exists(yhat):
        print (yhat, 'not found.')
        sys.exit()
    y = [float(x) for x in open(y).readlines()]
    yhat = [float(x) for x in open(yhat).readlines()]
    calculateScores(y, yhat)
