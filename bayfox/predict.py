import numpy as np
import numpy.matlib
import attr
from bayfox.modelparams import get_draws
from bayfox.utils import alphaT

@attr.s()
class Prediction:
    ensemble = attr.ib()

    def percentile(self, q=None, interpolation='nearest'):
        """Compute the qth ranked percentile from ensemble members.

        Parameters
        ----------
        q : float ,sequence of floats, or None, optional
            Percentiles (i.e. [0, 100]) to compute. Default is 5%, 50%, 95%.
        interpolation : str, optional
            Passed to numpy.percentile. Default is 'nearest'.

        Returns
        -------
        perc : ndarray
            A 2d (nxm) array of floats where n is the number of predictands in
            the ensemble and m is the number of percentiles ('len(q)').
        """
        if q is None:
            q = [5, 50, 95]
        q = np.array(q, dtype=np.float64, copy=True)

        # Because analog ensembles have 3 dims
        target_axis = list(range(self.ensemble.ndim))[1:]

        perc = np.percentile(self.ensemble, q=q, axis=target_axis,
                             interpolation=interpolation)
        return perc.T


def predict_d18oc(seatemp, d18osw, foram=None, seasonal_seatemp=False,
                  drawsfun=get_draws):
    """Predict δ18O of foram calcite given seawater temperature and seawater δ18O.

    Parameters
    ----------
    seatemp : array_like or scalar
        n-length array or scalar of sea-surface temperature (°C).
    d18osw : array_like or scalar
        n-length array or scalar of δ18O of seawater (‰; VSMOW). If not scalar,
        must be the same length as ``seatemp``.
    foram : str, optional
        Foraminifera group name of ``d18oc`` sample. Can be 'T. sacculifer',
        'N. pachyderma', 'G. bulloides', 'N. incompta' or ``None``.
        If ``None``, pooled calibration model is used.
    seasonal_seatemp : bool, optional
        Indicates whether sea-surface temperature is annual or seasonal
        estimate. If ``True``, ``foram`` must be specified.
    drawsfun : function-like, optional
        For debugging and testing. Object to be called to get MCMC model
        parameter draws. Don't mess with this.

    Returns
    -------
    prediction : Prediction
        Model prediction estimating δ18O of planktic foraminiferal calcite
        (‰; VPDB).
    """
    seatemp = np.atleast_1d(seatemp)
    d18osw = np.atleast_1d(d18osw)
    seasonal_seatemp = bool(seasonal_seatemp)

    alpha, beta, tau = drawsfun(foram=foram, seasonal_seatemp=seasonal_seatemp)

    # Unit adjustment.
    d18osw_adj = d18osw - 0.27

    mu = alpha + seatemp[:, np.newaxis] * beta + d18osw_adj[:, np.newaxis]
    y = np.random.normal(mu, tau)

    return Prediction(ensemble=y)


def pHCorrect(d18oc,seatemp,S,pH,cortype):
    """
    # function d18OpH = pHCorrect(d18oc,T,S,pH,cortype)
    #
    # this is an add-on for the BAYFOX forward model that corrects
    # forward-modeled d18Oc for the "pH effect", sometimes called the
    # "carbonate ion effect". The BAYFOX calibration uses coretop sediments
    # that were deposited at presumably preindustrial/late Holocene ocean pH.
    # However ocean pH changes with atmospheric pCO2 so for deep-time
    # applications, a correction needs to be made to account for this. The pH
    # of the ocean influences the speciation of dissolved inorganic carbon,
    # thus impacting the net fractionation factor when calcite is formed.
    #
    # ----- Inputs -----
    # d18Oc: The 1000-member ensemble estimate of d18O of calcite from bayfox_forward (N x 1000)
    #
    # seatemp: SST (scalar or vector)
    # S: SSS (scalar or vector)
    # pH: pH values for the paleo time period (scalar)
    # cortype: either 0 or 1. Enter 1 to use the reduced sensitivity of O. universa.
    #
    # ----- Outputs -----
    #
    # d18opH: A 1000-member ensemble estimate of d18O of calcite, corrected for
    # the pH effect.
    #
    # Matlab version by J. Tierney
    #    2021
    # Python version by Mingsong Li
    #     Peking University
    #     June 23, 2021
    """
    
    #
    # ensure vector for pH
    seatemp = np.atleast_1d(seatemp)
    pH = np.atleast_1d(pH)
    S = np.atleast_1d(S)
    #normalize the curve to PI pH
    PIpH = 8.16 # best estimate of preindustrial pH. 
    # SOURCE: Jiang et al., 2019, https://doi.org/10.1038/s41598-019-55039-4,
    # 1770-1850 global mean (weighted by latitude).
    # set pH range for theoretical curve
    # pHrange = (6.5:.01:8.8)';
    # use alphaT to get the fractionation factor alpha

    AT = alphaT(seatemp,S,pH)
    ATb = alphaT(seatemp,S,PIpH)

    # calculate carbonate and convert to VPDB scale
    # don't need to account for d18Osw since we are interested in relative change.
    d18Ocf =  ((AT - 1) * 1000) * 0.97001 - 29.99  #Brand et al 2014 conversion
    d18Ocb = ((ATb - 1) * 1000) * 0.97001 - 29.99

    # normalize data
    d18OcN = d18Ocf - d18Ocb

    # shape
    d18oc_0 = d18oc.shape[0]
    d18oc_1 = d18oc.shape[1]

    if cortype == 1:
        # use 1-sigma for Orbulina
        dcErr = 0.14
        # Orbulina regression
        pH_cor = -0.27 * (pH - PIpH)
        d18opH = d18oc + np.random.normal(np.matlib.repmat(pH_cor,d18oc_0,d18oc_1),np.matlib.repmat(dcErr,d18oc_0,d18oc_1))
    else:
        # define the 1-sigma error. Best estimate based on culture studies is 0.27
        dcErr = 0.27
        # normal random sampling
        d18opH = d18oc + np.random.normal(np.matlib.repmat(d18OcN,1,d18oc_1),np.matlib.repmat(dcErr,d18oc_0,d18oc_1))

    return Prediction(ensemble=d18opH)

def predict_seatemp(d18oc, d18osw, prior_mean, prior_std, foram=None,
                    seasonal_seatemp=False, drawsfun=get_draws):
    """Predict sea-surface temperature given δ18O of calcite and seawater δ18O.

    Parameters
    ----------
    d18oc : array_like or scalar
        n-length array or scalar of δ18O of planktic foraminiferal calcite
        (‰; VPDB).
    d18osw : array_like or scalar
        n-length array or scalar of δ18O of seawater (‰; VSMOW). If not scalar, must be the same length as
        ``d18oc``.
    prior_mean : scalar
        Prior mean of sea-surface temperature (°C).
    prior_std : scalar
        Prior standard deviation of sea-surface temperature (°C).
    foram : str, optional
        Foraminifera group name of ``d18oc`` sample. Can be 'T. sacculifer',
        'N. pachyderma', 'G. bulloides', 'N. incompta' or ``None``.
        If ``None``, pooled calibration model is used.
    seasonal_seatemp : bool, optional
        Indicates whether sea-surface temperature is annual or seasonal
        estimate. If ``True``, ``foram`` must be specified.
    drawsfun : function-like, optional
        For debugging and testing. Object to be called to get MCMC model
        parameter draws. Don't mess with this.

    Returns
    -------
    prediction : Prediction
        Model prediction giving estimated sea-surface temperature (°C).
    """
    d18oc = np.atleast_1d(d18oc)
    d18osw = np.atleast_1d(d18osw)
    seasonal_seatemp = bool(seasonal_seatemp)

    alpha, beta, tau = drawsfun(foram=foram, seasonal_seatemp=seasonal_seatemp)

    # Unit adjustment.
    d18osw_adj = d18osw - 0.27

    prior_mean = np.atleast_1d(prior_mean)
    prior_inv_cov = np.atleast_1d(prior_std).astype(float) ** -2

    precision = tau ** -2
    post_inv_cov = prior_inv_cov[..., np.newaxis] + precision * beta ** 2
    post_cov = 1 / post_inv_cov
    mean_first_factor = ((prior_inv_cov * prior_mean)[:, np.newaxis] + precision
                         * beta * ((d18oc - d18osw_adj)[:, np.newaxis] - alpha))
    mean_full = post_cov * mean_first_factor

    y = np.random.normal(loc=mean_full, scale=np.sqrt(post_cov))
    return Prediction(ensemble=y)
