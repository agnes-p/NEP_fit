import numpy as np
import numpy.ma as ma
import scipy.signal as spsig
import scipy.stats.mstats as spsm
import scipy.optimize as spopt
import pandas as pd
from matplotlib import pyplot as plt


def estimate_baseline_noise(
    xdata,
    ydata,
    skew_target=2.0,
    inc=0.005,
    skew_chunks=10,
    bl_bin=None,
    make_plots=False,
    sigma=3,
):
    """
    Function that estimates the frequency dependent baseline and noise level for a spectrum by analyzing the skew statistic
    of the data points. The assumption is that the overall noise distribution in the spectrum is approximately Gaussian, and
    that the presence of peaks will skew the data toward positive y values (and therefore a postive value of skewness).
    A common statistical test for consistency with a Gaussian distribution is a skew statistic between -2 and +2. This function
    by default divides the spectrum into 10 chunks. For each chunk, it calculates the skew and if the value is greater than 2,
    it iteratively removes the top 0.5% of the data points until the remaining data points have a skew value less than 2.

    The remaining points are used as an estimate of the baseline, and a moving average and standard deviation are calculated by convolution.
    By default, the window size for the convolution is 1/20 of the data length. The outermost 1/40 of the data points are extended and
    repeated to prevent the baseline from decaying at the data edges.

    The function returns 2 curves: an average baseline and a standard deviation of the baseline, each of the same length as xdata.
    If make_plots is True, graphs showing the intial data histogram and the baseline data histogram are created, along with a graph
    showing the data, baseline points, baseline average, and the 3sigma baseline. The Figure and list of axes are additionally returned.

    Parameters
    ----------
    xdata : array-like (1D)
        x values of spectrum data points
    ydata : array-like (1D)
        y values of spectrum data points, must be same size as xdata
    skew_target : float, optional
        Target value for skewness statistic in baseline estimation (default: 2.0)
    inc : float, optional
        Fraction of data points to remove on each iteration of the skew test (default: 0.005)
    skew_chunks : int, optional
        Number of chunks to divide the spectrum into for the skew test. This setting might be adjusted if the noise level varies
        significantly across the spectrum (default: 10)
    bl_bin : int or None, optional
        Window size for baseline average and standard deviation calculation. The default of None adopts a value of len(ydata)//20
    make_plots : boolean, optional
        If True, generates and returns a Figure and Axes list with the raw ydata histogram, the baseline histogram,
        and the data points with baseline, average, and 3sigma values (default: False)
    sigma : float, optional
        The multiple of the standard deviation to add to the average baseline to set the noise level.
        Only used if make_plots is True (default: 3)

    Returns
    -------
    array-like
        Baseline average, same length as xdata
    array-like
        Standard deviation of the baseline, same length as xdata
    pyplot.Figure or None
        Matplotlib Figure object containing plots (only if make_plots is True)
    list of pyplot.Axes or None
        List of 3 Axes objects for each plot on the figure (only if make_plots is True)
    """

    if xdata.shape != ydata.shape:
        raise ValueError("xdata and ydata must have the same shape")

    if xdata.ndim > 1 or ydata.ndim > 1:
        raise ValueError("xdata and ydata must be 1 dimensional")

    if skew_chunks < 1:
        raise ValueError("skew_chunks must be >= 1")

    chunk_size = len(ydata) // skew_chunks + 1
    yy = ma.array([])
    for i in range(0, skew_chunks):
        cutoff = 0
        while True:
            if (i + 1) * chunk_size >= len(ydata):
                yyy = spsm.trim(ydata[i * chunk_size :], (0, cutoff), relative=True)
            else:
                yyy = spsm.trim(
                    ydata[i * chunk_size : (i + 1) * chunk_size],
                    (0, cutoff),
                    relative=True,
                )

            d = spsm.describe(yyy)
            if d.skewness < skew_target:
                yy = ma.concatenate([yy, yyy])
                break
            else:
                cutoff += inc

    if bl_bin is None:
        bl_bin = len(xdata) // 20

    bl = ydata[np.logical_not(yy.mask)]
    bl_pre = bl[0 : bl_bin // 2]
    bl_post = bl[-(bl_bin // 2) :]
    bl_pad = np.concatenate([bl_pre, bl, bl_post])
    blx = xdata[np.logical_not(yy.mask)]
    baseline_ = spsig.oaconvolve(bl_pad, np.ones(bl_bin) / bl_bin, mode="same")
    stdev_ = np.sqrt(
        spsig.oaconvolve(
            (bl_pad - baseline_) ** 2, np.ones(bl_bin) / bl_bin, mode="same"
        )
    )
    baseline = baseline_[bl_bin // 2 : -(bl_bin // 2)]
    stdev = stdev_[bl_bin // 2 : -(bl_bin // 2)]
    if xdata[0] > xdata[-1]:
        thebaseline = np.interp(xdata, blx[::-1], baseline[::-1])
        thestdev = np.interp(xdata, blx[::-1], stdev[::-1])
    else:
        thebaseline = np.interp(xdata, blx, baseline)
        thestdev = np.interp(xdata, blx, stdev)

    if make_plots:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        ax = axes[0]
        ax.hist(ydata, bins=1000, color="#022851", label="Data")
        ax.legend()
        ax = axes[1]
        ax.hist(yy, bins=1000, color="#ffbf00", label="Baseline")

        ax.legend()

        ax = axes[2]
        ax.plot(xdata, ydata, color="#022851", label="Data")
        ax.plot(xdata, yy, color="#ffbf00", label="Baseline Points")
        ax.plot(xdata, thebaseline, color="#c01230", label="Baseline Average")
        ax.plot(
            xdata,
            thebaseline + sigma * thestdev,
            color="#266041",
            label=f"Baseline + {sigma}$\sigma$",
        )
        ax.set_ylim(0, 10 * np.max(thebaseline + sigma * thestdev))
        ax.legend()

    if make_plots:
        return thebaseline, thestdev, fig, axes
    else:
        return thebaseline, thestdev


def bc_gauss(x, A, x0, w):
    """
    Calculate value of amplitude-normalized Gaussian peak at x:

        f(x) = A * exp(-(x-x0)^2/2w^2)

    Parameters
    ----------
    x : np.array (1D)
        Dependent variable for Gaussian
    A : np.array (1D)
        Amplitude of Gaussian(s)
    x0 : np.array (1D)
        Center position of Gaussian(s) (i.e., f(x0) = A)
    w : np.array (1D)
        Standard deviation of Gaussian(s) (FWHM = 2sqrt(2ln2)*w)

    Returns:

    f : np.array (1D)
        Value of Gaussian function(s) at x values provided
    """

    try:
        A.size
    except AttributeError:
        raise AttributeError("A, x0, and w must be 1D numpy arrays")

    assert A.size == x0.size == w.size
    assert x.ndim == A.ndim == x0.ndim == w.ndim == 1

    return np.sum(
        A[np.newaxis, :]
        * np.exp(
            -(((x[:, np.newaxis] - x0[np.newaxis, :]) / w[np.newaxis, :]) ** 2.0) / 2.0
        ),
        axis=1,
    )


def bc_gauss_list(x, *peaks):
    """
    Calculate value of amplitude-normalized Gaussian peak at x:

        f(x) = A * exp(-(x-x0)^2/2w^2)

    Parameters
    ----------
    x : np.array (1D)
        Dependent variable for Gaussian
    peaks : np.array (N,3)
        array of peak parameters (A, x0, w) for N peaks

    Returns
    -------
    f : np.array (1D)
        Value of Gaussian function(s) at x values provided
    """
    p = np.asarray(peaks)
    pp = p.reshape(p.size // 3, 3)

    return bc_gauss(x, pp[:, 0], pp[:, 1], pp[:, 2])


def locate_peaks(x, y, window=7, order=5, thresh=None):
    """
    Locate the approximate positions of peaks using a smoothed second derivative search.

    The y data array is filtered and differentiated with a Savitzky-Golay filter,
    then peaks are located using scipy.signal argrelmin with an order of half
    the Savitzky-Golay window size. Peaks are only flagged if the second derivative
    is negative; this is to prevent finding spurious "peaks" due to noise on the
    side of a strong peak.
    
    For strong peaks, occasionally the second derivative test locates a peak on each side
    of the main feature instead of one at the center. To prevent this, an additional test
    is performed: if there is no zero-crossing in the second derivative between consecutive peaks,
    and the first derivatives have opposite signs, the two peaks are merged into a single
    feature at their average position.

    Parameters
    ----------
    x : array-like (1D)
        x array for peak detection
    y : array-like (1D)
        y array for peak detection
    window : int (optional)
        Window size for Savitzky-Golay filter. Must be odd, positive, and greater than order. A good choice
        for the window size is comparable to the expected linewidth of a spectral line in points (default: 7)
    order : int (optional)
        Polynomial order for Savitzky-Golay filter. Must be positive (default: 5)
    thresh : None or array-like (1D)
        If provided, must be an array of size y that contains the minimum y value required for a peak (default: None)

    Returns
    -------
    np.array (1D)
        x values of located peaks
    np.array (1D)
        y values of located peaks
    np.array (1D)
        indices of located peaks
    """
    y_pre = y[0 : window // 2]
    y_post = y[-(window // 2) :]
    y_pad = np.concatenate([y_pre, y, y_post])

    coeffs = spsig.savgol_coeffs(window, order, deriv=2, delta=x[1] - x[0])
    coeffs2 = spsig.savgol_coeffs(window, order, deriv=1, delta=x[1] - x[0])
    ysg = spsig.oaconvolve(y_pad, coeffs, mode="same")
    ysg2 = spsig.oaconvolve(y_pad, coeffs2, mode="same")
    yy = ysg[window // 2 : -(window // 2)]
    yy2 = ysg2[window // 2 : -(window // 2)]

    neg = np.where(yy < 0, yy, np.zeros_like(yy))
    locs = spsig.argrelmin(neg, order=window // 2)

    if thresh is None:
        return x[locs], y[locs], np.asarray(locs).flatten()
    else:
        xout_ = x[locs]
        yout_ = y[locs]
        t = thresh[locs]

        p_ = np.asarray(locs).flatten()
        pos = p_[np.where(yout_ > t)]


        if len(pos) == 0:
            return x[pos], y[pos], pos
        if len(pos) == 1:
            return x[pos], y[pos], pos
        newpos = []
        lastavg = False
        for (p1,p2) in zip(pos[:-1],pos[1:]):
            if lastavg:
                lastavg = False
                continue
            if 0.0 not in neg[p1:p2] and yy2[p1]*yy2[p2] < 0.0:
                newpos.append((p1+p2)//2)
                lastavg = True
            else:
                newpos.append(p1)
                lastavg = False

        if not lastavg:
            newpos.append(pos[-1])

        newp = np.asarray(newpos)    

        xout = x[newp]
        yout = y[newp]

        return xout, yout, newp


def find_blocks(m, thresh=0.25, params_per_line=3):
    """
    Finds indices of nearly diagonal blocks in a matrix.

    This function is used to find blocks of correlated peaks using the covariance matrix of a previous fit.
    For this purpose, m is a 2D boolean matrix, generated by comparing the absolute value of the covariance
    matrix to a threshold:

        popt, pcov = spopt.curve_fit(...)
        m = np.abs(pcov)>5e-6
        find_blocks(m)

    The algorithm starts with a (params_per_line x params_per_line) block containing the variance-covariance terms for the first peak,
    then expands the block peak-by-peak. At each step, the fraction of newly-added cross-correlation matrix elements (i.e., new elements
    that do not involve variance-covariance terms for a single peak) is calculated, and if it is less than thresh (default 0.25), the current block
    is taken off the matrix. The algorithm repeats until all blocks are accounted for.

    The return values correspond to the ending indices of each block. These can be used to slice the vector of parameters into sub-blocks
    for independent fitting.

    Parameters
    ----------
    m : np.array (2D, square)
        Matrix containing booleans in an approximate block diagonal form. The matrix dimension must be an integer multiple of params_per_line
    thresh : float
        Threshold for terminating a block based on fraction of new cross-peak terms (default: 0.25)
    params_per_line:
        Number of fitted parameters per peak (default: 3)

    Returns
    -------
    np.array (1D)
        Array of end indices for each block
    """
    blocks = []
    i = params_per_line
    mm = m[:, :]
    while i < mm.shape[0]:
        if (
            np.sum(mm[: i + params_per_line, : i + params_per_line])
            - np.sum(mm[:i, :i])
            - params_per_line**2
        ) / (
            np.size(mm[: i + params_per_line, : i + params_per_line])
            - np.size(mm[:i, :i])
            - params_per_line**2
        ) < thresh:
            blocks.append(i)
            mm = mm[i:, i:]
            i = params_per_line
        else:
            i += params_per_line

    blocks.append(len(mm))
    return np.cumsum(np.asarray(blocks)) // 3


def do_fit(
    func, xfit, yfit, pos, min_A=0.0, x_bound=0.1, w0=0.05, min_w=0.04, max_w=0.15
):
    """
    Fits peaks in (xfit,yfit) to func based on approximate indices given in pos.

    This function is designed to fit multiple peaks to a Gaussian lineshape based
    on 3 parameters per line: A, x0, and w. The baseline is assumed to be 0 (if nonzero,
    subtract a baseline model from yfit before calling this function). Bounds on the
    parameters are used to restrict the scope of the fit, and this may be configured
    (see optional parameters).

    The most important is the parameter w0, which is used as the initial width (standard deviation)
    of the lineshape. From experience, it is better to start with an underestimate than an
    overestimate. The default parameters are set using an assumed width of 0.1 MHz.

    Parameters
    ----------
    func : callable(x,*p)
        Fit function; must take an arbitrary list of parameters in groups of 3.
    xfit : array-like (1D)
        x values to be fitted
    yfit : array-like (1D)
        y values to be fitted
    pos : array-like (1D)
        Indices of approximate peaks in (xfit,yfit) arrays, used as an initial guess for the fit
    min_A : float (optional)
        Minimum value for a peak amplitude in the fit (default 0.0)
    x_bound : float (optional)
        Range to allow x0 to vary during the fit (bound = x0 +/- x_bound)
    w0 : float (optional)
        Initial guess for line width. It is better to underestimate the width, so start with a smaller number (default: 0.05)
    min_w : float (optional)
        Minimum boundary for linewidth term
    max_w : float (optional)
        Maximum boundary for linewidth term

    Returns
    -------
    array-like (1D)
        Optimized parameters [A1,x01,w1,A2,x02,w2,...]
    array-like (2D)
        Variance-covariance matrix
    """
    A = np.asarray([yfit[x] for x in pos])
    x0 = np.asarray([xfit[x] for x in pos])
    w = np.full(A.shape, w0)

    aup = np.full(A.shape, np.max(yfit) * 2)
    alow = np.full(A.shape, min_A)

    xup = x0 + x_bound
    xlow = x0 - x_bound

    wup = np.full(w.shape, max_w)
    wlow = np.full(w.shape, min_w)

    blow = np.asarray([alow, xlow, wlow])
    bup = np.asarray([aup, xup, wup])

    bounds = (blow.T.flatten(), bup.T.flatten())

    args = np.asarray([A, x0, w]).T
    return spopt.curve_fit(func, xfit, yfit, p0=(args.flatten()), bounds=(bounds))


def fit_spectrum(
    xdat,
    ydat,
    width=500,
    overlap=10,
    min_snr=3,
    cov_thresh=5e-6,
    x0_err_limit=0.3,
    blparams={},
    pfparams={},
    fitparams={},
    blockparams={},
):
    """
    Wrapper function that analyzes and fits an entire spectrum to Gaussians.

    The function first estimates the baseline and noise using the skew filter approach implemented
    in estimate_baseline_noise. The blparams dictionary is passed to this function to control its
    behavior.

    Then, xdat and ydat are split into chunks of size width+overlap, and peaks are located
    within the first width points (i.e., if a peak lies in the overlap region, it will be included in the
    following chunk). Peaks are located using the locate_peaks function with the thresh parameter set to
    min_snr*stdev, where stdev comes from estimate_baseline_noise. Additional peak-locating parameters
    can be passed using the pfparams dictionary.

    Within each chunk, the spectrum is initially fit using the do_fit function. Optional arguments to
    do_fit can be passed in the fitparams dict. After the initial fit, the variance-covariance matrix is
    compared with cov_thresh element-by-element and then broken into blocks using find_blocks (optional parameters
    passed through blockparams). Each block is then fit, and if any peaks have an x0 error greater than x0_err_limit,
    the line with the greatest error is removed and the fit repeated.

    The optimized parameter values and 1 sigma uncertaintes (computed without correlation!) are returned in a pandas
    Dataframe, along with the baseline and the snr threshold.

    Parameters
    ----------
    xdat : array-like (1D)
        x data values for the fit
    ydat : array-like (1D)
        y data values to be fit
    width : int (optional)
        Number of data points to include in each chunk. The effect of this parameter on speed and performance is difficult to predict (default: 500)
    overlap : int (optional)
        Number of "extra" data points to include in each chunk. These points are not included in the peak finding for the current chunk (default: 10)
    min_snr : float (optional)
        Minimum signal-to-noise ratio to be used in the peak detection (default: 3)
    cov_thresh : float (optiona)
        Threshold for converting absolute value of the covariance matrix to booleans. Making this number larger will fit more peaks independently,
        potentially compromising performance for closely-spaced peaks. Making it smaller will fit more peaks together, which is more likely to result
        in correlated errors. (default: 5e-6)
    x0_err_limit : float (optional)
        Maximum allowed error on the line center. Lines with error exceeding this will be thrown out, and the spectrum refitted.
    blparams : dict (optional)
        Parameters passed to estimate_baseline_noise
    pfparams : dict (optional)
        Parameters passed to locate_peaks
    fitparams : dict (optional)
        Parameters passed to do_fit
    blockparams : dict (optional)
        Parameters passed to find_blocks

    Returns
    -------
    np.array (1D, size of xdat)
        Baseline
    np.array (1D, size of xdat)
        Peak detection threshold
    pandas.Dataframe
        Optimized peak parameters and uncertainties
    """
    bl, sd = estimate_baseline_noise(xdat, ydat, **blparams)
    all_params = []
    all_errs = []

    for i in range(len(xdat) // width + 1):
        ipos = i * width

        xfit = xdat[ipos : ipos + width + overlap]
        blfit = bl[ipos : ipos + width + overlap]
        yfit = ydat[ipos : ipos + width + overlap] - blfit
        sigfit = sd[ipos : ipos + width + overlap]

        if len(xfit) < overlap:
            break

        xmin = np.min(xfit)
        xmax = np.max(xfit)
        xp, yp, pp = locate_peaks(
            xfit[:-overlap],
            yfit[:-overlap],
            thresh=min_snr * sigfit[:-overlap],
            **pfparams,
        )
        curr_pos = pp

        if len(curr_pos) > 0:
            while True:
                popt, pcov = do_fit(bc_gauss_list, xfit, yfit, pp, **fitparams)

                opt_params = popt.reshape(len(popt) // 3, 3)
                opt_err = np.sqrt(np.diag(pcov).reshape(len(popt) // 3, 3))

                tot_err = np.sqrt(
                    (opt_err[:, 1] / opt_params[:, 2]) ** 2
                    + (opt_err[:, 2] / opt_params[:, 2]) ** 2
                )
                keep = np.isfinite(tot_err) & np.invert(np.isnan(tot_err))

                if False in keep:
                    print(f"Removing indeterminate lines")
                    curr_pos = curr_pos[keep]
                    if len(curr_pos) == 0:
                        break
                    continue

                # use structure of covariance matrix to break fit into pieces
                m = np.abs(pcov) > cov_thresh
                cutoffs = find_blocks(m, **blockparams)

                if len(cutoffs) > 1:
                    opt_params = []
                    opt_err = []
                    last_cutoff = 0
                    for c in cutoffs:
                        p = curr_pos[last_cutoff:c]
                        last_cutoff = c
                        (fi,) = np.where(curr_pos == p[0])[0]
                        (li,) = np.where(curr_pos == p[-1])[0]
                        if fi > 0:
                            start_index = (curr_pos[fi - 1] + curr_pos[fi]) // 2
                        else:
                            start_index = 0
                        if li == len(curr_pos) - 1:
                            end_index = len(xfit) - 1
                        else:
                            end_index = (curr_pos[li] + curr_pos[li + 1]) // 2

                        while True:
                            if len(p) == 0:
                                break
                            pt, pv = do_fit(
                                bc_gauss_list,
                                xfit[start_index:end_index],
                                yfit[start_index:end_index],
                                p - start_index,
                            )
                            op = pt.reshape(len(pt) // 3, 3)
                            oe = np.sqrt(np.diag(pv).reshape(len(pt) // 3, 3))

                            # tot_err = np.sqrt((oe[:,1]/op[:,2])**2+(oe[:,2]/op[:,2])**2)
                            if np.any(oe[:, 1] > x0_err_limit):
                                i = np.argmax(oe[:, 1] - op[:, 2])
                                p = np.delete(p, i)
                                print(f"Removing a line and retrying...")
                                continue
                            break

                        if len(opt_params) == 0:
                            opt_params = op
                            opt_err = oe
                        else:
                            opt_params = np.concatenate([opt_params, op], axis=0)
                            opt_err = np.concatenate([opt_err, oe], axis=0)

                break

            if len(all_params) == 0:
                all_params = opt_params
                all_errs = opt_err
            else:
                all_params = np.concatenate([all_params, opt_params], axis=0)
                all_errs = np.concatenate([all_errs, opt_err], axis=0)

    df = pd.DataFrame(
        np.concatenate([all_params, all_errs], axis=1),
        columns=["A", "$x_0$", "w", "$\sigma$(A)", "$\sigma$($x_0$)", "$\sigma$(w)"],
    )
    return bl, sd * min_snr, df


def plot_fit(x, y, bl, sd, df, peaks_per_chunk=10, interp_factor=10):
    """
    Generates a plot showing the fit results.

    To prevent memory issues with large 2D arrays, the implementation is based on a python loop
    over sets of peaks (given by peaks_per_chunk). By default, the resolution of the fit is 10
    times greater than the resolution of the underlying data; this can be changed with interp_factor.

    Keep in mind that each chunk generates a (len(x)*interp_factor)x(peaks_per_chunk) array for
    the calculation of the model. Increasing peaks_per_chunk will increase the speed of the calculation
    at the expense of additional memory usage.

    Parameters
    ----------
    x : array-like
        x data values
    y : array-like
        y data values
    bl : array-like
        baseline at each x point (returned by fit_spectrum)
    sd : array-like
        peak threshold at each x point (returned by fit_spectrum)
    df : pandas.DataFrame
        DataFrame containing peak parameters and errors (returned by fit_spectrum)
    peaks_per_chunk : int (optional)
        Number of peaks to include per iteration when calculating the total model (default: 10)
    interp_factor : int
        Factor by which to increase the resolution of x for plotting the fit (default: 10)

    Returns
    -------
    matplotlib.Figure
        Figure object containing plot
    matplotlib.Axes
        Axes object containing plot
    """
    model_x = np.linspace(np.min(x), np.max(x), interp_factor * len(x))

    if x[0] > x[-1]:
        opt_bl = np.interp(df["$x_0$"].to_numpy(), x[::-1], bl[::-1])
        nicebl = np.interp(model_x, x[::-1], bl[::-1])
    else:
        opt_bl = np.interp(df["$x_0$"].to_numpy(), xf, bl)
        nicebl = np.interp(model_x, x, bl)

    model_y = nicebl

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(x, y, label="Data")

    for i in range(0, len(df), peaks_per_chunk):
        p = df[["A", "$x_0$", "w"]][i : i + peaks_per_chunk].to_numpy()
        model_y += bc_gauss_list(model_x, *p)

    # for p in opt_params:
    #     ax.plot(model_x,bc_gauss_list(nicex,*p)+nicebl,color='#00000066',linestyle='dotted')
    ax.plot(model_x, model_y, label="Fit")
    ax.plot(x, bl + sd, color="#00000033", linestyle="dotted", label="Threshold")

    ax.errorbar(
        df["$x_0$"],
        df["A"] + opt_bl,
        xerr=df["$\sigma$($x_0$)"],
        yerr=df["$\sigma$(A)"],
        fmt="ro",
        markersize=3,
        zorder=10,
        label="Peaks",
    )
    ax.legend()

    return fig, ax
