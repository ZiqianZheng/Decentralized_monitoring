import numpy as np
import multiprocessing
import time


def bayes_cdf(x_list_sort, x):
    """
    This function estimates the cdf of continuous and discrete distributions.
    Input:
        x_list_sort: sorted list
        x: new sample
    Output:
        Bayesian estimation of the cdf
    """
    if np.ndim(x_list_sort) == 1:
        return (np.searchsorted(x_list_sort, x) + 1)/(len(x_list_sort)+2)

    m, n = x_list_sort.shape
    cdf = np.zeros(m)
    for i in range(m):
        cdf[i] = np.searchsorted(x_list_sort[i], x[i])
    return (cdf+1)/(n+2)


def data_generate(data_type, shape, gamma_shape=5, gamma_scale=1, log_mean=1, log_sig=0.5, df_t=3):
    """
    Generate random variables with different shape.
    Input:
        data_type: An integer represents the distribution.
        shape: A tuple, the shape of generated data.
    Output:
        Random variables with given distribution and shape.
    """
    if data_type == 1:
        return np.random.normal(0, 1, shape)
    if data_type == 2:
        return np.random.standard_t(df_t, shape) / np.sqrt(df_t / (df_t - 2))
    if data_type == 3:
        return np.random.exponential(1, shape) - 1
    if data_type == 4:
        return (np.random.lognormal(log_mean, log_sig, shape) - np.exp(log_mean + log_sig ** 2 / 2)) \
               / np.sqrt(np.exp(log_mean * 2 + log_sig ** 2) * (np.exp(log_sig ** 2) - 1))
    if data_type == 5:
        return (np.random.gamma(gamma_shape, gamma_scale, shape) - gamma_shape * gamma_scale) \
               / np.sqrt(gamma_shape * gamma_scale ** 2)


def rl0_device(n_stream, k, h, r):
    """
    Simulate device RL based on the theoretical distribution.
    Input:
        n_stream: Number of data stream.
        k: Allowance parameter.
        h: Threshold.
        r: Parameters in top-r method.
    Output:
        Simulated RL
    """
    W1 = np.zeros(n_stream)
    W2 = np.zeros(n_stream)
    W = np.zeros(n_stream)
    RL = 0
    while np.sum(np.sort(W)[-r:]) <= h:
        RL += 1
        x = np.random.random(n_stream)
        W1 = W1 - np.log(x) - k
        W2 = W2 - np.log(1-x) - k
        W1[W1 < 0] = 0
        W2[W2 < 0] = 0
        W = np.maximum(W1, W2)
    return RL


def arl0_device(n_stream, k, h, r, N, n_thread=15, verbose=0):
    """
    Simulate device RL based on the theoretical distribution.
    Input:
        n_stream: Number of data stream.
        k: Allowance parameter.
        h: Threshold.
        r: Parameters in top-r method.
        N: Number of replications.
        n_thread: Number of multiprocess threads.
        verbose: Control how much information are printed.
    Output:
        Simulated ARL
    """
    pool = multiprocessing.Pool(n_thread)
    if verbose > 1:
        print('Start Simulation!')
        print('Number of simulation runs: ' + str(N) + '. Number threads: ' + str(n_thread))
    tasks = [[n_stream, k, h, r] for _ in range(N)]
    RL_list = pool.starmap(rl0_device, tasks)
    pool.close()
    mu, sigma = np.mean(RL_list), np.std(RL_list)/np.sqrt(N)
    if verbose > 0:
        print('{:.2f}\t{:.2f}\t{:.2f}'.format(h, mu, sigma))
    return mu, sigma


def arl0_function(offline_list, h):
    """
    Interpolation of the ARL0 function based on offline simulation results.
    Input:
        offline_list: Offline simulation results of A(h) with h = 0, 0.5, 1, ..., len(offline_list)*0.5-0.5.
    Output:
        A_hat(h)
    """
    n = len(offline_list)
    if h < n * 0.5 - 0.5:
        i = int(h/0.5)
    else:
        i = n - 2
    return (2*h-i)*offline_list[i+1] + (i+1-2*h)*offline_list[i]


def rl_proposed(n_device, n_observe, distribution_type, n_ic, r_device, r_server, ARL_dict, h, k=1.2, delta=5,
                n_shift=None, size_shift=1, t_shift=50, data_ic_file=None, data_ol_file=None, debug=False,
                max_rl=None):
    """
    Simulate RL of the proposed scheme.
    Input:
        n_device: Number of edge devices.
        n_observe: Observable devices.
        distribution_type: List of lists, representing the distribution of each data stream.
        n_ic: Number of in-control samples.
        r_device: Top-r parameter used by each edge device.
        r_server: Top-r parameter used by the central server.
        ARL_dict: Dictionary of offline simulated ARL function.
        h: Threshold.
        k: Allowance parameter.
        delta: Compensation parameter.
        n_shift: Number of shifted data streams when the device is out of control.
        size_shift: Magnitude of the mean shift.
        t_shift: Shifted time.
        data_ic_file: In control data
        data_ol_file: Online monitoring data.
    Output:
        Simulated RL.
    """
    n_stream = [len(temp) for temp in distribution_type]
    n_stream_sum = np.sum(n_stream)
    shift_lookup = [[False]*n_stream[i] for i in range(n_device)]
    shift_direction = 1 if np.random.random(1) > 0.5 else -1
    shift_flag = n_shift is not None
    if max_rl is None:
        max_rl = 10**10
    shift_idx = []
    if shift_flag:
        shift_idx = np.sort(np.random.choice(n_stream_sum, n_shift, replace=False))
        i, offset = 0, 0
        for idx in shift_idx:
            while idx-offset >= n_stream[i]:
                i += 1
                offset += n_stream[i]
            shift_lookup[i][idx-offset] = True

    data_ic = []
    data_ol = []
    offset1 = 0
    offset2 = 0
    for i in range(n_device):
        if data_ic_file is None:
            data_ic_temp = np.zeros((n_stream[i], n_ic))
            for j in range(n_stream[i]):
                data_ic_temp[j] = data_generate(data_type=distribution_type[i][j], shape=n_ic)
        else:
            data_ic_temp = data_ic_file[offset1:offset1+n_stream[i]]
            offset1 += n_stream[i]
        if data_ol_file is not None:
            data_ol_temp = data_ol_file[offset2:offset2+n_stream[i]]
            offset2 += n_stream[i]
            data_ol += [data_ol_temp]
        data_ic_temp = np.sort(data_ic_temp, axis=1)
        data_ic += [data_ic_temp]

    W1 = [np.zeros(n_stream[i]) for i in range(n_device)]
    W2 = [np.zeros(n_stream[i]) for i in range(n_device)]
    W = [np.zeros(n_stream[i]) for i in range(n_device)]
    V, U, U_hat, tau = np.zeros(n_device), np.zeros(n_device), np.zeros(n_device), np.zeros(n_device)
    RL = 0
    Observe_history = []
    while np.sum(np.sort(U_hat)[-r_server:]) <= h and RL < max_rl:
        E = U_hat + tau*delta
        Observe = np.argsort(E)[-n_observe:]
        Observe_history.append(Observe)
        tau += 1
        for i in range(n_device):
            for j in range(n_stream[i]):
                if data_ol_file is None:
                    x = data_generate(distribution_type[i][j], 1)[0]
                    if shift_flag and shift_lookup[i][j] and RL >= t_shift:
                        x += size_shift * shift_direction
                else:
                    x = data_ol[i][j, RL]
                cdf = bayes_cdf(data_ic[i][j], x)
                W1[i][j] = max(W1[i][j] - np.log(cdf) - k, 0)
                W2[i][j] = max(W2[i][j] - np.log(1-cdf) - k, 0)
            W[i] = np.maximum(W1[i], W2[i])
            V[i] = np.sum(np.sort(W[i])[-r_device[i]:])
            U[i] = arl0_function(ARL_dict[(n_stream[i], r_device[i])], V[i])
        for i in Observe:
            U_hat[i] = U[i]
            tau[i] = 0
        if shift_flag and RL < t_shift and np.sum(np.sort(U_hat)[-r_server:]) > h:
            W1 = [np.zeros(n_stream[i]) for i in range(n_device)]
            W2 = [np.zeros(n_stream[i]) for i in range(n_device)]
            W = [np.zeros(n_stream[i]) for i in range(n_device)]
            V, U, U_hat, tau = np.zeros(n_device), np.zeros(n_device), np.zeros(n_device), np.zeros(n_device)
        RL += 1
        if data_ol_file is not None and RL == len(data_ol[0][0]):
            break
        elif data_ol_file is not None and RL < 30 and np.sum(np.sort(U_hat)[-r_server:]) > h:
            W1 = [np.zeros(n_stream[i]) for i in range(n_device)]
            W2 = [np.zeros(n_stream[i]) for i in range(n_device)]
            W = [np.zeros(n_stream[i]) for i in range(n_device)]
            V, U, U_hat, tau = np.zeros(n_device), np.zeros(n_device), np.zeros(n_device), np.zeros(n_device)
    # for i in range(n_device):
    #     for j in range(n_stream[i]):
    #         if shift_lookup[i][j]:
    #             print(W1[i][j], W2[i][j], W[i][j])
    if n_shift is None:
        return (RL, Observe_history) if debug else RL
    else:
        return (RL - t_shift, Observe_history, shift_idx) if debug else RL


def arl_proposed(n_device, n_observe, distribution_type, n_ic, r_device, r_server, ARL_dict, h, k=1.2, delta=5,
                 n_shift=None, size_shift=1, t_shift=20, n_thread=15, N=10000, verbose=1):
    """
    Simulate ARL based of the proposed method.
    Input:
        n_thread: Number of thread.
        verbose: Control how much information are printed.
        See function 'rl' for the rest of the arguments.
    Output:
        Simulated ARL
    """
    pool = multiprocessing.Pool(n_thread)
    if verbose > 1:
        print('Start Simulation!')
        print('Number of simulation runs: ' + str(N) + '. Number threads: ' + str(n_thread))
    tasks = [[n_device, n_observe, distribution_type, n_ic, r_device, r_server, ARL_dict, h, k, delta, n_shift,
              size_shift, t_shift] for _ in range(N)]
    RL_list = pool.starmap(rl_proposed, tasks)
    pool.close()
    mu, sigma = np.mean(RL_list), np.std(RL_list)/np.sqrt(N)
    if verbose > 0:
        print('{:.2f}\t{:.2f}\t{:.2f}'.format(h, mu, sigma))
    return mu, sigma


def arl_proposed_data(n_device, n_observe, distribution_type, n_ic, r_device, r_server, ARL_dict, h, k=1.2, delta=5,
                      n_shift=None, size_shift=1, t_shift=20, n_thread=15, N=10000, data_ic_file=None,
                      data_ol_file=None):
    """
    Simulate ARL based of the proposed method with given data.
    Input:
        n_thread: Number of thread.
        verbose: Control how much information are printed.
        See function 'rl' for the rest of the arguments.
    Output:
        Simulated ARL
    """
    pool = multiprocessing.Pool(n_thread)
    tasks = [[n_device, n_observe, distribution_type, n_ic, r_device, r_server, ARL_dict, h, k, delta, n_shift,
              size_shift, t_shift, data_ic_file, data_ol_file[i]] for i in range(N)]
    RL_list = pool.starmap(rl_proposed, tasks)
    pool.close()
    return RL_list


def validate_property_2(n_device, n_observe, distribution_type, n_ic, r_device, r_server, ARL_dict, h, k=1.2, delta=5,
                        n_shift=2, size_shift=1, t_shift=20, n_thread=15, N=10000):
    """
    Validate Property 2 of the proposed monitoring scheme.
    """
    pool = multiprocessing.Pool(n_thread)
    tasks = [[n_device, n_observe, distribution_type, n_ic, r_device, r_server, ARL_dict, h, k, delta, n_shift,
              size_shift, t_shift, None, None, True] for _ in range(N)]
    result_list = pool.starmap(rl_proposed, tasks)
    return result_list


def rl_proposed_NT(n_device, n_observe, distribution_type, n_ic, r_device, r_server, h, k=1.2, delta=5,
                   n_shift=None, size_shift=1, t_shift=50):
    """
    Simulate RL of the proposed scheme without time transformation.
    Input:
        n_device: Number of edge devices.
        n_observe: Observable devices.
        distribution_type: List of lists, representing the distribution of each data stream.
        n_ic: Number of in-control samples.
        r_device: Top-r parameter used by each edge device.
        r_server: Top-r parameter used by the central server.
        ARL_dict: Dictionary of offline simulated ARL function.
        h: Threshold.
        k: Allowance parameter.
        delta: Compensation parameter.
        n_shift: Number of shifted data streams when the device is out of control.
        size_shift: Magnitude of the mean shift.
        t_shift: Shifted time.
    Output:
        Simulated RL.
    """
    n_stream = [len(temp) for temp in distribution_type]
    n_stream_sum = np.sum(n_stream)
    shift_lookup = [[False]*n_stream[i] for i in range(n_device)]
    shift_direction = 1 if np.random.random(1) > 0.5 else -1
    shift_flag = n_shift is not None
    if shift_flag:
        shift_idx = np.sort(np.random.choice(n_stream_sum, n_shift, replace=False))
        i, offset = 0, 0
        for idx in shift_idx:
            while idx-offset >= n_stream[i]:
                i += 1
                offset += n_stream[i]
            shift_lookup[i][idx-offset] = True

    data_ic = []
    for i in range(n_device):
        data_ic_temp = np.zeros((n_stream[i], n_ic))
        for j in range(n_stream[i]):
            data_ic_temp[j] = data_generate(data_type=distribution_type[i][j], shape=n_ic)
        data_ic_temp = np.sort(data_ic_temp, axis=1)
        data_ic += [data_ic_temp]

    W1 = [np.zeros(n_stream[i]) for i in range(n_device)]
    W2 = [np.zeros(n_stream[i]) for i in range(n_device)]
    W = [np.zeros(n_stream[i]) for i in range(n_device)]
    V, U_hat, tau = np.zeros(n_device), np.zeros(n_device), np.zeros(n_device)
    RL = 0
    while np.sum(np.sort(U_hat)[-r_server:]) <= h:
        RL += 1
        E = U_hat + tau*delta
        Observe = np.argsort(E)[-n_observe:]
        tau += 1
        for i in range(n_device):
            for j in range(n_stream[i]):
                x = data_generate(distribution_type[i][j], 1)[0]
                if shift_flag and shift_lookup[i][j] and RL >= t_shift:
                    x += size_shift * shift_direction
                cdf = bayes_cdf(data_ic[i][j], x)
                W1[i][j] = max(W1[i][j] - np.log(cdf) - k, 0)
                W2[i][j] = max(W2[i][j] - np.log(1-cdf) - k, 0)
            W[i] = np.maximum(W1[i], W2[i])
            V[i] = np.sum(np.sort(W[i])[-r_device[i]:])
        for i in Observe:
            U_hat[i] = V[i]
            tau[i] = 0
        if shift_flag and RL < t_shift and np.sum(np.sort(U_hat)[-r_server:]) > h:
            W1 = [np.zeros(n_stream[i]) for i in range(n_device)]
            W2 = [np.zeros(n_stream[i]) for i in range(n_device)]
            W = [np.zeros(n_stream[i]) for i in range(n_device)]
            V, U_hat, tau = np.zeros(n_device), np.zeros(n_device), np.zeros(n_device)
    if n_shift is None:
        return RL
    else:
        return RL - t_shift


def arl_proposed_NT(n_device, n_observe, distribution_type, n_ic, r_device, r_server, h, k=1.2, delta=5,
                    n_shift=None, size_shift=1, t_shift=50, n_thread=15, N=10000, verbose=1):
    """
    Simulate ARL based of the proposed method without time transformation.
    Input:
        n_thread: Number of thread.
        verbose: Control how much information are printed.
        See function 'rl' for the rest of the arguments.
    Output:
        Simulated ARL
    """
    pool = multiprocessing.Pool(n_thread)
    if verbose > 1:
        print('Start Simulation!')
        print('Number of simulation runs: ' + str(N) + '. Number threads: ' + str(n_thread))
    tasks = [[n_device, n_observe, distribution_type, n_ic, r_device, r_server, h, k, delta, n_shift, size_shift,
              t_shift] for _ in range(N)]
    RL_list = pool.starmap(rl_proposed_NT, tasks)
    pool.close()
    mu, sigma = np.mean(RL_list), np.std(RL_list)/np.sqrt(N)
    if verbose > 0:
        print('{:.2f}\t{:.2f}\t{:.2f}'.format(h, mu, sigma))
    return mu, sigma


def interval_indicator(x, ic_interval):
    """
    [Local statistics construction for QUANTS]This function calculates the interval indicator. A+ in the paper QUANTS.
    Input:
        x: New sample
        ic_interval: d-1 interval boundaries.
    Output:
        A np array with length of d-1. Represent the cumulative interval indicator.
    """
    d1 = len(ic_interval)
    indicator = np.zeros(d1)
    for i in range(d1):
        if x > ic_interval[i]:
            indicator[i] = 1
        else:
            break
    return indicator


def interval_indicator_r(theta):
    """
    [Local statistics construction, QUANTS]This function randomly generate the interval indicator.
    A+ in the paper QUANTS.
    Input:
    theta: Parameter of QUANTS.
    Output:
    A np array with length of d-1. Represent the cumulative interval indicator.
    """
    theta_normalize = theta/np.sum(theta)
    d = len(theta)
    indicator = np.zeros(d-1)
    p = np.random.random()
    i = 0
    while p - theta_normalize[i] >= 0:
        p -= theta_normalize[i]
        indicator[i] = 1
        i += 1
    return indicator


def rl_quants(n_observe, distribution_type, n_ic, r, h, k=1, d=10, n_shift=None, size_shift=1, t_shift=30,
              data_ic_file=None, data_ol_file=None):
    """
    Simulate RL of quants method.
    Input:
        n_observe: Observable streams.
        distribution_type: List, representing the distribution of each data stream.
        n_ic: Number of in-control samples.
        r: Top-r parameter used by the central server.
        h: Threshold.
        k: Allowance parameter.
        d: Number of intervals in QUANTS method.
        n_shift: Number of shifted data streams when the device is out of control.
        size_shift: Magnitude of the mean shift.
        t_shift: Shifted time.
    Output:
        Simulated RL.
    """
    n_stream = len(distribution_type)
    shift_lookup = [False] * n_stream
    shift_direction = 1 if np.random.random(1) > 0.5 else -1
    shift_flag = n_shift is not None
    if shift_flag:
        shift_idx = np.random.choice(n_stream, n_shift, replace=False)
        for i in shift_idx:
            shift_lookup[i] = True
    quantiles = np.zeros((n_stream, d-1))
    for i in range(n_stream):
        if data_ic_file is None:
            data_temp = data_generate(distribution_type[i], shape=n_ic)
        else:
            data_temp = data_ic_file[i]
        quantiles[i] = np.quantile(data_temp, (np.arange(d - 1) + 1) / d)

    g_neg = (np.arange(d - 1) + 1) / d
    g_pos = 1 - g_neg
    W = np.zeros(n_stream)
    S_pos1, S_neg1 = np.zeros((n_stream, d-1)), np.zeros((n_stream, d-1))
    S_pos2, S_neg2 = np.zeros((n_stream, d-1)), np.zeros((n_stream, d-1))
    theta = np.ones((n_stream, d))
    RL = 0
    while np.sum(np.sort(W)[-r:]) <= h:
        A_pos = np.zeros((n_stream, d-1))
        observe = np.argsort(W)[-n_observe:]
        for i in range(n_stream):
            if i in observe:
                if data_ol_file is not None:
                    x = data_ol_file[i, RL]
                else:
                    x = data_generate(distribution_type[i], 1)[0]
                    if shift_flag and shift_lookup[i] and RL >= t_shift:
                        x += size_shift * shift_direction
                A_pos[i] = interval_indicator(x, quantiles[i])
                interval = int(np.sum(A_pos[i]))
                theta[i][interval] += 1
            else:
                A_pos[i] = interval_indicator_r(theta[i])
        A_neg = 1 - A_pos
        C_pos = np.sum((S_pos1 + A_pos - S_pos2 - g_pos) ** 2 / (S_pos2 + g_pos), axis=1)
        C_neg = np.sum((S_neg1 + A_neg - S_neg2 - g_neg) ** 2 / (S_neg2 + g_neg), axis=1)
        W_pos, W_neg = C_pos - k, C_neg - k
        W_pos[W_pos < 0] = 0
        W_neg[W_neg < 0] = 0
        W = np.maximum(W_pos, W_neg)
        if shift_flag and RL < t_shift and np.sum(np.sort(W)[-r:]) > h:
            W = np.zeros(n_stream)
            S_pos1, S_neg1 = np.zeros((n_stream, d-1)), np.zeros((n_stream, d-1))
            S_pos2, S_neg2 = np.zeros((n_stream, d-1)), np.zeros((n_stream, d-1))
            theta = np.ones((n_stream, d))
        else:
            S_pos1 = (S_pos1 + A_pos) * W_pos[:, np.newaxis] / C_pos[:, np.newaxis]
            S_pos2 = (S_pos2 + g_pos) * W_pos[:, np.newaxis] / C_pos[:, np.newaxis]
            S_neg1 = (S_neg1 + A_neg) * W_neg[:, np.newaxis] / C_neg[:, np.newaxis]
            S_neg2 = (S_neg2 + A_neg) * W_neg[:, np.newaxis] / C_neg[:, np.newaxis]
        RL += 1
        if data_ol_file is not None and RL == len(data_ol_file[0]):
            break
        elif data_ol_file is not None and RL < 30 and np.sum(np.sort(W)[-r:]) > h:
            W = np.zeros(n_stream)
            S_pos1, S_neg1 = np.zeros((n_stream, d-1)), np.zeros((n_stream, d-1))
            S_pos2, S_neg2 = np.zeros((n_stream, d-1)), np.zeros((n_stream, d-1))
            theta = np.ones((n_stream, d))
    if n_shift is None:
        return RL
    else:
        return RL - t_shift


def arl_quants(n_observe, distribution_type, n_ic, r, h, k=1, d=10, n_shift=None, size_shift=1, t_shift=30,
               n_thread=15, N=10000, verbose=1):
    """
    Simulate ARL based of the QUANTS method.
    Input:
        n_thread: Number of thread.
        verbose: Control how much information are printed.
        See function 'rl_quants' for the rest of the arguments.
    Output:
        Simulated ARL
    """
    pool = multiprocessing.Pool(n_thread)
    if verbose > 1:
        print('Start Simulation!')
        print('Number of simulation runs: ' + str(N) + '. Number threads: ' + str(n_thread))
    tasks = [[n_observe, distribution_type, n_ic, r, h, k, d, n_shift, size_shift, t_shift] for _ in range(N)]
    RL_list = pool.starmap(rl_quants, tasks)
    pool.close()
    mu, sigma = np.mean(RL_list), np.std(RL_list)/np.sqrt(N)
    if verbose > 0:
        print('{:.2f}\t{:.2f}\t{:.2f}'.format(h, mu, sigma))
    return mu, sigma


def arl_quants_data(n_observe, distribution_type, n_ic, r, h, k=1, d=10, n_shift=None, size_shift=1, t_shift=30,
                    n_thread=15, N=10000, data_ic_file=None, data_ol_file=None):
    """
    Simulate ARL based of the QUANTS method with given data.
    Input:
        n_thread: Number of thread.
        verbose: Control how much information are printed.
        See function 'rl_quants' for the rest of the arguments.
    Output:
        Simulated ARL
    """
    pool = multiprocessing.Pool(n_thread)
    tasks = [[n_observe, distribution_type, n_ic, r, h, k, d, n_shift, size_shift, t_shift, data_ic_file,
              data_ol_file[i]] for i in range(N)]
    RL_list = pool.starmap(rl_quants, tasks)
    pool.close()
    return RL_list


def rl_qh(distribution_type, n_ic, h, k=1, n_shift=None, size_shift=1, t_shift=30):
    """
    Simulate RL of QH01.
    Input:
        distribution_type: List, representing the distribution of each data stream.
        n_ic: Number of in-control samples.
        h: Threshold.
        k: Allowance parameter.
        n_shift: Number of shifted data streams when the device is out of control.
        size_shift: Magnitude of the mean shift.
        t_shift: Shifted time.
    Output:
        Simulated RL.
    """
    n_stream = len(distribution_type)
    shift_lookup = [False] * n_stream
    shift_direction = 1 if np.random.random(1) > 0.5 else -1
    shift_flag = n_shift is not None
    if shift_flag:
        shift_idx = np.random.choice(n_stream, n_shift, replace=False)
        for i in shift_idx:
            shift_lookup[i] = True
    data_ic = np.zeros((n_stream, n_ic))
    for i in range(n_stream):
        data_ic[i] = data_generate(distribution_type[i], shape=n_ic)
    idx_down = np.argmin(data_ic, axis=0)
    idx_up = np.argmax(data_ic, axis=0)
    del data_ic
    p_down = np.zeros(n_stream)
    p_up = np.zeros(n_stream)
    for i in range(n_stream):
        p_down[idx_down[i]] += 1
        p_up[idx_up[i]] += 1
    p_down += 1e-4
    p_up += 1e-4
    p_down = p_down/n_ic
    p_up = p_up/n_ic

    RL = 0
    S_up1, S_up2, S_down1, S_down2 = np.zeros(n_stream), np.zeros(n_stream), np.zeros(n_stream), np.zeros(n_stream)
    y_up, y_down = 0, 0
    while max(y_up, y_down) <= h:
        RL += 1
        data = np.zeros(n_stream)
        for i in range(n_stream):
            data[i] = data_generate(distribution_type[i], 1)[0]
            if shift_flag and shift_lookup[i] and RL >= t_shift:
                data[i] += size_shift * shift_direction
        ksi_up, ksi_down = np.zeros(n_stream), np.zeros(n_stream)
        ksi_down[np.argmin(data)] = 1
        ksi_up[np.argmax(data)] = 1

        C_down = np.sum((S_down1 + ksi_down - S_down2 - p_down) ** 2 / (S_down2 + p_down))
        y_down = max(0, C_down-k)
        C_up = np.sum((S_up1 + ksi_up - S_up2 - p_up) ** 2 / (S_up2 + p_up))
        y_up = max(0, C_up-k)

        if shift_flag and RL < t_shift and max(y_up, y_down) > h:
            S_up1, S_up2, = np.zeros(n_stream), np.zeros(n_stream)
            S_down1, S_down2 = np.zeros(n_stream), np.zeros(n_stream)
            y_up, y_down = 0, 0
        else:
            S_down1 = (S_down1 + ksi_down) * y_down / C_down
            S_down2 = (S_down2 + p_down) * y_down / C_down
            S_up1 = (S_up1 + ksi_up) * y_up / C_up
            S_up2 = (S_up2 + p_up) * y_up / C_up
    print(y_up, y_down)
    if not shift_flag:
        return RL
    else:
        return RL - t_shift


def arl_qh(distribution_type, n_ic, h, k=1, n_shift=None, size_shift=1, t_shift=30, n_thread=15, N=10000, verbose=1):
    """
    Simulate ARL based of QH01.
    Input:
        n_thread: Number of thread.
        verbose: Control how much information are printed.
        See function 'rl_quants' for the rest of the arguments.
    Output:
        Simulated ARL
    """
    pool = multiprocessing.Pool(n_thread)
    if verbose > 1:
        print('Start Simulation!')
        print('Number of simulation runs: ' + str(N) + '. Number threads: ' + str(n_thread))
    tasks = [[distribution_type, n_ic, h, k, n_shift, size_shift, t_shift] for _ in range(N)]
    RL_list = pool.starmap(rl_qh, tasks)
    pool.close()
    mu, sigma = np.mean(RL_list), np.std(RL_list)/np.sqrt(N)
    if verbose > 0:
        print('{:.2f}\t{:.2f}\t{:.2f}'.format(h, mu, sigma))
    return mu, sigma


def spatial_rank(data_ic, x):
    """
    Calculate the spatial rank based on in-control data and a new observation.
    """
    temp = x[:, np.newaxis] - data_ic
    temp_sum = np.sum(temp**2, axis=0)
    temp_sum[temp_sum == 0] = 1e-5
    temp = temp/temp_sum
    return np.mean(temp, axis=1)


def rl_SRAS_full(distribution_type, n_ic, lambda_, h, n_shift=None, size_shift=1, t_shift=30):
    """
    Simulate RL of SRAS with full observations.
    Input:
        distribution_type: List, representing the distribution of each data stream.
        n_ic: Number of in-control samples.
        lambda_: Parameter of the EWMA statistic.
        h: Threshold.
        n_shift: Number of shifted data streams when the device is out of control.
        size_shift: Magnitude of the mean shift.
        t_shift: Shifted time.
    Output:
        Simulated RL.
    """
    n_stream = len(distribution_type)
    shift_lookup = [False] * n_stream
    shift_direction = 1 if np.random.random(1) > 0.5 else -1
    shift_flag = n_shift is not None
    if shift_flag:
        shift_idx = np.random.choice(n_stream, n_shift, replace=False)
        for i in shift_idx:
            shift_lookup[i] = True
    data_ic = np.zeros((n_stream, n_ic))
    for i in range(n_stream):
        data_ic[i] = data_generate(distribution_type[i], shape=n_ic)

    RL = 0
    v = np.zeros(n_stream)
    while np.sum(v**2)*10**4 <= h:
        RL += 1
        data = np.zeros(n_stream)
        for i in range(n_stream):
            data[i] = data_generate(distribution_type[i], 1)[0]
            if shift_flag and shift_lookup[i] and RL >= t_shift:
                data[i] += size_shift * shift_direction
        R = spatial_rank(data_ic, data)
        v = (1-lambda_) * v + lambda_ * R
        if shift_flag and RL < t_shift and np.sum(v**2)*10**4 > h:
            v = np.zeros(n_stream)
    if not shift_flag:
        return RL
    else:
        return RL - t_shift


def arl_SRAS_full(distribution_type, n_ic, lambda_, h, n_shift=None, size_shift=1, t_shift=30, n_thread=15, N=10000,
                  verbose=1):
    """
    Simulate ARL based of QH01.
    Input:
        n_thread: Number of thread.
        verbose: Control how much information are printed.
        See function 'rl_quants' for the rest of the arguments.
    Output:
        Simulated ARL
    """
    pool = multiprocessing.Pool(n_thread)
    if verbose > 1:
        print('Start Simulation!')
        print('Number of simulation runs: ' + str(N) + '. Number threads: ' + str(n_thread))
    tasks = [[distribution_type, n_ic, lambda_, h, n_shift, size_shift, t_shift] for _ in range(N)]
    RL_list = pool.starmap(rl_SRAS_full, tasks)
    pool.close()
    mu, sigma = np.mean(RL_list), np.std(RL_list)/np.sqrt(N)
    if verbose > 0:
        print('{:.2f}\t{:.2f}\t{:.2f}'.format(h, mu, sigma))
    return mu, sigma


def NW_estimation(bandwidth, data_ic, observation, observe_idx):
    """
    NW_estimation that estimate the unobservable data streams.
    Input:
        bandwidth: Bandwidth of the kernel function.
        data_ic: In-control data.
        observation: Observed data.
        observe_idx: Indices of the observed data streams.
    Output:
        Full vector of observation.
    """
    data_ic_observe = data_ic[observe_idx]
    weights = np.exp(-np.sum((data_ic_observe - observation[:, np.newaxis])**2, axis=0)/bandwidth**2/2)
    data_estimate = data_ic @ weights/np.sum(weights)
    data_estimate[observe_idx] = observation
    return data_estimate


def rl_SRAS(n_observe, distribution_type, n_ic, lambda_, h, bandwidth, delta, n_shift=None, size_shift=1, t_shift=30,
            data_ic_file=None, data_ol_file=None, bootstrap_flag=False):
    """
    Simulate RL of SRAS.
    Input:
        n_observe: Observable streams.
        distribution_type: List, representing the distribution of each data stream.
        n_ic: Number of in-control samples.
        lambda_: Parameter of the EWMA statistic.
        h: Threshold.
        bandwidth: ba
        delta: Compensation parameter.
        n_shift: Number of shifted data streams when the device is out of control.
        size_shift: Magnitude of the mean shift.
        t_shift: Shifted time.
        data_ic_file: In control data from file.

    Output:
        Simulated RL.
    """
    n_stream = len(distribution_type)
    shift_lookup = [False] * n_stream
    shift_direction = 1 if np.random.random(1) > 0.5 else -1
    shift_flag = n_shift is not None
    if shift_flag:
        shift_idx = np.random.choice(n_stream, n_shift, replace=False)
        for i in shift_idx:
            shift_lookup[i] = True
    if data_ic_file is None:
        data_ic = np.zeros((n_stream, n_ic))
        for i in range(n_stream):
            data_ic[i] = data_generate(distribution_type[i], shape=n_ic)
    else:
        data_ic = data_ic_file

    RL = 0
    v = np.zeros(n_stream)
    z = np.zeros(n_stream)
    L = np.zeros(n_stream)
    while np.sum(v**2)*10**4 <= h:
        z = (1-lambda_) * z + lambda_ * L
        observe_idx = np.sort(np.argsort(z)[-n_observe:])
        bootstrap_idx = np.random.choice(np.arange(data_ic.shape[1]), 1)

        observation = np.zeros(n_observe)
        for i, idx in enumerate(observe_idx):
            if bootstrap_flag:
                observation[i] = data_ic[i, bootstrap_idx]
            elif data_ol_file is not None:
                observation[i] = data_ol_file[i, RL]
            else:
                observation[i] = data_generate(distribution_type[idx], 1)[0]
                if shift_flag and shift_lookup[idx] and RL >= t_shift:
                    observation[i] += size_shift * shift_direction
        if n_observe < n_stream:
            data = NW_estimation(bandwidth=bandwidth, data_ic=data_ic, observation=observation, observe_idx=observe_idx)
        else:
            data = observation.copy()
        R = spatial_rank(data_ic, data)
        v = (1-lambda_) * v + lambda_ * R
        L = R**2 + data
        L[observe_idx] -= delta
        if shift_flag and RL < t_shift and np.sum(v**2)*10**4 > h:
            v, z, L = np.zeros(n_stream), np.zeros(n_stream), np.zeros(n_stream)

        RL += 1
        # print(RL, data_ol_file is not None, np.sum(v**2)*10**4 > h)
        if data_ol_file is not None and RL == len(data_ol_file[0]):
            # print(111)
            # print(len(data_ol_file[0]))
            break
        elif data_ol_file is not None and RL < 30 and np.sum(v**2)*10**4 > h:
            # print(222)
            v, z, L = np.zeros(n_stream), np.zeros(n_stream), np.zeros(n_stream)
    if not shift_flag:
        return RL
    else:
        return RL - t_shift


def arl_SRAS(n_observe, distribution_type, n_ic, lambda_, h, bandwidth, delta, n_shift=None, size_shift=1, t_shift=30,
             n_thread=15, N=10000, verbose=1, data_ic_file=None, data_ol_file=None, bootstrap_flag=False, debug=False):
    """
    Simulate ARL based of QH01.
    Input:
        n_thread: Number of thread.
        verbose: Control how much information are printed.
        See function 'rl_quants' for the rest of the arguments.
    Output:
        Simulated ARL
    """
    pool = multiprocessing.Pool(n_thread)
    if verbose > 1:
        print('Start Simulation!')
        print('Number of simulation runs: ' + str(N) + '. Number threads: ' + str(n_thread))
    if data_ol_file is None:
        tasks = [[n_observe, distribution_type, n_ic, lambda_, h, bandwidth, delta, n_shift, size_shift, t_shift,
                  data_ic_file, data_ol_file, bootstrap_flag] for _ in range(N)]
    else:
        tasks = [[n_observe, distribution_type, n_ic, lambda_, h, bandwidth, delta, n_shift, size_shift, t_shift,
                  data_ic_file, data_ol_file[i], bootstrap_flag] for i in range(N)]
    RL_list = pool.starmap(rl_SRAS, tasks)
    pool.close()
    if debug:
        return RL_list
    mu, sigma = np.mean(RL_list), np.std(RL_list)/np.sqrt(N)
    if verbose > 0:
        print('{:.2f}\t{:.2f}\t{:.2f}'.format(h, mu, sigma))
    return mu, sigma


if __name__ == '__main__':
    t = time.time()
    a = data_generate(5, (20, 1000))
    a = np.sort(a, axis=1)
    b = np.random.random(20)
    print(bayes_cdf(a, b))
    print(time.time()-t)
