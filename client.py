import socket
import sys
from collections import deque

import numpy as np
import threading
import RepeatedTimer


class Monitoring:
    def __init__(self, n_stream=4, n_aggregation=1) -> None:
        self.n_stream = n_stream
        self.data = np.zeros(n_stream)
        self.data_history = deque([self.data]*10, 10)
        self.local_statistics_pos = np.zeros(n_stream)
        self.local_statistics_neg = np.zeros(n_stream)
        self.local_statistics = np.zeros(n_stream)
        self.device_statistic = 0
        self.time_statistic = 0
        self.r = n_aggregation  # number of local statistics aggregated to get device-level statistic


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


# Initialize the variables
print('Args: ' + str(sys.argv))
print('Starting....')
if len(sys.argv) >= 3:
    HOST = sys.argv[1]
    PORT = int(sys.argv[2])
else:
    HOST = "10.141.20.236"
    PORT = 12345

n_stream, n_aggregation = 4, 1
m = Monitoring(n_stream=n_stream, n_aggregation=n_aggregation)
lock = threading.Lock()
ARL_list = np.load(f'data/A_{n_stream}_{n_aggregation}.npy')
k = 1.2  # Allowance parameter for updating the CUSUM statistics
time_interval = 2

# Simulate in control data
data_ic = np.zeros((n_stream, 1000))
distribution_type = [(i % 5) + 1 for i in range(n_stream)]
for i, type_ in enumerate(distribution_type):
    data_ic[i] = data_generate(type_, 1000)
data_ic = np.sort(data_ic, axis=1)


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


def bayes_cdf(x_list_sort, x):
    """
    Estimate the cdf of continuous and discrete distributions.
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


def collect_data():
    global m
    lock.acquire()
    for i in range(m.n_stream):
        m.data[i] = data_generate(distribution_type[i], 1)
        cdf = bayes_cdf(data_ic[i], m.data[i])
        m.local_statistics_pos[i] = max(m.local_statistics_pos[i] - np.log(cdf) - k, 0)
        m.local_statistics_neg[i] = max(m.local_statistics_neg[i] - np.log(1-cdf) - k, 0)
        m.local_statistics[i] = max(m.local_statistics_pos[i], m.local_statistics_neg[i])
    m.device_statistic = np.sum(sorted(m.local_statistics)[-m.r:])
    m.time_statistic = arl0_function(ARL_list, m.device_statistic)
    lock.release()


def connect_to_server(host, port):
    """ Create a socket connection to the server. """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    return s


def handle_connection(s):
    """ Handle the lifecycle of the server connection. """
    print('Successfully connected to server.')
    timer_collect_data = None

    try:
        while True:
            package = s.recv(1024).decode('utf-8')
            if not package:
                break  # Handle empty package in case of connection drop
            print(f'Receive message: {package}')
            msg = package[:5].strip().lower()
            if msg == 'start':
                timer_collect_data = RepeatedTimer.RepeatedTimer(time_interval, collect_data)
            elif msg[:4] == 'quit':
                print('Disconnect from server. Shutdown client.')
                break
            elif msg[:4] == 'send':
                msg_send = str(m.time_statistic).encode('utf-8')
                print(f'Send message {msg_send}')
                s.send(msg_send)

    except Exception as e:
        print("An error occurred: %s", e)
    finally:
        if timer_collect_data:
            timer_collect_data.stop()
        s.close()
        print('Connection closed.')


def communicate_with_server():
    """ Establish and manage server communication. """
    s = connect_to_server(HOST, PORT)
    handle_connection(s)


if __name__ == '__main__':
    communicate_with_server()

