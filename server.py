import pickle
import socket
import threading
import time
from time import strftime, localtime
import numpy as np
import RepeatedTimer


class ConnectionInfo:
    n_client_current = 0  # current number of clients
    connection_list = {}
    error_time_max = 2
    error_time_list = {}

    global_statistic = 0
    local_statistic = {}
    transmission_gap = {}
    client_importance = {}

    iteration = 0
    alarm_time = 0

    def __init__(self, n_client: int, n_observable: int, n_aggregation: int,
                 h: int=1000, delta: int=100) -> None:
        self.n_client = n_client
        self.n_observable = n_observable
        self.r = n_aggregation
        self.h = h  # Threshold
        self.delta = delta


HOST = "10.141.20.236"
PORT = 12345
time_interval = 2
lock = threading.Lock()

n_client = 1  # expected total number of clients
n_observable = 1
n_aggregation = 1
h = 1000
delta = 100
conn_info = ConnectionInfo(n_client, n_observable, n_aggregation, h, delta)
timer_communication = None


def connect():
    global timer_communication, conn_info
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("Server Started waiting for client to connect.")
        s.bind((HOST, PORT))
        while conn_info.n_client_current < conn_info.n_client:
            s.listen()
            conn, addr = s.accept()
            client_ip = addr[0]
            lock.acquire()
            if not client_ip in conn_info.connection_list:
                print("Connected by " + client_ip)
                conn_info.connection_list[client_ip] = conn
                conn_info.n_client_current += 1
                conn_info.error_time_list[client_ip] = 0
                conn_info.local_statistic[client_ip] = 0
                conn_info.transmission_gap[client_ip] = 0
                conn_info.client_importance[client_ip] = 0
            lock.release()
            time.sleep(0.1)
    msg = 'start'.encode('utf-8')
    for conn in conn_info.connection_list.values():
        conn.sendall(msg)
    timer_communication = RepeatedTimer.RepeatedTimer(time_interval, request_data)


def request_data():
    global conn_info
    print('\nTime: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()) + ' ')

    ip_sorted = sorted(conn_info.client_importance, key=conn_info.client_importance.get)
    for ip in ip_sorted[:-conn_info.n_observable]:
        lock.acquire()
        conn_info.transmission_gap[ip] += 1
        conn_info.client_importance[ip] += conn_info.delta
        lock.release()
    thread_list = []
    for ip in ip_sorted[-conn_info.n_observable:]:
        t = threading.Thread(target=get_data, args=(ip,))
        t.start()
        thread_list.append(t)
    for t in thread_list:
        t.join()

    lock.acquire()
    conn_info.iteration += 1
    local_stat_list = np.sort(list(conn_info.local_statistic.values()))
    conn_info.global_statistic = np.sum(local_stat_list[-conn_info.r:])
    print(f'Global statistic: {conn_info.global_statistic:.2f}')
    if conn_info.global_statistic > conn_info.h:
        print("ALARM! ALARM! ALARM! ALARM! ALARM! ALARM! ")
        conn_info.alarm_time += 1
    lock.release()


def get_data(ip):
    global conn_info
    lock.acquire()
    conn = conn_info.connection_list[ip]
    lock.release()
    try:
        msg = 'send'+str(conn_info.iteration)
        conn.sendall(msg.encode('utf-8'))
        data_recv = conn.recv(1024).decode('utf-8')
        data_recv = float(data_recv)
        lock.acquire()
        conn_info.local_statistic[ip] = data_recv
        conn_info.transmission_gap[ip] = 0
        conn_info.client_importance[ip] = data_recv
        print(f'\t\u2605\u2605\u2605 Client ip: {ip}')
        print(f'\t\t Received statistic: {data_recv:.2f}')
        lock.release()
    except BaseException as e:
        print(f'\t Client ip: {ip} {e}')
        if conn_info.error_time_list[ip] > conn_info.error_time_max:
            disconnect()
            return
        conn_info.error_time_list[ip] += 1


def disconnect():
    global timer_communication, conn_info
    if timer_communication:
        timer_communication.stop()
    lock.acquire()
    for conn in conn_info.connection_list.values():
        conn.close()
    lock.release()
    print(f"Shutdown server")


if __name__ == '__main__':
    connect()

