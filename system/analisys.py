import numpy as np

from typing import List, Dict


def admittance_matrix(system: Dict[int, Dict]) -> np.ndarray:
    num_buses = len(system)
    Y_system = np.zeros((num_buses, num_buses), dtype=complex)

    # Valores a nodos
    for from_id, bus_data in system.items():
        for to_id, y_ij in bus_data['conn']:
            Y_system[from_id-1][to_id-1] = -y_ij 
        Y_system[from_id-1][from_id-1] = sum([conn[1] for conn in bus_data['conn']]) + bus_data['conn_gnd']
    
    return Y_system

def impedance_matrix(Y_system: np.ndarray) -> np.ndarray:
    Z_system = sp.linalg.inv(Y_system)
    return Z_system

def injected_current(Y_system: np.array, E_system: np.array):
    return np.dot(Y_system, E_system)

def system_current(Y_system: np.array, E_system: np.array):
    I_system = Y_system * (E_system.T - E_system)
    return I_system

def injected_power(Y_system: np.array, E_system: np.array, ) -> List[float, float]:

    n = len(Y_system)
    P_injected = np.zeros((n, 1), dtype=float)
    Q_injected = np.zeros((n, 1), dtype=float)

    # Matriz de angulos de Y
    theta_system = np.angle(Y_system, deg=True)
    # Matriz de angulos de E
    delta_system = np.angle(E_system, deg=True)
    for i in range(n): 
        for k in range(n):
            angle = np.radians(theta_system[i,k] - delta_system[i,0] + delta_system[k,0])
            constant = np.abs(Y_system[i,k]*E_system[i,0]*E_system[k,0])
            # Calculo de potencia activa
            P_injected[i,0] += constant * np.cos(angle)
            # Calculo de potencia reactiva
            Q_injected[i,0] -= constant * np.sin(angle)
    return P_injected, Q_injected

def transfered_power(E_system: np.array, I_system: np.array):
    S_transfer = np.multiply(E_system, np.conjugate(I_system))
    return S_transfer

def power_loss_m1(E_system: np.array, I_injected: np.array) -> np.array:
    loss = np.matmul(E_system.T, np.conjugate(I_injected))
    loss = loss.squeeze()
    return loss

def power_loss_m2(S_injected: np.array) -> np.array:
    loss = 0
    n = len(S_injected)
    for i in range(n):
        loss += S_injected[i]
    loss = loss.squeeze()
    return loss

def power_loss_m3(S_transfer: np.array) -> np.array:
    loss = 0
    n = len(S_transfer)
    for i in range(n-1):
        for k in range(i+1,n):
            loss += S_transfer[i,k] + S_transfer[k,i]
    loss = loss.squeeze()
    return loss