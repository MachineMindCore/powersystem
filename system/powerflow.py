import numpy as np

from typing import List
from system.helper import show_variable

def flat_profile(system) -> List[np.ndarray]:
    """
    Calcula el perfil plano como prerrequisito de cualquier método de flujo de potencia.


    Args:
        system (Dict[int, Dict]): Diccionario que describe el sistema de potencia.


    Returns:
        X0 (ndarray[float]): Vector de perfil plano de variables desconocidas.
        V0 (ndarray[complex]): Vector de perfil plano de voltaje complejo.
        IND (ndarry[str]): Vector de índice de variables desconocidas.
    """

    
    dim = len(system.keys())
    E = np.ones((dim, 1))
    d = np.zeros((dim, 1))
    
    miss_E = np.array([], dtype=str)
    miss_d = np.array([], dtype=str)
    for i, data in system.items():
        if data['E'] != None:
            E[i-1,0] = data['E']
        else:
            miss_E = np.append(miss_E, ['E'+str(i)], axis=0)
        if data['d'] != None:
            d[i-1,0] = data['d']
        else:
            miss_d = np.append(miss_d, ['d'+str(i)], axis=0)
    E_unknown = np.ones((len(miss_E), 1))
    d_unknown = np.zeros((len(miss_d), 1))

    V0 = E * np.exp(1j * d)
    X0 = np.concatenate(((d_unknown), (E_unknown)), dtype=float)
    IND = np.concatenate(((miss_d), (miss_E)), dtype=str)
    return X0, V0, IND

def compute_C(system, IND) -> np.ndarray:
    """
    Calcula los valores de potencia neta en los nodos con variables desconocidas.

    Args:
        system (Dict[int, Dict]): Diccionario que describe el sistema de potencia.
        IND (ndarry[str]): Vector de índice de variables desconocidas.

    Returns:
        C(ndarray[float]): Vector de potencias (P o Q) netas de los nodos de variables desconocidas.
    """
    C = np.zeros((len(IND), 1), dtype=float)
    for i, key in enumerate(IND):
        type = key[0]
        k = int(key[-1])
        if type == 'd':
            C[i] = system[k]['P_g'] - system[k]['P_d']
        if type == 'E':
            C[i] = system[k]['Q_g'] - system[k]['Q_d']
    return C

def b_prime(Y_mag, Y_ang, IND_d) -> np.ndarray:
    """
    Calcula la matriz B', aproximacion de J1.

    Args:
        Y_mag(ndarray[float]): Magnitud de la matriz de admitancia.
        Y_ang(ndarray[float]): Ángulo de la matriz de admitancia.
        IND_d(List[str]): Índices de angulos desconocidos.
    Returns: 
        Bp(ndarray[float]): Matriz B'
    """
    dim_d = len(IND_d)
    Bp = np.zeros((dim_d, dim_d), dtype=float)
    for row, key_row in enumerate(IND_d):
        for column, key_column in enumerate(IND_d):
            i_key = key_row
            j_key = key_column
            Bp[row,column] = Y_mag[i_key,j_key] * np.sin(Y_ang[i_key,j_key])
    return Bp

def b_prime_prime(Y_mag, Y_ang, IND_E) -> np.ndarray:
    """
    Calcula la matriz B'', aproximacion de J4.

    Args:
        Y_mag(ndarray[float]): Magnitud de la matriz de admitancia.
        Y_ang(ndarray[float]): Ángulo de la matriz de admitancia.
        IND_E(List[str]): Índices de tensiones desconocidas.
    Returns: 
        Bpp(ndarray[float]): Matriz B''
    """    
    dim_E = len(IND_E)
    Bpp = np.zeros((dim_E, dim_E), dtype=float)
    for row, key_row in enumerate(IND_E):
        for column, key_column in enumerate(IND_E):
            i_key = key_row
            j_key = key_column
            Bpp[row,column] = Y_mag[i_key,j_key] * np.sin(Y_ang[i_key,j_key])
    return Bpp

def newton_raphson_fast_decoupled(Y, X0, V0, C, IND, tol = 0.001, max_steps = 100) -> List[np.ndarray]:
    """
    Calcula las tensiones y ángulos nodales de un sistema basado en una suposición inicial.
    Esta función imprime un resumen no retornable de variables por iteración.
    Cualquiera de las condiciones de parada detiene las iteraciones.


    Args:
        Y(ndarray[complex]): Matriz de admitancia del sistema.
        X0 (ndarray[float]): Vector de perfil plano de variables desconocidas.
        V0 (ndarray[complex]): Vector de perfil plano de voltaje complejo.
        C(ndarray[float]): Vector de potencias (P o Q) netas de los nodos de variables desconocidas.
        IND (ndarry[str]): Vector de índice de variables desconocidas.
        tol(float): Error de tolerancia (condición de parada).
        max_steps(int): Máximo número de iteraciones (condición de parada).

    Returns:
        E(ndarray[float]): Vector de magnitud de voltajes correctos.
        d(ndarray[float]): Vector de ángulo de voltajes correctos.
    """

    # pre-iteracion
    ## Indices
    IND_d_str = list(filter(lambda ind: ind[0] == 'd', IND))
    IND_E_str = list(filter(lambda ind: ind[0] == 'E', IND))
    
    IND_d = [int(key[1:])-1 for key in IND_d_str]
    IND_E = [int(key[1:])-1 for key in IND_E_str]
    ## Dimensiones
    d_dim = len(IND_d)
    E_dim = len(IND_E)
    dim = len(Y)
    # Variables
    Y_mag = np.abs(Y)
    Y_ang = np.angle(Y)
    P = C[:d_dim]
    Q = C[d_dim:]

    P_k = np.zeros((d_dim, 1), dtype=float)
    Q_k = np.zeros((E_dim, 1), dtype=float)
    E_k = np.abs(V0)
    d_k = np.angle(V0)
    dd_k = np.zeros((d_dim, 1), dtype=float)
    dE_k = np.zeros((E_dim, 1), dtype=float)
    dP_k = np.zeros((d_dim, 1), dtype=float)
    dQ_k = np.zeros((E_dim, 1), dtype=float)
    k = 0

    # Separacion de X0 en X_Ek y X_dk
    X_dk = X0[:d_dim]
    X_Ek = X0[d_dim:]

    # Calculo de Bp y Bpp
    Bp = b_prime(Y_mag, Y_ang, IND_d)
    Bp_inv = np.linalg.inv(Bp)
    Bpp = b_prime_prime(Y_mag, Y_ang, IND_E)
    Bpp_inv = np.linalg.inv(Bpp)

    while True:
        k += 1
        # Proceso
        ## Actualizacion de P y Q
        for row, key_row in enumerate(IND_d):
            i = key_row
            P_k[row,0] = 0
            for j in range(dim):
                P_k[row,0] += E_k[i,0]*E_k[j,0]*Y_mag[i,j]*np.cos(Y_ang[i,j]-d_k[i,0]+d_k[j,0])
        
        for row, key_row in enumerate(IND_E):
            i = key_row
            Q_k[row,0] = 0
            for j in range(dim):
                Q_k[row,0] -= E_k[i,0]*E_k[j,0]*Y_mag[i,j]*np.sin(Y_ang[i,j]-d_k[i,0]+d_k[j,0])
        
        # Actualizacion de dP y dQ
        dP_k = P - P_k
        dQ_k = Q - Q_k

        # Actualizacion de dd y dE
        dd_k = np.matmul(-Bp_inv, np.divide(dP_k, E_k[IND_d]))
        dE_k = np.matmul(-Bpp_inv, np.divide(dQ_k, E_k[IND_E]))
        # Actulizacion de X_dk y X_Ek
        X_dk = X_dk + dd_k
        X_Ek = X_Ek + dE_k

        # Actualizacion d_k y E_k
        d_k[IND_d] = X_dk
        E_k[IND_E] = X_Ek 

        # Actualizacion del error
        error_P = max(np.abs(dP_k)).squeeze()
        error_Q = max(np.abs(dQ_k)).squeeze()
        error = max(error_P, error_Q)
        
        # Condicion
        if error < tol or k > max_steps:
            break
    return E_k, d_k