import numpy as np
import math

def generate_weights(n_divisions=6, theta_index=0, phi_index=0):
    """
    Genera subdivisiones en coordenadas esféricas para un octante.
    
    Args:
        n_divisions (int): Número de divisiones en cada plano (XY, XZ, YZ).
    
    Returns:
        dict: Diccionario con subdivisiones en coordenadas esféricas.
    """
    # Crear ángulos según las divisiones
    angles = np.linspace(0, np.pi/2, n_divisions + 1)  # divisiones del plano
    subdivisions = {i: angles[i] for i in range(n_divisions+1)}
    
    # print(f"ESTO ES PARA THETA: {theta_index} Y PARA PHI: {phi_index}")
    #
    # print(f"w1: {math.sin(subdivisions[theta_index])*math.cos(subdivisions[phi_index])},   SEQUENCES")
    # print(f"w2: {math.sin(subdivisions[theta_index])*math.sin(subdivisions[phi_index])},     LOCdif")
    # print(f"w3: {math.cos(subdivisions[theta_index])},   CCdif")
    
    
    weights = {"w1": math.sin(subdivisions[theta_index])*math.cos(subdivisions[phi_index]),
               "w2": math.sin(subdivisions[theta_index])*math.sin(subdivisions[phi_index]),
               "w3": math.cos(subdivisions[theta_index])
               }
    
    return weights

# def generate_subdivisions(n_divisions=6):
#     """
#     Genera subdivisiones en coordenadas esféricas para un octante.
#
#     Args:
#         n_divisions (int): Número de divisiones en cada plano (XY, XZ, YZ).
#
#     Returns:
#         dict: Diccionario con subdivisiones en coordenadas esféricas.
#     """
#     # Crear ángulos según las divisiones
#     angles = np.linspace(0, np.pi/2, n_divisions + 1)  # divisiones del plano
#     subdivisions = {i: angles[i] for i in range(n_divisions+1)}
#     return subdivisions


# # Generar las subdivisiones
# n_divisions = 6
# subdivisions = generate_subdivisions(n_divisions)
#
# # Seleccionar una combinación específica
# theta_index = 2  # índice de 0 a n_divisions-1
# phi_index = 0  # índice de 0 a n_divisions-1
#
#
#
# phi = subdivisions[phi_index]
# theta = subdivisions[theta_index]
#
# w1 = math.sin(subdivisions[theta_index])*math.cos(subdivisions[phi_index])
# w2 = math.sin(subdivisions[theta_index])*math.sin(subdivisions[phi_index])
# w3 = math.cos(subdivisions[theta_index])
# print(f"W1: {w1}")
# print(f"W2: {w2}")
# print(f"W3: {w3}")

