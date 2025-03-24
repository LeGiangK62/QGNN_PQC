import torch
import torch.nn as nn
from pennylane import numpy as np
import pennylane as qml

num_nodes_model = 5
num_edges_model = int(num_nodes_model * (num_nodes_model-1)/2)
n_qubits = num_nodes_model + num_edges_model

num_layers = 1
num_ent_lay = 3
mode = 'custom' #'strong
mode_num = 13

if mode == 'strong':
    weight_shapes = {"weights": (num_layers, num_ent_lay, 3, 3)} # Strong
elif mode == 'basic':
    weight_shapes = {"weights": (num_layers, num_ent_lay, 3)}
elif mode == 'custom':
    if mode_num == 1:
        weight_shapes = {"weights": (num_layers, 3, 2)} # custom circuit 1
    elif mode_num == 13:
        weight_shapes = {"weights": (num_layers, 3, 4)} # circuit 13
    elif mode_num == 14:
        weight_shapes = {"weights": (num_layers, 3, 4)} # circuit 13
    elif mode_num == 15:
        weight_shapes = {"weights": (num_layers, 3, 2)} # circuit 13
    elif mode_num == 139:
        weight_shapes = {"weights": (num_layers, 3, 3)} # circuit 13



dev = qml.device("default.qubit", wires=n_qubits)


def circuit_1(weights, wires):
    for idx, wire in enumerate(wires):
        qml.RX(weights[idx, 0], wires=wire)
        qml.RZ(weights[idx, 1], wires=wire)

def circuit_2(weights, wires):
    for idx, wire in enumerate(wires):
        qml.RX(weights[idx, 0], wires=wire)
        qml.RZ(weights[idx, 1], wires=wire)
    for idx, wire in enumerate(wires[:-1]):
        qml.CNOT(wires=[wire, wires[idx+1]])

def circuit_3(weights, wires):
    for idx, wire in enumerate(wires):
        qml.RX(weights[idx], wires=wire)
        qml.RZ(weights[idx], wires=wire)

def circuit_13(weights, wires):
    for idx, wire in enumerate(wires):
        qml.RY(weights[idx, 0], wires=wire)
    for idx, wire in enumerate(wires[::-1]):
        qml.CRZ(weights[idx, 1], wires=[wire, wires[-(idx)]])

    for idx, wire in enumerate(wires):
        qml.RY(weights[idx, 2], wires=wire)
    target =  wires[1:] + wires[:1]
    for idx, wire in enumerate(wires[-1:] + wires[:-1]):
        qml.CRZ(weights[idx, 3], wires=[wire, target[idx]])


def circuit_139(weights, wires):
    for idx, wire in enumerate(wires):
        qml.RY(weights[idx, 0], wires=wire)
    for idx, wire in enumerate(wires[::-1]):
        qml.CRZ(weights[idx, 1], wires=[wire, wires[-(idx)]])

    for idx, wire in enumerate(wires):
        qml.RY(weights[idx, 2], wires=wire)
    target =  wires[1:] + wires[:1]
    for idx, wire in enumerate(wires[-1:] + wires[:-1]):
        qml.CRZ(weights[idx, 1], wires=[wire, target[idx]])

def circuit_14(weights, wires):
    for idx, wire in enumerate(wires):
        qml.RY(weights[idx, 0], wires=wire)
    for idx, wire in enumerate(wires[::-1]):
        qml.CRX(weights[idx, 1], wires=[wire, wires[-(idx)]])

    for idx, wire in enumerate(wires):
        qml.RY(weights[idx, 2], wires=wire)
    target =  wires[1:] + wires[:1]
    for idx, wire in enumerate(wires[-1:] + wires[:-1]):
        qml.CRX(weights[idx, 3], wires=[wire, target[idx]])

def circuit_15(weights, wires):
    for idx, wire in enumerate(wires):
        qml.RY(weights[idx, 0], wires=wire)
    for idx, wire in enumerate(wires[::-1]):
        qml.CNOT(wires=[wire, wires[-(idx)]])

    for idx, wire in enumerate(wires):
        qml.RY(weights[idx, 1], wires=wire)
    target =  wires[1:] + wires[:1]
    for idx, wire in enumerate(wires[-1:] + wires[:-1]):
        qml.CNOT(wires=[wire, target[idx]])




def custom_pqc(weights, wires, mode=1):
    if mode == 1:
        circuit_1(weights, wires)
    elif mode == 2:
        circuit_2(weights, wires)
    elif mode == 3:
        circuit_3(weights, wires)
    elif mode == 13:
        circuit_13(weights, wires)
    elif mode == 14:
        circuit_14(weights, wires)
    elif mode == 15:
        circuit_15(weights, wires)


def qgnn_layer_4(theta, num_edges):
    qml.StronglyEntanglingLayers(weights=theta, wires=[0, num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    qml.StronglyEntanglingLayers(weights=theta, wires=[1, num_edges+1, num_edges+2])
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    qml.StronglyEntanglingLayers(weights=theta, wires=[2, num_edges+0, num_edges+1])
    qml.StronglyEntanglingLayers(weights=theta, wires=[3, num_edges+1, num_edges+2])
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    qml.StronglyEntanglingLayers(weights=theta, wires=[4, num_edges+2, num_edges+3])
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    qml.StronglyEntanglingLayers(weights=theta, wires=[5, num_edges+2, num_edges+3])

def custom_4(theta, num_edges, num_mode):
    custom_pqc(weights=theta, wires=[0, num_edges+0, num_edges+1],mode=num_mode)
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    custom_pqc(weights=theta, wires=[1, num_edges+1, num_edges+2],mode=num_mode)
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    custom_pqc(weights=theta, wires=[2, num_edges+0, num_edges+1],mode=num_mode)
    custom_pqc(weights=theta, wires=[3, num_edges+1, num_edges+2],mode=num_mode)
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    custom_pqc(weights=theta, wires=[4, num_edges+2, num_edges+3],mode=num_mode)
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    custom_pqc(weights=theta, wires=[5, num_edges+2, num_edges+3],mode=num_mode)


def qgnn_layer_4_Basic(theta, num_edges):
    qml.BasicEntanglerLayers(weights=theta, wires=[0, num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    qml.BasicEntanglerLayers(weights=theta, wires=[1, num_edges+1, num_edges+2])
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    qml.BasicEntanglerLayers(weights=theta, wires=[2, num_edges+0, num_edges+1])
    qml.BasicEntanglerLayers(weights=theta, wires=[3, num_edges+1, num_edges+2])
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    qml.BasicEntanglerLayers(weights=theta, wires=[4, num_edges+2, num_edges+3])
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    qml.BasicEntanglerLayers(weights=theta, wires=[5, num_edges+2, num_edges+3])

def unpack_inputs_single(inputs):
    adjacency_matrix = inputs[:num_edges_model].reshape(-1, num_edges_model)
    node_features = inputs[num_edges_model:].reshape(-1, num_nodes_model)
    return adjacency_matrix, node_features


def qgnn_layer_5_Basic(theta, num_edges):
    qml.BasicEntanglerLayers(weights=theta, wires=[0, num_edges+0, num_edges + 1])
    qml.SWAP(wires = [num_edges + 0, num_edges+1])
    qml.BasicEntanglerLayers(weights=theta, wires=[1, num_edges+1, num_edges + 2])
    qml.SWAP(wires = [num_edges + 1, num_edges+2])
    qml.BasicEntanglerLayers(weights=theta, wires=[2, num_edges+2, num_edges + 3])
    qml.SWAP(wires = [num_edges + 2, num_edges + 3])
    qml.BasicEntanglerLayers(weights=theta, wires=[3, num_edges+3, num_edges + 4])
    qml.BasicEntanglerLayers(weights=theta, wires=[4, num_edges+0, num_edges + 1])
    qml.SWAP(wires = [num_edges+0, num_edges+1])
    qml.BasicEntanglerLayers(weights=theta, wires=[5, num_edges+1, num_edges + 2])
    qml.SWAP(wires = [num_edges+1, num_edges+2])
    qml.SWAP(wires = [num_edges+2, num_edges+3])
    qml.BasicEntanglerLayers(weights=theta, wires=[6, num_edges+3, num_edges + 4])
    qml.BasicEntanglerLayers(weights=theta, wires=[7, num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    qml.SWAP(wires=[num_edges+2, num_edges+3])
    qml.BasicEntanglerLayers(weights=theta, wires=[8, num_edges+3, num_edges+4])
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    qml.SWAP(wires=[num_edges+2, num_edges+3])
    qml.BasicEntanglerLayers(weights=theta, wires=[9, num_edges+3, num_edges+4])

def qgnn_layer_5(theta, num_edges):
    qml.StronglyEntanglingLayers(weights=theta, wires=[0, num_edges+0, num_edges + 1])
    qml.SWAP(wires = [num_edges + 0, num_edges+1])
    qml.StronglyEntanglingLayers(weights=theta, wires=[1, num_edges+1, num_edges + 2])
    qml.SWAP(wires = [num_edges + 1, num_edges+2])
    qml.StronglyEntanglingLayers(weights=theta, wires=[2, num_edges+2, num_edges + 3])
    qml.SWAP(wires = [num_edges + 2, num_edges + 3])
    qml.StronglyEntanglingLayers(weights=theta, wires=[3, num_edges+3, num_edges + 4])
    qml.StronglyEntanglingLayers(weights=theta, wires=[4, num_edges+0, num_edges + 1])
    qml.SWAP(wires = [num_edges+0, num_edges+1])
    qml.StronglyEntanglingLayers(weights=theta, wires=[5, num_edges+1, num_edges + 2])
    qml.SWAP(wires = [num_edges+1, num_edges+2])
    qml.SWAP(wires = [num_edges+2, num_edges+3])
    qml.StronglyEntanglingLayers(weights=theta, wires=[6, num_edges+3, num_edges + 4])
    qml.StronglyEntanglingLayers(weights=theta, wires=[7, num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    qml.SWAP(wires=[num_edges+2, num_edges+3])
    qml.StronglyEntanglingLayers(weights=theta, wires=[8, num_edges+3, num_edges+4])
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    qml.SWAP(wires=[num_edges+2, num_edges+3])
    qml.StronglyEntanglingLayers(weights=theta, wires=[9, num_edges+3, num_edges+4])

def custom_5(theta, num_edges, mode=1):
    custom_pqc(weights=theta, wires=[0, num_edges+0, num_edges + 1], mode=mode)
    qml.SWAP(wires = [num_edges + 0, num_edges+1])
    custom_pqc(weights=theta, wires=[1, num_edges+1, num_edges + 2], mode=mode)
    qml.SWAP(wires = [num_edges + 1, num_edges+2])
    custom_pqc(weights=theta, wires=[2, num_edges+2, num_edges + 3], mode=mode)
    qml.SWAP(wires = [num_edges + 2, num_edges + 3])
    custom_pqc(weights=theta, wires=[3, num_edges+3, num_edges + 4], mode=mode)
    custom_pqc(weights=theta, wires=[4, num_edges+0, num_edges + 1], mode=mode)
    qml.SWAP(wires = [num_edges+0, num_edges+1])
    custom_pqc(weights=theta, wires=[5, num_edges+1, num_edges + 2])
    qml.SWAP(wires = [num_edges+1, num_edges+2])
    qml.SWAP(wires = [num_edges+2, num_edges+3])
    custom_pqc(weights=theta, wires=[6, num_edges+3, num_edges + 4], mode=mode)
    custom_pqc(weights=theta, wires=[7, num_edges+0, num_edges+1], mode=mode)
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    qml.SWAP(wires=[num_edges+2, num_edges+3])
    custom_pqc(weights=theta, wires=[8, num_edges+3, num_edges+4], mode=mode)
    qml.SWAP(wires=[num_edges+0, num_edges+1])
    qml.SWAP(wires=[num_edges+1, num_edges+2])
    qml.SWAP(wires=[num_edges+2, num_edges+3])
    custom_pqc(weights=theta, wires=[9, num_edges+3, num_edges+4], mode=mode)


@qml.qnode(dev, interface='torch', diff_method='backprop')
def qgnn_circuit(inputs, weights):
    adjacency_matrix, vertex_features = unpack_inputs_single(inputs)

    for i in range(num_edges_model):
        # qml.AmplitudeEmbedding(features=adjacency_matrix[:,i], wires=i, normalize=True, pad_with=0.)
        qml.RY(adjacency_matrix[:,i], wires=i)
    for i in range(num_nodes_model):
        # qml.AmplitudeEmbedding(features=vertex_features[:,i], wires=num_edges_model+i, normalize=True, pad_with=0.)
        qml.RY(vertex_features[:,i], wires=num_edges_model + i)
    for each_lay in range(num_layers):
        if num_nodes_model == 4:
            if mode == 'strong':
                qgnn_layer_4(weights[each_lay].to(torch.float32),num_edges_model)
            elif mode == 'basic':
                qgnn_layer_4_Basic(weights[each_lay].to(torch.float32),num_edges_model)
            elif mode == 'custom':
                custom_4(weights[each_lay].to(torch.float32),num_edges_model,mode_num)
        elif num_nodes_model == 5:
            if mode == 'strong':
                qgnn_layer_5(weights[each_lay].to(torch.float32),num_edges_model)
            elif mode == 'basic':
                qgnn_layer_5_Basic(weights[each_lay].to(torch.float32),num_edges_model)
            elif mode == 'custom':
                custom_5(weights[each_lay].to(torch.float32),num_edges_model,mode_num)
    return [qml.expval(qml.PauliZ(i)) for i in range(num_edges_model,n_qubits)]