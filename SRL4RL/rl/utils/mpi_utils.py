import numpy as np
import torch
from mpi4py import MPI


# sync_networks across the different cores
def sync_networks(network):
    """
    netowrk is the network you want to sync

    """
    comm = MPI.COMM_WORLD
    flat_params = _get_flat_params_or_grads(network, mode="params")
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params_or_grads(network, flat_params, mode="params")


def sync_params(parameters):
    """
    netowrk is the network you want to sync

    """
    comm = MPI.COMM_WORLD
    mode = "params"
    attr = "data" if mode == "params" else "grad"
    flat_params = np.array(getattr(parameters, attr).cpu().numpy().flatten())
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    pointer = 0
    getattr(parameters, attr).copy_(
        torch.tensor(flat_params[pointer : pointer + parameters.data.numel()]).view_as(
            parameters.data
        )
    )
    pointer += parameters.data.numel()


def sync_grads(network):
    flat_grads = _get_flat_params_or_grads(network, mode="grads")
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    _set_flat_params_or_grads(network, global_grads, mode="grads")


# get the flat grads or params
def _get_flat_params_or_grads(network, mode="params"):
    """
    include two kinds: grads and params

    """
    attr = "data" if mode == "params" else "grad"
    return np.concatenate(
        [getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()]
    )


def _set_flat_params_or_grads(network, flat_params, mode="params"):
    """
    include two kinds: grads and params

    """
    attr = "data" if mode == "params" else "grad"
    # the pointer
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(
            torch.tensor(flat_params[pointer : pointer + param.data.numel()]).view_as(
                param.data
            )
        )
        pointer += param.data.numel()


def sync_grads_single(parameters):
    ## _get_flat_params_or_grads
    attr = "grad"
    flat_grads = np.array(getattr(parameters, attr).cpu().numpy().flatten())

    ## sync_grads
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)

    ## _set_flat_params_or_grads
    # the pointer
    pointer = 0
    getattr(parameters, attr).copy_(
        torch.tensor(global_grads[pointer : pointer + parameters.data.numel()]).view_as(
            parameters.data
        )
    )
    pointer += parameters.data.numel()
