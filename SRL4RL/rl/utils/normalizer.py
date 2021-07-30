import threading
import numpy as np
from mpi4py import MPI
import torch
from SRL4RL.utils.nn_torch import numpy2pytorch, pytorch2numpy

variance = lambda x: ((x - x.mean()) ** 2).sum() / (len(x) - 1)


class Normalizer:
    def __init__(self, size, eps=1e-4, default_clip_range=np.inf, initNormalizer=None, name=''):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        self.reset(initNormalizer, name)

    def reset(self, initNormalizer=None, name=''):
        # some local information
        size = self.size
        if type(self.size) is list:
            size = self.size[0]
        if initNormalizer is not None:
            self.local_sum = initNormalizer[name + '_mean'] * 10000
            self.local_sumsq = np.square(initNormalizer[name + '_mean']) * 10000
            self.local_count = np.ones(1, np.float32) * 10000
            # get the total sum sumsq and sum count
            self.total_sum = initNormalizer[name + '_mean'] * 10000
            self.total_sumsq = np.square(initNormalizer[name + '_mean']) * 10000
            self.total_count = np.ones(1, np.float32) * 10000
            # get the mean and std
            self.mean = initNormalizer[name + '_mean']
            self.std = initNormalizer[name + '_std']
        else:
            self.local_sum = np.zeros(size, np.float32)
            self.local_sumsq = np.zeros(size, np.float32)
            self.local_count = np.zeros(1, np.float32)
            # get the total sum sumsq and sum count
            self.total_sum = np.zeros(size, np.float32)
            self.total_sumsq = np.zeros(size, np.float32)
            self.total_count = np.zeros(1, np.float32)
            # get the mean and std
            self.mean = np.zeros(self.size, np.float32)
            self.std = np.ones(self.size, np.float32)
        # thread locker
        self.lock = threading.Lock()

    # update the parameters of the normalizer
    def update(self, v):
        if type(self.size) is list:
            v = v.reshape([-1] + self.size)
            with self.lock:
                self.local_sum += v.sum(axis=0).sum(axis=-1).sum(axis=-1) / (self.size[-2] * self.size[-1])
                self.local_sumsq += (np.square(v)).sum(axis=0).sum(axis=-1).sum(axis=-1) / (
                            self.size[-2] * self.size[-1])
                self.local_count[0] += v.shape[0]
        else:
            v = v.reshape(-1, self.size)
            # do the computing
            with self.lock:
                self.local_sum += v.sum(axis=0)
                self.local_sumsq += (np.square(v)).sum(axis=0)
                self.local_count[0] += v.shape[0]

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
            # reset
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0
        # synrc the stats
        sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
        # update the total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(
            self.total_sum / self.total_count)))

    # average across the cpu's data
    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        if torch.is_tensor(v):
            v = torch.clamp(
                (v - numpy2pytorch(self.mean, differentiable=False, device=v.device)) / numpy2pytorch(self.std,
                                                                                                      differentiable=False,
                                                                                                      device=v.device),
                -clip_range, clip_range)
        else:
            v = np.clip((v - self.mean) / self.std, -clip_range, clip_range)
        return v

    def denormalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        if type(self.size) is list:
            raise NotImplementedError
            assert len(v.shape) == 4, 'adapt code'
            matMean = torch.zeros((v.shape))
            matStd = torch.zeros((v.shape))
            for i in range(len(self.size)):
                matMean[:, i] = torch.tensor(self.mean[i])
                matStd[:, i] = torch.tensor(self.std[i])
            if torch.is_tensor(v):
                matMean.to(device=v.device), matStd.to(device=v.device)
                v = torch.clamp(v * matStd + matMean, clip_range[0], clip_range[1])
            else:
                matMean, matStd = pytorch2numpy(matMean), pytorch2numpy(matStd)
                v = np.clip(v * matStd + matMean, clip_range[0], clip_range[1])
        else:
            if torch.is_tensor(v):
                v = v * numpy2pytorch(self.std, differentiable=False, device=v.device) + numpy2pytorch(self.mean,
                                                                                                       differentiable=False,
                                                                                                       device=v.device)
            else:
                v = v * self.std + self.mean
        return v

    def excludeUnpicklable(self):
        unpicklable = self.__dict__.copy()
        del unpicklable['lock']
        return unpicklable

    def update_normalizer(self, x):
        if torch.is_tensor(x):
            x = pytorch2numpy(x)
        self.update(x)
        self.recompute_stats()


class TimeNormalizer:
    def __init__(self, maxStepsReal):  # eps=1e-2
        len = maxStepsReal + 1
        time_steps = np.arange(0, maxStepsReal + 1)
        self.std = np.sqrt(np.sum(time_steps ** 2) / len - (np.sum(time_steps) / len) ** 2)
        self.mean = np.sum(time_steps) / len

    # update the parameters of the normalizer
    def update(self, v):
        pass

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        pass

    def recompute_stats(self):
        pass

    # average across the cpu's data
    def _mpi_average(self, x):
        pass

    # normalize the observation
    def normalize(self, v):
        if torch.is_tensor(v):
            v = (v - numpy2pytorch(self.mean, differentiable=False, device=v.device)) / numpy2pytorch(self.std,
                differentiable=False,device=v.device)
        else:
            v = (v - self.mean) / self.std
        return v

    def update_normalizer(self, x):
        pass