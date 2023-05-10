import random
import re
import time
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

def reparameterize(mu, logsigma):
    std = torch.exp(0.5*logsigma)
    eps = torch.randn_like(std)
    return mu + eps*std

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False
    
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
    
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


# utils latent model

def initialize_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def create_feature_actions(feature_, action_):
    N = feature_.size(0)
    # Flatten sequence of features.
    f = feature_[:, :-1].view(N, -1)
    n_f = feature_[:, 1:].view(N, -1)
    # Flatten sequence of actions.
    a = action_[:, :-1].view(N, -1)
    n_a = action_[:, 1:].view(N, -1)
    # Concatenate feature and action.
    fa = torch.cat([f, a], dim=-1)
    n_fa = torch.cat([n_f, n_a], dim=-1)
    return fa, n_fa

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)

def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False

def build_mlp(
    input_dim,
    output_dim,
    hidden_units=[64, 64],
    hidden_activation=nn.Tanh(),
    output_activation=None,
):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)

def calculate_gaussian_log_prob(log_std, noise):
    return (-0.5 * noise.pow(2) - log_std).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_std.size(-1)

def calculate_log_pi(log_std, noise, action):
    gaussian_log_prob = calculate_gaussian_log_prob(log_std, noise)
    return gaussian_log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

def reparameterize_actor(mean, log_std):
    noise = torch.randn_like(mean)
    action = torch.tanh(mean + noise * log_std.exp())
    return action, calculate_log_pi(log_std, noise, action)

def calculate_kl_divergence(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow_(2)
    t1 = ((p_mean - q_mean) / q_std).pow_(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

# DRIBO utils

from collections import deque, namedtuple
import sys

RESERVED_NAMES = ("get", "items")

def tuple_itemgetter(i):
    def _tuple_itemgetter(obj):
        return tuple.__getitem__(obj, i)
    return _tuple_itemgetter

def namedarraytuple(typename, field_names, return_namedtuple_cls=False,
        classname_suffix=False):
    """
    Returns a new subclass of a namedtuple which exposes indexing / slicing
    reads and writes applied to all contained objects, which must share
    indexing (__getitem__) behavior (e.g. numpy arrays or torch tensors).

    (Code follows pattern of collections.namedtuple.)

    >>> PointsCls = namedarraytuple('Points', ['x', 'y'])
    >>> p = PointsCls(np.array([0, 1]), y=np.array([10, 11]))
    >>> p
    Points(x=array([0, 1]), y=array([10, 11]))
    >>> p.x                         # fields accessible by name
    array([0, 1])
    >>> p[0]                        # get location across all fields
    Points(x=0, y=10)               # (location can be index or slice)
    >>> p.get(0)                    # regular tuple-indexing into field
    array([0, 1])
    >>> x, y = p                    # unpack like a regular tuple
    >>> x
    array([0, 1])
    >>> p[1] = 2                    # assign value to location of all fields
    >>> p
    Points(x=array([0, 2]), y=array([10, 2]))
    >>> p[1] = PointsCls(3, 30)     # assign to location field-by-field
    >>> p
    Points(x=array([0, 3]), y=array([10, 30]))
    >>> 'x' in p                    # check field name instead of object
    True
    """
    nt_typename = typename
    if classname_suffix:
        nt_typename += "_nt"  # Helpful to identify which style of tuple.
        typename += "_nat"

    try:  # For pickling, get location where this function was called.
        # NOTE: (pickling might not work for nested class definition.)
        module = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        module = None
    NtCls = namedtuple(nt_typename, field_names, module=module)

    def __getitem__(self, loc):
        try:
            return type(self)(*(None if s is None else s[loc] for s in self))
        except IndexError as e:
            for j, s in enumerate(self):
                if s is None:
                    continue
                try:
                    _ = s[loc]
                except IndexError:
                    raise Exception(f"Occured in {self.__class__} at field "
                        f"'{self._fields[j]}'.") from e

    __getitem__.__doc__ = (f"Return a new {typename} instance containing "
        "the selected index or slice from each field.")

    def __setitem__(self, loc, value):
        """
        If input value is the same named[array]tuple type, iterate through its
        fields and assign values into selected index or slice of corresponding
        field.  Else, assign whole of value to selected index or slice of
        all fields.  Ignore fields that are both None.
        """
        if not (isinstance(value, tuple) and  # Check for matching structure.
                getattr(value, "_fields", None) == self._fields):
            # Repeat value for each but respect any None.
            value = tuple(None if s is None else value for s in self)
        try:
            for j, (s, v) in enumerate(zip(self, value)):
                if s is not None or v is not None:
                    s[loc] = v
        except (ValueError, IndexError, TypeError) as e:
            raise Exception(f"Occured in {self.__class__} at field "
                f"'{self._fields[j]}'.") from e

    def __contains__(self, key):
        "Checks presence of field name (unlike tuple; like dict)."
        return key in self._fields

    def get(self, index):
        "Retrieve value as if indexing into regular tuple."
        return tuple.__getitem__(self, index)

    def items(self):
        "Iterate ordered (field_name, value) pairs (like OrderedDict)."
        for k, v in zip(self._fields, self):
            yield k, v

    for method in (__getitem__, __setitem__, get, items):
        method.__qualname__ = f'{typename}.{method.__name__}'

    arg_list = repr(NtCls._fields).replace("'", "")[1:-1]
    class_namespace = {
        '__doc__': f'{typename}({arg_list})',
        '__slots__': (),
        '__getitem__': __getitem__,
        '__setitem__': __setitem__,
        '__contains__': __contains__,
        'get': get,
        'items': items,
    }

    for index, name in enumerate(NtCls._fields):
        if name in RESERVED_NAMES:
            raise ValueError(f"Disallowed field name: {name}.")
        itemgetter_object = tuple_itemgetter(index)
        doc = f'Alias for field number {index}'
        class_namespace[name] = property(itemgetter_object, doc=doc)

    result = type(typename, (NtCls,), class_namespace)
    result.__module__ = NtCls.__module__

    if return_namedtuple_cls:
        return result, NtCls
    return result

class Scheduler(object):
    def __call__(self, **kwargs):
        raise NotImplementedError()


class LinearScheduler(Scheduler):
    def __init__(
        self, start_value, end_value, n_iterations, start_iteration=0
    ):
        self.start_value = start_value
        self.end_value = end_value
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.m = (end_value - start_value) / n_iterations

    def __call__(self, iteration):
        if iteration > self.start_iteration + self.n_iterations:
            return self.end_value
        elif iteration <= self.start_iteration:
            return self.start_value
        else:
            return (
                (iteration - self.start_iteration) * self.m + self.start_value
            )

class ExponentialScheduler(LinearScheduler):
    def __init__(
        self, start_value, end_value, n_iterations, start_iteration=0,
    ):
        super(ExponentialScheduler, self).__init__(
            start_value=np.log10(start_value),
            end_value=np.log10(end_value),
            n_iterations=n_iterations,
            start_iteration=start_iteration
        )

    def __call__(self, iteration):
        linear_value = super(ExponentialScheduler, self).__call__(iteration)
        return np.power(10., linear_value)

def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image