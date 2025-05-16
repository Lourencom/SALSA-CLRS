# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sampling utilities adapted from https://github.com/deepmind/clrs/blob/master/clrs/_src/samplers.py"""

import abc
import collections
import inspect
import types

from typing import Any, Callable, List, Optional, Tuple, Dict
from loguru import logger


from clrs._src import probing
from clrs._src import specs

import numpy as np
import networkx as nx
from tqdm import trange
from scipy.spatial import Delaunay

from .specs import SPECS
from .algorithms import get_algorithm

_Array = np.ndarray
_DataPoint = probing.DataPoint
Trajectory = List[_DataPoint]
Trajectories = List[Trajectory]


Algorithm = Callable[..., Any]
Features = collections.namedtuple('Features', ['inputs', 'hints', 'lengths'])
FeaturesChunked = collections.namedtuple(
    'Features', ['inputs', 'hints', 'is_first', 'is_last'])
Feedback = collections.namedtuple('Feedback', ['features', 'outputs'])


class Sampler(abc.ABC):
  """Sampler abstract base class."""

  def __init__(
      self,
      algorithm: Algorithm,
      spec: specs.Spec,
      seed: Optional[int] = None,
      graph_generator: Optional[str] = None,
      graph_generator_kwargs: Optional[Dict[str, Any]] = None,
      **kwargs,
  ):
    """Initializes a `Sampler`.

    Args:
      algorithm: The algorithm to sample from
      spec: The algorithm spec.
      seed: RNG seed.
      **kwargs: Algorithm kwargs.
    """

    # Use `RandomState` to ensure deterministic sampling across Numpy versions.
    self._rng = np.random.RandomState(seed)
    self._graph_generator = graph_generator
    self._graph_generator_kwargs = graph_generator_kwargs
    self._spec = spec
    self._algorithm = algorithm
    self._kwargs = kwargs


  def _get_graph_generator_kwargs(self):
    return self._graph_generator_kwargs

  def next(self) -> Feedback:
    data = self._sample_data(**self._kwargs)
    _ , probes = self._algorithm(*data)
    inp, outp, hint = probing.split_stages(probes, self._spec)
    return inp, outp, hint


  def _create_graph(self, n, weighted, directed, low=0.0, high=1.0, **kwargs):
    """Create graph."""
    # Pass 'directed' and 'weighted' (and n, low, high) to the specific graph generators.
    # The **kwargs will contain other specific parameters like p_range for ER, k for WS.
    common_args = {'n': n, 'directed': directed, 'weighted': weighted, 'low': low, 'high': high}
    generator_specific_args = {**common_args, **kwargs}

    if self._graph_generator is None or self._graph_generator == 'er':
      mat =  self._random_er_graph(**generator_specific_args)
    elif self._graph_generator == 'ws':
      # Note: nx.watts_strogatz_graph is inherently undirected.
      # 'directed' and 'weighted' from common_args will be passed but might be ignored by _watt_strogatz_graph
      # or need specific handling within _watt_strogatz_graph if it were to support them.
      mat =  self._watt_strogatz_graph(**generator_specific_args)
    elif self._graph_generator == 'grid':
      mat =  self._grid_graph(**generator_specific_args)
    elif self._graph_generator == 'delaunay':
      mat =  self._random_delaunay_graph(**generator_specific_args)
    elif self._graph_generator == 'path':
      mat =  self._path_graph(**generator_specific_args)
    elif self._graph_generator == 'tree':
      mat =  self._tree_graph(**generator_specific_args)
    elif self._graph_generator == 'complete':
      # For a complete graph, n is the primary parameter.
      # 'directed' will determine if the resulting matrix (if weighted) has symmetric weights.
      _n = self._select_parameter(generator_specific_args['n']) # n might be a list/tuple
      mat = np.ones((_n,_n)) 
    else:
      raise ValueError(f'Unknown graph generator {self._graph_generator}.')
    n = mat.shape[0]
    if weighted:
      weights = self._rng.uniform(low=low, high=high, size=(n, n))
      if not directed:
        weights *= np.transpose(weights)
        weights = np.sqrt(weights + 1e-3)  # Add epsilon to protect underflow
      mat = mat.astype(float) * weights
    return mat
  
  @abc.abstractmethod
  def _sample_data(self, *args, **kwargs) -> List[_Array]:
    pass

  def _select_parameter(self, parameter, parameter_range=None):
    if parameter_range is not None:
      assert len(parameter_range) == 2
      return self._rng.uniform(*parameter_range)
    if isinstance(parameter, list) or isinstance(parameter, tuple):
      return self._rng.choice(parameter)
    else:
      return parameter

  def _random_er_graph(self, n, p=None, p_range=None, directed=False, acyclic=False, connected=True,
                       weighted=False, low=0.0, high=1.0, *args, **kwargs):
    """Random Erdos-Renyi graph."""
    n_actual = self._select_parameter(n)
    p_actual = self._select_parameter(p, p_range)

    while True:
      g = nx.erdos_renyi_graph(n_actual, p_actual, directed=directed)
      if connected:
        is_graph_connected = False
        if directed:
          is_graph_connected = nx.is_weakly_connected(g)
        else:
          is_graph_connected = nx.is_connected(g)
        
        if not is_graph_connected:
          continue
      
      return nx.to_numpy_array(g)

  def _watt_strogatz_graph(self, n, k, *args, p=None, p_range=None, directed=False, weighted=False, low=0.0, high=1.0, **kwargs):
    """Watts-Strogatz graph."""
    if directed:
        logger.warning("Watts-Strogatz graph generator currently produces undirected graphs. 'directed=True' will be ignored for structure.")

    n_actual = self._select_parameter(n)
    k_actual = self._select_parameter(k)
    p_actual = self._select_parameter(p, p_range)
    g = nx.connected_watts_strogatz_graph(n_actual, k_actual, p_actual)
    return nx.to_numpy_array(g)

  def _path_graph(self, n, *args, directed=False, weighted=False, low=0.0, high=1.0, **kwargs):
    """Path graph."""
    _n_actual = self._select_parameter(n)
    if directed:
        g = nx.path_graph(_n_actual, create_using=nx.DiGraph)
    else:
        g = nx.path_graph(_n_actual)
    return nx.to_numpy_array(g)

  def _grid_graph(self, dimensions, n, *args, directed=False, weighted=False, low=0.0, high=1.0, **kwargs):
    """2D grid graph."""
    if directed:
        logger.warning("Grid graph generator currently produces undirected graphs. 'directed=True' will be ignored for structure.")
    
    _dims_actual = dimensions
    if n != np.prod(_dims_actual) and n is not None:
        _n_from_kwargs = self._select_parameter(n)
        if _n_from_kwargs != np.prod(_dims_actual):
             logger.warning(f"Grid dimensions {dimensions} do not match n={_n_from_kwargs}. Using dimensions.")

    mat = nx.to_numpy_array(nx.grid_graph(_dims_actual))
    return mat
  
  def _random_delaunay_graph(self, n, *args, directed=False, weighted=False, low=0.0, high=1.0, **kwargs):
    """Random delaunay graph."""
    if directed:
        logger.warning("Delaunay graph generator currently produces undirected graphs. 'directed=True' will be ignored for structure.")

    n_actual = self._select_parameter(n)
    points = np.random.rand(n_actual, 2)
    tri = Delaunay(points)
    G = nx.Graph()
    G.add_nodes_from(range(n_actual))
    for edge_group in tri.simplices:
        nx.add_cycle(G, edge_group)
    return nx.to_numpy_array(G)

  def _tree_graph(self, n, r, *args, directed=False, weighted=False, low=0.0, high=1.0, **kwargs):
    """Tree."""
    n_actual = self._select_parameter(n)
    r_actual = self._select_parameter(r)
    if n_actual < 2:
      raise ValueError(f'Cannot generate tree of size {n_actual}.')
    
    if directed:
        mat = np.zeros((n_actual, n_actual))
        for i in range(1, n_actual):
            parent = (i - 1) // r_actual
            mat[parent, i] = 1
    else:
        mat = np.zeros((n_actual, n_actual))
        for i in range(1, n_actual):
            parent = (i - 1) // r_actual
            mat[i, parent] = 1
        mat = mat + mat.T
        mat = mat.astype(bool).astype(int)
    return mat



class DfsSampler(Sampler):
  """DFS sampler."""

  def _sample_data(self):
    params = self._get_graph_generator_kwargs().copy()
    # Handle 'directed': Use user's value, or default to False.
    if 'directed' not in params:
        params['directed'] = False
    # Handle 'weighted': DFS is typically unweighted.
    if 'weighted' not in params:
        params['weighted'] = False
    params.pop('acyclic', None)  # Remove 'acyclic' if present
    graph = self._create_graph(**params)
    return [graph]


class BfsSampler(Sampler):
  """BFS sampler."""

  def _sample_data(self):
    params = self._get_graph_generator_kwargs().copy()
    # Handle 'directed': Use user's value, or default to False.
    if 'directed' not in params:
        params['directed'] = False
    # Handle 'weighted': BFS is typically unweighted.
    if 'weighted' not in params:
        params['weighted'] = False
    params.pop('acyclic', None)  # Remove 'acyclic' if present
    graph = self._create_graph(**params)
    source_node = self._rng.choice(graph.shape[0])
    return [graph, source_node]

class ArticulationSampler(Sampler):
  """Articulation Point sampler."""

  def _sample_data(self):
    params = self._get_graph_generator_kwargs().copy()
    # Handle 'directed': Use user's value, or default to False.
    if 'directed' not in params:
        params['directed'] = False
    # Handle 'weighted': Articulation points typically on unweighted graphs.
    if 'weighted' not in params:
        params['weighted'] = False
    params.pop('acyclic', None)  # Remove 'acyclic' if present
    graph = self._create_graph(**params)
    return [graph]


class MSTSampler(Sampler):
  """MST sampler for Kruskal's algorithm."""

  def _sample_data(self, low=0.0, high=1.0):
    params = self._get_graph_generator_kwargs().copy()
    # Handle 'directed': Use user's value, or default to False.
    if 'directed' not in params:
        params['directed'] = False
    
    params['weighted'] = True  # MST requires weighted graphs
    # Add default low/high for weights if not provided by user in graph_generator_kwargs
    if 'low' not in params:
        params['low'] = low
    if 'high' not in params:
        params['high'] = high
            
    params.pop('acyclic', None)  # Remove 'acyclic' if present
    
    graph = self._create_graph(**params)
    return [graph]


class BellmanFordSampler(Sampler):
  """Bellman-Ford sampler."""

  def _sample_data(self, low=0.0, high=1.0):
    params = self._get_graph_generator_kwargs().copy()
    # Handle 'directed': Use user's value, or default to False.
    if 'directed' not in params:
        params['directed'] = False
    
    params['weighted'] = True  # Bellman-Ford and related (Prim, Dijkstra) require weighted graphs
    # Add default low/high for weights if not provided by user in graph_generator_kwargs
    if 'low' not in params:
        params['low'] = low
    if 'high' not in params:
        params['high'] = high

    params.pop('acyclic', None)  # Remove 'acyclic' if present
    
    graph = self._create_graph(**params)
    source_node = self._rng.choice(graph.shape[0])
    return [graph, source_node]

class MISSampler(Sampler):
  """MIS sampler for fast mis algos."""

  def _sample_data(self):
    params = self._get_graph_generator_kwargs().copy()
    # Handle 'directed': Use user's value, or default to False.
    if 'directed' not in params:
        params['directed'] = False
    # Handle 'weighted': MIS is typically on unweighted graphs.
    if 'weighted' not in params:
        params['weighted'] = False
    params.pop('acyclic', None)  # Remove 'acyclic' if present
    graph = self._create_graph(**params)
    return [graph]


def build_sampler(
    name: str,
    graph_generator: str = 'er',
    graph_generator_kwargs: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[Sampler, specs.Spec]:
  """Builds a sampler. See `Sampler` documentation."""
  if name not in SPECS or name not in SAMPLERS:
    raise NotImplementedError(f'No implementation of algorithm {name}.')
  spec = SPECS[name]
  algorithm = get_algorithm(name)
  sampler_class = SAMPLERS[name]
  # Ignore kwargs not accepted by the sampler.
  sampler_args = inspect.signature(sampler_class._sample_data).parameters  # pylint:disable=protected-access
  clean_kwargs = {k: kwargs[k] for k in kwargs if k in sampler_args}

  if set(clean_kwargs) != set(kwargs):
    logger.warning(f'Ignoring kwargs {set(kwargs).difference(clean_kwargs)} when building sampler class {sampler_class}')
  sampler = sampler_class(algorithm, spec, seed=seed, graph_generator=graph_generator, graph_generator_kwargs=graph_generator_kwargs,
                          **clean_kwargs)
  return sampler, spec



SAMPLERS = {
    'dfs': DfsSampler,
    # 'articulation_points': ArticulationSampler,
    # 'bridges': ArticulationSampler,
    'bfs': BfsSampler,
    # 'mst_kruskal': MSTSampler,
    'mst_prim': BellmanFordSampler,
    # 'bellman_ford': BellmanFordSampler,
    'dijkstra': BellmanFordSampler,
    'fast_mis': MISSampler,
    'eccentricity': BfsSampler,
    # 'eccentricity_path': BfsSampler,
}

