"""Topology-aware GNN detector operating on the WDN junction-pipe graph.

Unlike ``GNNDetector`` in ``baselines.py`` (which constructs a k-NN graph in
sample space), this detector uses the **actual** water-network topology from
the EPANET ``.inp`` file: nodes = junctions/tanks/reservoirs, edges = pipes.

For each scenario (row), the tabular feature vector is broadcast onto every
junction node and concatenated with that junction's *static* topological
features (elevation, node type, centrality, degree).  A 2-layer GCN then
propagates information along pipe edges before a graph-level mean pool
produces a scenario-level representation that is fed to a binary classifier
head.

This lets the detector see how the scenario's aggregated sensor readings
interact with the network structure — a property the sample-space GNN cannot
express.
"""

from __future__ import annotations

from typing import Any, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    import torch as _torch
    import torch.nn as _nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _torch = None  # type: ignore[assignment]
    _nn = None  # type: ignore[assignment]
    _HAS_TORCH = False

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Batch, Data as _PyGData

    _HAS_PYG = True
except ImportError:  # pragma: no cover
    _HAS_PYG = False

from .baselines import (
    TabularDetectorBaseline,
    prepare_feature_matrix,
    select_numeric_feature_columns,
)


class TopologyAwareGNNDetector:
    """GCN detector that propagates scenario features over the WDN topology.

    A single reference graph is built once from the training network's
    EPANET file.  For every scenario (row of tabular features), the features
    are broadcast to every node, concatenated with the node's static
    topological features, and processed by a 2-layer GCN followed by mean
    pooling and a linear classifier head.

    Falls back to :class:`TabularDetectorBaseline` (random forest) when
    PyTorch / PyTorch Geometric are unavailable or when the reference
    topology cannot be loaded.

    Parameters
    ----------
    benchmark_path:
        Path to an EPANET ``.inp`` file that defines the reference topology.
    hidden_dim:
        GCN hidden dimension.
    epochs:
        Training epochs.
    lr:
        Adam learning rate.
    batch_size:
        Scenarios per mini-batch (each scenario becomes a graph in a batch).
    random_state:
        Seed for reproducibility.
    device:
        PyTorch device string.
    """

    def __init__(
        self,
        *,
        benchmark_path: str | None = None,
        hidden_dim: int = 64,
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 32,
        random_state: int = 42,
        device: str = "cpu",
    ) -> None:
        self.benchmark_path = benchmark_path
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

        self._model: Any = None
        self._scaler: Any = None
        self._edge_index: Any = None  # tensor (2, E)
        self._node_static: Any = None  # tensor (N, S)
        self._n_nodes: int = 0
        self._n_features: int = 0
        self.feature_columns_: list[str] = []
        self._fallback: TabularDetectorBaseline | None = None

        if not (_HAS_TORCH and _HAS_PYG):
            self._fallback = TabularDetectorBaseline(
                algorithm="random_forest", random_state=self.random_state,
            )

    # ------------------------------------------------------------------
    # Topology loading
    # ------------------------------------------------------------------

    def _load_reference_graph(self, path: str | None = None) -> bool:
        """Build/replace the reference WDN graph. Returns True on success."""
        target = path if path is not None else self.benchmark_path
        if target is None:
            return False
        try:
            from ..data.epanet import build_pyg_graph

            graph = build_pyg_graph(target)
        except Exception:
            return False

        if graph is None:
            return False

        if hasattr(graph, "edge_index"):
            edge_index = graph.edge_index
            x = graph.x
            n_nodes = int(graph.num_nodes)
        else:
            edge_index = _torch.from_numpy(graph["edge_index"]) if _HAS_TORCH else None
            x = _torch.from_numpy(graph["x"]) if _HAS_TORCH else None
            n_nodes = int(graph["num_nodes"])

        if edge_index is None or x is None or n_nodes == 0:
            return False

        self._edge_index = edge_index.to(self.device).long()
        static_idx = [2, 3, 5, 6]
        self._node_static = x[:, static_idx].float().to(self.device)
        self._n_nodes = n_nodes
        return True

    def set_topology(self, benchmark_path: str) -> bool:
        """Swap to a different topology (e.g. for cross-network inference).

        GCN weights are shared across nodes, so the model trained on one
        topology can be applied to another.  Returns True on success.
        """
        return self._load_reference_graph(benchmark_path)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _build_model(self, n_row_features: int) -> Any:
        hidden_dim = self.hidden_dim
        static_dim = int(self._node_static.shape[1])
        in_dim = n_row_features + static_dim

        class _TopologyGCN(_nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = GCNConv(in_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.classifier = _nn.Sequential(
                    _nn.Linear(hidden_dim, hidden_dim),
                    _nn.ReLU(),
                    _nn.Dropout(0.3),
                    _nn.Linear(hidden_dim, 2),
                )

            def forward(self, x: Any, edge_index: Any, batch: Any) -> Any:
                h = _torch.relu(self.conv1(x, edge_index))
                h = _torch.dropout(h, p=0.3, train=self.training)
                h = _torch.relu(self.conv2(h, edge_index))
                pooled = global_mean_pool(h, batch)  # (G, hidden)
                return self.classifier(pooled)

        return _TopologyGCN()

    # ------------------------------------------------------------------
    # Per-row graph construction
    # ------------------------------------------------------------------

    def _row_to_graph(self, row_features: Any) -> Any:
        """Build a single PyG Data object for one scenario row.

        The row's tabular features are broadcast to every node and
        concatenated with static topological features.
        """
        # row_features: 1-D tensor (n_row_features,)
        broadcast = row_features.unsqueeze(0).expand(self._n_nodes, -1)  # (N, F_row)
        x = _torch.cat([broadcast, self._node_static], dim=1)  # (N, F_row + F_static)
        return _PyGData(x=x, edge_index=self._edge_index)

    def _batched_graphs(self, x_scaled: Any, indices: Sequence[int]) -> Any:
        """Stack graphs for a batch of row indices into a PyG Batch."""
        graphs = [self._row_to_graph(x_scaled[i]) for i in indices]
        return Batch.from_data_list(graphs)

    # ------------------------------------------------------------------
    # Public fit / predict API
    # ------------------------------------------------------------------

    def fit(self, x: "np.ndarray", y: "np.ndarray") -> "TopologyAwareGNNDetector":
        if self._fallback is not None:
            self._fallback.fit(x, y)
            return self

        if not self._load_reference_graph():
            # Graceful degrade to tabular fallback when topology cannot be loaded
            self._fallback = TabularDetectorBaseline(
                algorithm="random_forest", random_state=self.random_state,
            )
            self._fallback.fit(x, y)
            return self

        _torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        x_scaled_np = self._scaler.fit_transform(x)
        x_scaled = _torch.tensor(x_scaled_np, dtype=_torch.float32, device=self.device)
        y_t = _torch.tensor(y, dtype=_torch.long, device=self.device)
        n = x_scaled.shape[0]
        self._n_features = x_scaled.shape[1]

        self._model = self._build_model(self._n_features).to(self.device)
        optimizer = _torch.optim.Adam(self._model.parameters(), lr=self.lr)

        # Class weights for imbalance
        n_pos = max(1, int((y == 1).sum()))
        n_neg = max(1, int((y == 0).sum()))
        weight = _torch.tensor([1.0 / n_neg, 1.0 / n_pos], device=self.device)
        weight = weight / weight.sum() * 2.0
        loss_fn = _nn.CrossEntropyLoss(weight=weight)

        self._model.train()
        for _ in range(self.epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                batch_idx = perm[start : start + self.batch_size].tolist()
                batch = self._batched_graphs(x_scaled, batch_idx)
                batch = batch.to(self.device)
                optimizer.zero_grad()
                logits = self._model(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(logits, y_t[_torch.tensor(batch_idx, device=self.device)])
                loss.backward()
                optimizer.step()

        return self

    def fit_from_frame(
        self,
        frame: "pd.DataFrame",
        *,
        target_column: str = "event_label",
        feature_columns: Sequence[str] | None = None,
    ) -> "TopologyAwareGNNDetector":
        x, y, used_columns = prepare_feature_matrix(
            frame, target_column=target_column, feature_columns=feature_columns,
        )
        self.feature_columns_ = list(used_columns)
        return self.fit(x, y)

    def predict_proba(self, x: "np.ndarray") -> "np.ndarray":
        if self._fallback is not None:
            return self._fallback.predict_proba(x)

        x_scaled_np = self._scaler.transform(x)
        x_scaled = _torch.tensor(x_scaled_np, dtype=_torch.float32, device=self.device)
        n = x_scaled.shape[0]

        self._model.eval()
        probs = np.zeros((n, 2), dtype=float)
        with _torch.no_grad():
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = list(range(start, end))
                batch = self._batched_graphs(x_scaled, batch_idx).to(self.device)
                logits = self._model(batch.x, batch.edge_index, batch.batch)
                probs[start:end] = _torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, x: "np.ndarray") -> "np.ndarray":
        proba = self.predict_proba(x)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_from_frame(
        self,
        frame: "pd.DataFrame",
        *,
        target_column: str = "event_label",
    ) -> "np.ndarray":
        if not self.feature_columns_:
            self.feature_columns_ = select_numeric_feature_columns(
                frame, target_column=target_column,
            )
        x, _, _ = prepare_feature_matrix(
            frame, target_column=target_column, feature_columns=self.feature_columns_,
        )
        return self.predict(x)
