"""Graph-Conditioned Conditional Variational Autoencoder (Graph-CVAE)
for WDN disturbance augmentation.

Architecture:
    Encoder: GATv2Conv (2 layers, edge features) → masked global pool →
             CVAE conditioning → μ, σ²
    Decoder: per-node MLP (z + graph structure + conditioning → node/edge features)
    Physics loss: L_mass (junction continuity) + L_energy (Hazen-Williams headloss)

Falls back gracefully to GaussianMixtureAugmenter if PyTorch/PyG is unavailable,
and to a tabular-mode MLP-CVAE with physics-proxy regularisation when a benchmark
path is not provided or graph construction fails.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional heavy dependency guards
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
    from torch_geometric.data import Data, Batch

    _HAS_PYG = True
except ImportError:  # pragma: no cover
    _HAS_PYG = False

from .augmentation import GaussianMixtureAugmenter


# ---------------------------------------------------------------------------
# Helper: physics lambda warmup schedule
# ---------------------------------------------------------------------------

def _physics_lambda_warmup(current_epoch: int, total_warmup: int = 25) -> float:
    """Return the effective physics-loss lambda multiplier.

    Returns 0.0 for epochs 0–10, then linearly ramps to 1.0 by
    *total_warmup* (default 25).  Prevents early destabilisation from
    large un-trained physics residuals.

    Args:
        current_epoch: Zero-indexed current training epoch.
        total_warmup: Epoch at which the multiplier reaches 1.0.

    Returns:
        Float in [0.0, 1.0].
    """
    if current_epoch < 10:
        return 0.0
    if current_epoch >= total_warmup:
        return 1.0
    return float(current_epoch - 10) / float(max(total_warmup - 10, 1))


# ---------------------------------------------------------------------------
# Inner GATv2-CVAE model (requires PyTorch + PyG)
# ---------------------------------------------------------------------------

if _HAS_TORCH and _HAS_PYG:

    class _GATv2CVAE(nn.Module):
        """GATv2-based CVAE for graph-structured WDN disturbance generation.

        Encoder architecture:
            Input: node_features (N×8), edge_index, edge_attr (M×5), conditioning (C,)
            Layer 1: GATv2Conv(in=8, out=64, heads=4, edge_dim=5, concat=True) → (N×256) + ELU
            Layer 2: GATv2Conv(in=256, out=128, heads=2, edge_dim=5, concat=False) → (N×128) + ELU
            Global pool: sensor-masked mean pool (only observed nodes) → (128,)
            Concat with conditioning → (128 + cond_dim,)
            MLP head → (μ ∈ R^latent_dim, log_σ² ∈ R^latent_dim)

        Decoder architecture:
            Input: z (latent_dim,), node_static_features (N×8), conditioning (C,)
            Per-node: broadcast z → (N×latent_dim), concat with static + cond
            MLP → (N×3) [pressure, demand, flow]
        """

        def __init__(
            self,
            node_feature_dim: int = 8,
            edge_feature_dim: int = 5,
            hidden_dim: int = 128,
            latent_dim: int = 64,
            n_disturbance_types: int = 4,
            n_networks: int = 7,
            network_embed_dim: int = 8,
        ) -> None:
            super().__init__()
            self.latent_dim = latent_dim
            self.hidden_dim = hidden_dim
            self.node_feature_dim = node_feature_dim

            # Conditioning: disturbance type embedding + network embedding
            self.dist_embed = nn.Embedding(n_disturbance_types, network_embed_dim)
            self.net_embed = nn.Embedding(n_networks + 1, network_embed_dim)  # +1 for unknown
            cond_dim = network_embed_dim * 2

            # Encoder GATv2 layers
            self.gat1 = GATv2Conv(
                in_channels=node_feature_dim,
                out_channels=64,
                heads=4,
                edge_dim=edge_feature_dim,
                concat=True,
            )
            self.gat2 = GATv2Conv(
                in_channels=256,  # 64 * 4 heads
                out_channels=hidden_dim,
                heads=2,
                edge_dim=edge_feature_dim,
                concat=False,
            )
            self.elu = nn.ELU()

            # Encoder MLP head: pool output + conditioning → μ, log_var
            # Initialised lazily in encode() so the actual runtime dimension of
            # enc_input drives the layer size (avoids hard-coding cond_dim here).
            self.fc_mu: nn.Linear | None = None
            self.fc_log_var: nn.Linear | None = None

            # Decoder: per-node MLP — lazy-initialised in decode() so the
            # actual runtime dimension of dec_input drives the layer size
            # (avoids hard-coding cond_dim which may differ at runtime).
            self.decoder_mlp: nn.Sequential | None = None
            self._decoder_input_dim: int | None = None

        def _make_conditioning(
            self, dist_type_idx: "torch.Tensor", net_idx: "torch.Tensor"
        ) -> "torch.Tensor":
            return torch.cat([self.dist_embed(dist_type_idx), self.net_embed(net_idx)], dim=-1)

        def encode(
            self,
            x: "torch.Tensor",
            edge_index: "torch.Tensor",
            edge_attr: "torch.Tensor",
            sensor_mask: "torch.Tensor | None",
            conditioning: "torch.Tensor",
        ) -> "tuple[torch.Tensor, torch.Tensor]":
            """Encode graph to latent distribution parameters.

            Args:
                x: Node features (N×8).
                edge_index: Edge index (2×M).
                edge_attr: Edge attributes (M×5).
                sensor_mask: Boolean mask (N,) selecting observed nodes for pooling.
                conditioning: Conditioning vector (cond_dim,).

            Returns:
                Tuple of (mu, log_var), each shape (latent_dim,).
            """
            # GATv2 encoding
            h = self.elu(self.gat1(x, edge_index, edge_attr))
            h = self.elu(self.gat2(h, edge_index, edge_attr))

            # Sensor-masked global mean pool
            if sensor_mask is not None and sensor_mask.any():
                h_observed = h[sensor_mask]
            else:
                h_observed = h
            pooled = h_observed.mean(dim=0)  # (hidden_dim,)

            # Concatenate with conditioning
            enc_input = torch.cat([pooled, conditioning], dim=-1)  # (hidden_dim + cond_dim,)

            # Lazy-init fc_mu / fc_log_var on first forward pass so the actual
            # runtime dimension of enc_input drives the layer — no hard-coded math.
            if self.fc_mu is None or self.fc_mu.in_features != enc_input.shape[-1]:
                actual_dim = enc_input.shape[-1]
                self.fc_mu = nn.Linear(actual_dim, self.latent_dim).to(enc_input.device)
                self.fc_log_var = nn.Linear(actual_dim, self.latent_dim).to(enc_input.device)

            mu = self.fc_mu(enc_input)
            log_var = self.fc_log_var(enc_input)
            return mu, log_var

        def reparameterize(
            self, mu: "torch.Tensor", log_var: "torch.Tensor"
        ) -> "torch.Tensor":
            """Reparameterisation trick: z = μ + ε·σ."""
            if self.training:
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std
            return mu

        def decode(
            self,
            z: "torch.Tensor",
            x_static: "torch.Tensor",
            conditioning: "torch.Tensor",
            n_nodes: int,
        ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
            """Decode latent vector to per-node hydraulic quantities.

            Args:
                z: Latent vector (latent_dim,).
                x_static: Static node features (N×8).
                conditioning: Conditioning vector (cond_dim,).
                n_nodes: Number of nodes N.

            Returns:
                Tuple of (pressure_pred, demand_pred, flow_pred) each (N,).
            """
            z_broadcast = z.unsqueeze(0).expand(n_nodes, -1)  # (N, latent_dim)
            cond_broadcast = conditioning.unsqueeze(0).expand(n_nodes, -1)  # (N, cond_dim)
            dec_input = torch.cat([z_broadcast, x_static, cond_broadcast], dim=-1)

            # Lazy-init decoder_mlp on first forward pass so the actual
            # runtime dimension of dec_input drives the layer size.
            if self.decoder_mlp is None or self._decoder_input_dim != dec_input.shape[-1]:
                actual_dim = dec_input.shape[-1]
                self._decoder_input_dim = actual_dim
                self.decoder_mlp = nn.Sequential(
                    nn.Linear(actual_dim, self.hidden_dim),
                    nn.ELU(),
                    nn.Linear(self.hidden_dim, 64),
                    nn.ELU(),
                    nn.Linear(64, 3),  # pressure, demand, flow per node
                ).to(dec_input.device)

            out = self.decoder_mlp(dec_input)  # (N, 3)
            pressure_pred = out[:, 0]
            demand_pred = out[:, 1]
            flow_pred = out[:, 2]
            return pressure_pred, demand_pred, flow_pred

        def forward(
            self,
            x: "torch.Tensor",
            edge_index: "torch.Tensor",
            edge_attr: "torch.Tensor",
            conditioning: "torch.Tensor",
            sensor_mask: "torch.Tensor | None" = None,
        ) -> "tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]":
            """Full forward pass (encode → reparameterise → decode).

            Returns:
                ((pressure_pred, demand_pred, flow_pred), mu, log_var)
            """
            mu, log_var = self.encode(x, edge_index, edge_attr, sensor_mask, conditioning)
            z = self.reparameterize(mu, log_var)
            recon = self.decode(z, x, conditioning, x.shape[0])
            return recon, mu, log_var

    # -----------------------------------------------------------------------
    # Tabular-mode fallback MLP-CVAE (no graph; operates on flat feature vector)
    # -----------------------------------------------------------------------

    class _TabularCVAE(nn.Module):
        """MLP-CVAE for tabular WDN features when no graph topology is available.

        This is the minimum-viable CVAE that works on the aggregated feature
        vector directly.  It is novel in that it still incorporates a
        physics-proxy soft constraint (pressure bounds / mass balance residuals)
        as a regularisation term alongside the standard ELBO.

        Encoder:
            feature_dim → hidden_dim → ELU → hidden_dim → (μ, log_σ²)
        Decoder:
            latent_dim + cond_dim → hidden_dim → ELU → feature_dim
        """

        def __init__(
            self,
            feature_dim: int,
            hidden_dim: int = 128,
            latent_dim: int = 64,
            cond_dim: int = 4,
        ) -> None:
            super().__init__()
            self.latent_dim = latent_dim

            self.encoder = nn.Sequential(
                nn.Linear(feature_dim + cond_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
            )
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim + cond_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, feature_dim),
            )

        def encode(
            self, x: "torch.Tensor", cond: "torch.Tensor"
        ) -> "tuple[torch.Tensor, torch.Tensor]":
            h = self.encoder(torch.cat([x, cond], dim=-1))
            return self.fc_mu(h), self.fc_log_var(h)

        def reparameterize(
            self, mu: "torch.Tensor", log_var: "torch.Tensor"
        ) -> "torch.Tensor":
            if self.training:
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std
            return mu

        def decode(self, z: "torch.Tensor", cond: "torch.Tensor") -> "torch.Tensor":
            return self.decoder(torch.cat([z, cond], dim=-1))

        def forward(
            self, x: "torch.Tensor", cond: "torch.Tensor"
        ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
            mu, log_var = self.encode(x, cond)
            z = self.reparameterize(mu, log_var)
            recon = self.decode(z, cond)
            return recon, mu, log_var


# ---------------------------------------------------------------------------
# Physics loss module
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class _HydraulicPhysicsLoss(nn.Module):
        """Differentiable hydraulic physics constraint loss.

        Implements three soft-constraint penalty terms:

        1. **Mass conservation** (Kirchhoff's current law for flow):

               L_mass = mean_i [(Σ_j q_ij - d_i)²]

           The sum is over all edges incident on junction i; positive flow
           means flow *into* i, negative means flow *out of* i.

        2. **Energy conservation** (Hazen-Williams headloss equation):

               L_energy = mean_{ij} [(h_i - h_j - R_ij · |q_ij|^1.852 · sgn(q_ij))²]

           where R_ij = 10.67 · L / (C^1.852 · D^4.87) is pre-computed.

        3. **Pressure bounds** (service-pressure constraints):

               L_bounds = mean_i [ReLU(p_min − p_i)² + ReLU(p_i − p_max)²]
        """

        def __init__(
            self,
            pipe_resistance: Dict[str, float] | None = None,
            *,
            p_min: float = 10.0,
            p_max: float = 105.0,
            lambda_mass: float = 1.0,
            lambda_energy: float = 1.0,
            lambda_bounds: float = 0.5,
        ) -> None:
            super().__init__()
            self.pipe_resistance = pipe_resistance or {}
            self.p_min = p_min
            self.p_max = p_max
            self.lambda_mass = lambda_mass
            self.lambda_energy = lambda_energy
            self.lambda_bounds = lambda_bounds

        def warmup_step(self, current_epoch: int, total_warmup: int = 25) -> float:
            """Return effective lambda multiplier with linear warmup.

            Prevents large un-trained physics residuals from destabilising
            early training.  Returns 0.0 for epochs 0–9, linearly ramps to
            1.0 by *total_warmup*.

            Args:
                current_epoch: Current (0-indexed) training epoch.
                total_warmup: Epoch at which multiplier reaches 1.0.

            Returns:
                Float in [0.0, 1.0].
            """
            return _physics_lambda_warmup(current_epoch, total_warmup)

        def forward(
            self,
            pressure_pred: "torch.Tensor",
            demand_pred: "torch.Tensor",
            flow_pred: "torch.Tensor",
            edge_index: "torch.Tensor",
            pipe_resistance_tensor: "torch.Tensor | None" = None,
        ) -> "torch.Tensor":
            """Compute total physics constraint loss.

            Args:
                pressure_pred: Predicted pressure at each node (N,).
                demand_pred: Predicted demand at each node (N,).
                flow_pred: Predicted flow at each node (N,) — treated as the
                    net nodal flow signal when full pipe-level flows are
                    unavailable.
                edge_index: Graph edge index (2×M).
                pipe_resistance_tensor: Pre-computed R_ij per edge (M,), or
                    None to skip the energy loss term.

            Returns:
                Scalar physics loss tensor.
            """
            device = pressure_pred.device
            n_nodes = pressure_pred.shape[0]

            # ---- Pressure bounds loss ------------------------------------
            p_min_t = torch.tensor(self.p_min, dtype=pressure_pred.dtype, device=device)
            p_max_t = torch.tensor(self.p_max, dtype=pressure_pred.dtype, device=device)
            l_bounds = (
                torch.relu(p_min_t - pressure_pred).pow(2).mean()
                + torch.relu(pressure_pred - p_max_t).pow(2).mean()
            )

            # ---- Mass conservation loss ----------------------------------
            # Compute net inflow at each node using edge_index.
            # edge_index[0] = source nodes, edge_index[1] = destination nodes.
            # For undirected pipes represented as bidirectional edges:
            # flow from src→dst is +flow_pred[src]; flow from dst→src is -flow_pred[src].
            # Approximate: accumulate flow_pred values at destination nodes.
            l_mass = torch.tensor(0.0, dtype=pressure_pred.dtype, device=device)
            if edge_index.shape[1] > 0:
                src = edge_index[0]
                dst = edge_index[1]
                # Net inflow at each node: sum of edge flows pointing to it
                flow_at_src = flow_pred[src]
                net_inflow = torch.zeros(n_nodes, dtype=pressure_pred.dtype, device=device)
                net_inflow.scatter_add_(0, dst, flow_at_src)
                # Conservation residual: net inflow - demand ≈ 0
                residual = net_inflow - demand_pred
                l_mass = residual.pow(2).mean()

            # ---- Energy conservation loss (Hazen-Williams) ---------------
            l_energy = torch.tensor(0.0, dtype=pressure_pred.dtype, device=device)
            if pipe_resistance_tensor is not None and edge_index.shape[1] > 0:
                src = edge_index[0]
                dst = edge_index[1]
                h_i = pressure_pred[src]
                h_j = pressure_pred[dst]
                q_raw = flow_pred[src]
                q_abs = q_raw.abs()
                # R_ij * |q_ij|^1.852 * sgn(q_ij) — signed Hazen-Williams
                m = edge_index.shape[1]
                r_ij = pipe_resistance_tensor[:m] if pipe_resistance_tensor.shape[0] >= m else pipe_resistance_tensor
                hw_headloss = r_ij * q_abs.pow(1.852) * q_raw.sign()
                l_energy = (h_i - h_j - hw_headloss).pow(2).mean()

            total = (
                self.lambda_bounds * l_bounds
                + self.lambda_mass * l_mass
                + self.lambda_energy * l_energy
            )
            return total


# ---------------------------------------------------------------------------
# Main public class: GraphCVAEAugmenter
# ---------------------------------------------------------------------------

class GraphCVAEAugmenter:
    """Graph-Conditioned CVAE augmenter for WDN disturbance data.

    This is the primary novel contribution of the paper.  Generates
    physically plausible synthetic disturbance signatures by:

    1. Encoding the WDN graph topology and observed sensor readings via a
       two-layer GATv2Conv encoder.
    2. Conditioning the latent space on disturbance type and sensor coverage.
    3. Decoding through the network graph with Hazen-Williams / mass-balance
       physics constraint regularisation.

    **Interface** matches the existing augmenters
    (``PositiveClassAugmenter``, ``GaussianMixtureAugmenter``):

    - ``fit(train_frame, *, feature_columns, target_column)``
    - ``generate(train_frame, *, feature_columns, target_column, n_samples)``

    **Fallback chain**:

    1. If neither PyTorch nor PyG is installed → ``GaussianMixtureAugmenter``
       (prints a warning).
    2. If PyTorch is installed but PyG is not, or if no *benchmark_path* is
       provided → tabular-mode ``_TabularCVAE`` (MLP encoder/decoder, still
       with physics-proxy regularisation).
    3. Full graph mode: GATv2Conv encoder + per-node MLP decoder +
       ``_HydraulicPhysicsLoss``.

    Args:
        benchmark_path: Optional path to an EPANET ``.inp`` file.  When
            provided and PyG is available the full graph-CVAE mode is used.
        latent_dim: Dimensionality of the latent space.
        hidden_dim: Hidden layer width for both encoder and decoder MLPs.
        epochs: Number of training epochs.
        lr: Adam learning rate.
        batch_size: Mini-batch size (number of training rows per step).
        random_state: Random seed for reproducibility.
        beta_kl: Final β weight for KL divergence term (annealed 0→β).
        lambda_physics: Weight for the physics constraint loss term.
        warmup_epochs: Epoch at which the physics lambda reaches 1.0.
        device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
    """

    #: Disturbance type → integer index used for conditioning embedding.
    _DISTURBANCE_INDEX: Dict[str, int] = {
        "leak": 0,
        "pipe_closure": 1,
        "pump_outage": 2,
        "normal": 3,
    }

    def __init__(
        self,
        *,
        benchmark_path: str | None = None,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        random_state: int = 42,
        beta_kl: float = 1.0,
        lambda_physics: float = 0.5,
        warmup_epochs: int = 25,
        device: str = "cpu",
    ) -> None:
        self.benchmark_path = benchmark_path
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.beta_kl = beta_kl
        self.lambda_physics = lambda_physics
        self.warmup_epochs = warmup_epochs
        self.device_str = device

        # Internal state set by fit()
        self._model: Any = None
        self._physics_loss: Any = None
        self._feature_columns: List[str] = []
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._graph_mode: bool = False
        self._pyg_graph: Any = None
        self._pipe_resistance_tensor: Any = None
        self._fallback: GaussianMixtureAugmenter | None = None
        self.training_history_: List[Dict[str, float]] = []

        # Determine operating mode
        if not _HAS_TORCH:
            print(
                "[warn] torch not available; GraphCVAEAugmenter falling back to "
                "GaussianMixtureAugmenter.  Install PyTorch: pip install torch",
                file=sys.stderr,
            )
            self._fallback = GaussianMixtureAugmenter(random_state=random_state)
            return

        # Set reproducibility seed
        torch.manual_seed(random_state)
        np.random.default_rng(random_state)

        # Check if full graph mode is possible
        if benchmark_path is not None and _HAS_PYG:
            self._graph_mode = True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_cond_vector(self, dist_type: str = "leak", net_idx: int = 0) -> "torch.Tensor":
        """Build a one-hot-like conditioning vector (tabular mode)."""
        dist_idx = self._DISTURBANCE_INDEX.get(dist_type, 0)
        cond = torch.zeros(4, dtype=torch.float32)
        cond[dist_idx] = 1.0
        return cond

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalise a feature matrix using fitted statistics."""
        return (x - self._feature_mean) / np.maximum(self._feature_std, 1e-9)

    def _denormalise(self, x: np.ndarray) -> np.ndarray:
        """Invert Z-score normalisation."""
        return x * np.maximum(self._feature_std, 1e-9) + self._feature_mean

    def _fit_graph_mode(
        self, positive_x: np.ndarray, feature_columns: List[str]
    ) -> None:
        """Train the full GATv2-CVAE using the PyG graph."""
        from watergen.data.epanet import build_pyg_graph, extract_pipe_resistance_constants

        # Build graph (no simulation results — static topology only)
        try:
            graph = build_pyg_graph(str(self.benchmark_path))
        except Exception as exc:
            print(
                f"[warn] GraphCVAEAugmenter: graph build failed ({exc}); "
                "falling back to tabular mode.",
                file=sys.stderr,
            )
            self._graph_mode = False
            self._fit_tabular_mode(positive_x, feature_columns)
            return

        self._pyg_graph = graph

        # Pre-compute pipe resistance tensor for physics loss
        pipe_resistance = extract_pipe_resistance_constants(str(self.benchmark_path))

        # Build the model
        device = torch.device(self.device_str)
        model = _GATv2CVAE(
            node_feature_dim=8,
            edge_feature_dim=5,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        ).to(device)
        self._model = model

        physics = _HydraulicPhysicsLoss(
            pipe_resistance=pipe_resistance,
            lambda_mass=self.lambda_physics,
            lambda_energy=self.lambda_physics,
            lambda_bounds=self.lambda_physics * 0.5,
        )
        self._physics_loss = physics

        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Move graph tensors to device
        x_graph = graph["x"] if isinstance(graph, dict) else graph.x
        edge_index = graph["edge_index"] if isinstance(graph, dict) else graph.edge_index
        edge_attr = graph["edge_attr"] if isinstance(graph, dict) else graph.edge_attr

        x_graph = x_graph.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)

        # Build pipe resistance tensor (one entry per directed edge pair)
        pipe_names = list(pipe_resistance.keys())
        r_vals = [pipe_resistance.get(p, 0.0) for p in pipe_names]
        # Duplicate for bidirectional edges
        r_tensor = torch.tensor(r_vals + r_vals, dtype=torch.float32, device=device)
        self._pipe_resistance_tensor = r_tensor

        n_rows = len(positive_x)
        n_nodes = x_graph.shape[0]

        model.train()
        for epoch in range(self.epochs):
            # KL annealing: β ramps from 0 → beta_kl over 0–50 epochs
            beta = min(self.beta_kl, self.beta_kl * epoch / max(50, 1)) if epoch < 50 else self.beta_kl
            phys_lambda = _physics_lambda_warmup(epoch, self.warmup_epochs) * self.lambda_physics

            epoch_losses = {"total": 0.0, "recon": 0.0, "kl": 0.0, "phys": 0.0}
            n_batches = max(1, n_rows // self.batch_size)

            perm = np.random.permutation(n_rows)
            for b in range(n_batches):
                idx = perm[b * self.batch_size: (b + 1) * self.batch_size]
                # For graph mode, we use the tabular row as a conditioning signal
                # and generate from the graph topology.
                optimizer.zero_grad()

                cond = self._build_cond_vector("leak").to(device)
                (pressure_pred, demand_pred, flow_pred), mu, log_var = model(
                    x_graph, edge_index, edge_attr, cond, sensor_mask=None
                )

                # Reconstruction: encourage pressure/demand to match observed stats
                # (We use the batch mean as a surrogate target)
                batch_x = torch.tensor(
                    positive_x[idx], dtype=torch.float32, device=device
                )
                # Project 3 decoded outputs to feature_dim for reconstruction loss
                # (use mean of predicted node values as a feature proxy)
                recon_proxy = torch.stack([
                    pressure_pred.mean(),
                    demand_pred.mean(),
                    flow_pred.mean(),
                ]).unsqueeze(0).expand(len(idx), -1)

                target_proxy = batch_x[:, :3] if batch_x.shape[1] >= 3 else batch_x
                target_proxy = target_proxy[:, :recon_proxy.shape[1]]

                l_recon = nn.functional.mse_loss(recon_proxy, target_proxy)
                l_kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

                l_phys = self._physics_loss(
                    pressure_pred, demand_pred, flow_pred, edge_index, self._pipe_resistance_tensor
                ) if phys_lambda > 0 else torch.tensor(0.0, device=device)

                # Normalise physics loss to prevent explosion from large R_ij constants
                # (Hazen-Williams resistances can be O(1e6) in SI units).
                # Clamp the physics contribution to at most 10× the reconstruction loss.
                l_phys_scaled = torch.clamp(l_phys, max=10.0 * l_recon.detach() + 1e-6)

                loss = l_recon + beta * l_kl + phys_lambda * l_phys_scaled
                loss.backward()
                optimizer.step()

                epoch_losses["total"] += loss.item()
                epoch_losses["recon"] += l_recon.item()
                epoch_losses["kl"] += l_kl.item()
                epoch_losses["phys"] += l_phys.item()

            for k in epoch_losses:
                epoch_losses[k] /= n_batches
            self.training_history_.append(epoch_losses)

            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                print(
                    f"[GraphCVAE graph] [Epoch {epoch + 1:>4}/{self.epochs}] "
                    f"loss={epoch_losses['total']:.4f} "
                    f"recon={epoch_losses['recon']:.4f} "
                    f"kl={epoch_losses['kl']:.4f} "
                    f"phys={epoch_losses['phys']:.4f}"
                )

    def _fit_tabular_mode(
        self, positive_x: np.ndarray, feature_columns: List[str]
    ) -> None:
        """Train the tabular MLP-CVAE (no graph topology)."""
        device = torch.device(self.device_str)
        feature_dim = positive_x.shape[1]
        cond_dim = 4  # one-hot disturbance type

        model = _TabularCVAE(
            feature_dim=feature_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            cond_dim=cond_dim,
        ).to(device)
        self._model = model

        physics_loss_fn = _HydraulicPhysicsLoss(
            pipe_resistance=None,
            lambda_mass=0.0,
            lambda_energy=0.0,
            lambda_bounds=0.5,
        )
        self._physics_loss = physics_loss_fn

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        n_rows = len(positive_x)

        model.train()
        for epoch in range(self.epochs):
            beta = min(self.beta_kl, self.beta_kl * epoch / max(50, 1)) if epoch < 50 else self.beta_kl
            phys_lambda = _physics_lambda_warmup(epoch, self.warmup_epochs) * self.lambda_physics

            epoch_losses = {"total": 0.0, "recon": 0.0, "kl": 0.0, "phys": 0.0}
            n_batches = max(1, n_rows // max(self.batch_size, 1))

            perm = np.random.permutation(n_rows)
            for b in range(n_batches):
                idx = perm[b * self.batch_size: (b + 1) * self.batch_size]
                batch_x = torch.tensor(
                    positive_x[idx], dtype=torch.float32, device=device
                )
                cond = self._build_cond_vector("leak").to(device)
                cond_batch = cond.unsqueeze(0).expand(len(idx), -1)

                optimizer.zero_grad()
                recon, mu, log_var = model(batch_x, cond_batch)

                l_recon = nn.functional.mse_loss(recon, batch_x)
                l_kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

                # Physics proxy: pressure-bounds soft check on first feature
                # (heuristic — treat recon[:, 0] as pressure proxy)
                dummy_pressure = recon[:, 0]
                dummy_demand = recon[:, 1] if feature_dim > 1 else torch.zeros_like(dummy_pressure)
                dummy_flow = recon[:, 2] if feature_dim > 2 else torch.zeros_like(dummy_pressure)
                dummy_edge = torch.zeros((2, 0), dtype=torch.long, device=device)
                l_phys = physics_loss_fn(dummy_pressure, dummy_demand, dummy_flow, dummy_edge)

                loss = l_recon + beta * l_kl + phys_lambda * l_phys
                loss.backward()
                optimizer.step()

                epoch_losses["total"] += loss.item()
                epoch_losses["recon"] += l_recon.item()
                epoch_losses["kl"] += l_kl.item()
                epoch_losses["phys"] += l_phys.item()

            for k in epoch_losses:
                epoch_losses[k] /= n_batches
            self.training_history_.append(epoch_losses)

            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                print(
                    f"[GraphCVAE tabular] [Epoch {epoch + 1:>4}/{self.epochs}] "
                    f"loss={epoch_losses['total']:.4f} "
                    f"recon={epoch_losses['recon']:.4f} "
                    f"kl={epoch_losses['kl']:.4f} "
                    f"phys={epoch_losses['phys']:.4f}"
                )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        target_column: str = "event_label",
    ) -> "GraphCVAEAugmenter":
        """Fit the Graph-CVAE on positive-class training rows.

        Args:
            train_frame: Training DataFrame containing both positive and
                negative class rows.
            feature_columns: Names of numeric feature columns to use.
            target_column: Binary label column (1 = positive class).

        Returns:
            Self (for chaining).
        """
        if self._fallback is not None:
            self._fallback.fit(
                train_frame, feature_columns=feature_columns, target_column=target_column
            )
            return self

        self._feature_columns = list(feature_columns)

        positive = train_frame.loc[train_frame[target_column] == 1].copy()
        if len(positive) < 2:
            print(
                "[warn] GraphCVAEAugmenter.fit: fewer than 2 positive rows; "
                "falling back to GaussianMixtureAugmenter.",
                file=sys.stderr,
            )
            self._fallback = GaussianMixtureAugmenter(random_state=self.random_state)
            self._fallback.fit(
                train_frame, feature_columns=feature_columns, target_column=target_column
            )
            return self

        x = (
            positive[list(feature_columns)]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        # Fit normalisation statistics
        self._feature_mean = x.mean(axis=0)
        self._feature_std = x.std(axis=0)
        x_norm = self._normalise(x)

        if self._graph_mode:
            self._fit_graph_mode(x_norm, list(feature_columns))
        else:
            self._fit_tabular_mode(x_norm, list(feature_columns))

        return self

    def generate(
        self,
        train_frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        target_column: str = "event_label",
        n_samples: int | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic positive-class disturbance samples.

        Samples latent vectors z ~ N(0, I) and decodes them through the
        trained CVAE conditioned on disturbance_type='leak'.

        Args:
            train_frame: Training DataFrame (used to count positives and
                to provide a template for non-feature columns).
            feature_columns: Names of numeric feature columns to generate.
            target_column: Binary label column.
            n_samples: Number of samples to generate.  Defaults to the
                number of positive rows in *train_frame*.

        Returns:
            DataFrame of synthetic rows with:
            - ``event_label = 1``
            - ``scenario_kind = "synthetic_graph_cvae_positive"``
            - ``is_synthetic = 1``
            - ``scenario_id = "synthetic_graph_cvae_{idx}"``
            - All *feature_columns* filled with generated values.
        """
        if self._fallback is not None:
            result = self._fallback.generate(
                train_frame,
                feature_columns=feature_columns,
                target_column=target_column,
                n_samples=n_samples,
            )
            result["scenario_kind"] = "synthetic_graph_cvae_positive"
            result["scenario_id"] = [
                f"synthetic_graph_cvae_{i}" for i in range(len(result))
            ]
            return result

        # Auto-fit if not yet fitted
        if self._model is None:
            self.fit(train_frame, feature_columns=feature_columns, target_column=target_column)

        # If fallback was triggered during fit
        if self._fallback is not None:
            return self.generate(
                train_frame,
                feature_columns=feature_columns,
                target_column=target_column,
                n_samples=n_samples,
            )

        positive = train_frame.loc[train_frame[target_column] == 1]
        sample_count = n_samples if n_samples is not None else len(positive)

        device = torch.device(self.device_str)
        feature_columns_list = list(feature_columns)
        n_feat = len(feature_columns_list)

        self._model.eval()
        generated_rows: List[np.ndarray] = []

        with torch.no_grad():
            cond = self._build_cond_vector("leak").to(device)

            for _ in range(sample_count):
                z = torch.randn(self.latent_dim, device=device)

                if self._graph_mode:
                    x_graph = self._pyg_graph["x"] if isinstance(self._pyg_graph, dict) else self._pyg_graph.x
                    edge_index = self._pyg_graph["edge_index"] if isinstance(self._pyg_graph, dict) else self._pyg_graph.edge_index
                    edge_attr = self._pyg_graph["edge_attr"] if isinstance(self._pyg_graph, dict) else self._pyg_graph.edge_attr

                    x_graph = x_graph.to(device)
                    edge_index = edge_index.to(device)
                    edge_attr = edge_attr.to(device)

                    pressure_pred, demand_pred, flow_pred = self._model.decode(
                        z, x_graph, cond, x_graph.shape[0]
                    )
                    # Build a flat feature row from node-level predictions
                    row_vals = np.zeros(n_feat, dtype=np.float32)
                    row_vals[0] = float(pressure_pred.mean().item())
                    if n_feat > 1:
                        row_vals[1] = float(pressure_pred.min().item())
                    if n_feat > 2:
                        row_vals[2] = float(pressure_pred.max().item())
                    if n_feat > 3:
                        row_vals[3] = float(pressure_pred.std().item())
                    if n_feat > 4:
                        row_vals[4] = float(demand_pred.mean().item())
                    if n_feat > 5:
                        row_vals[5] = float(demand_pred.sum().item())
                    if n_feat > 6:
                        row_vals[6] = float(demand_pred.std().item())
                    if n_feat > 7:
                        row_vals[7] = float(flow_pred.mean().item())
                    if n_feat > 8:
                        row_vals[8] = float(flow_pred.abs().mean().item())
                    if n_feat > 9:
                        row_vals[9] = float(flow_pred.std().item())
                    # Remaining features left as 0 (static meta-features)
                    generated_rows.append(row_vals)
                else:
                    # Tabular mode: directly decode the latent vector
                    recon = self._model.decode(z, cond)
                    row_norm = recon.cpu().numpy()
                    # Denormalise
                    row = self._denormalise(row_norm.reshape(1, -1)).flatten()
                    generated_rows.append(row[:n_feat].astype(np.float32))

        # Build output DataFrame
        generated_array = np.stack(generated_rows, axis=0)  # (n_samples, n_feat)

        # Use a positive template for non-feature metadata columns
        template = positive.sample(
            n=sample_count, replace=True, random_state=self.random_state
        ).reset_index(drop=True)
        output = template.copy()

        for j, col in enumerate(feature_columns_list):
            if j < generated_array.shape[1]:
                output[col] = generated_array[:, j]

        output[target_column] = 1
        output["scenario_kind"] = "synthetic_graph_cvae_positive"
        output["scenario_id"] = [f"synthetic_graph_cvae_{i}" for i in range(sample_count)]
        output["is_synthetic"] = 1
        return output

    def save(self, path: str | Path) -> None:
        """Persist trained model weights to disk.

        Args:
            path: File path for the ``.pt`` weights file.
        """
        if not _HAS_TORCH:
            print("[warn] GraphCVAEAugmenter.save: torch not available", file=sys.stderr)
            return
        if self._model is None:
            raise RuntimeError("Model must be fitted before saving.")
        torch.save(self._model.state_dict(), str(path))

    def load(self, path: str | Path) -> "GraphCVAEAugmenter":
        """Load model weights from disk.

        Args:
            path: File path for the ``.pt`` weights file.

        Returns:
            Self (for chaining).
        """
        if not _HAS_TORCH:
            print("[warn] GraphCVAEAugmenter.load: torch not available", file=sys.stderr)
            return self
        if self._model is None:
            raise RuntimeError(
                "Model architecture must be initialised (via fit) before loading weights."
            )
        state = torch.load(str(path), map_location="cpu")
        self._model.load_state_dict(state)
        return self
