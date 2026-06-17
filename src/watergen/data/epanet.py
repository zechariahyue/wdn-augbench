"""Lightweight EPANET INP parsing and summary utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

# Optional heavy dependencies — checked at call time.
try:
    import wntr as _wntr  # noqa: F401

    _HAS_WNTR = True
except ImportError:
    _HAS_WNTR = False

try:
    import networkx as _nx  # noqa: F401

    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

try:
    import torch as _torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from torch_geometric.data import Data as _PyGData  # noqa: F401

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


@dataclass(frozen=True)
class EpanetSummary:
    """Basic structural summary for an EPANET input file."""

    path: Path
    section_names: List[str]
    junctions: int
    reservoirs: int
    tanks: int
    pipes: int
    pumps: int
    valves: int
    units: str | None = None
    headloss: str | None = None


SECTION_PATTERN = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")
COUNT_SECTIONS = {
    "JUNCTIONS": "junctions",
    "RESERVOIRS": "reservoirs",
    "TANKS": "tanks",
    "PIPES": "pipes",
    "PUMPS": "pumps",
    "VALVES": "valves",
}


def parse_epanet_sections(path: Path | str) -> Dict[str, List[str]]:
    """Parse an EPANET ``.inp`` file into section -> data lines.

    Repeated sections are merged in order, and comments/blank lines are ignored.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"INP file not found: {file_path}")

    sections: Dict[str, List[str]] = {}
    current_section: str | None = None

    for raw_line in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        match = SECTION_PATTERN.match(stripped)
        if match:
            current_section = match.group("name").strip().upper()
            sections.setdefault(current_section, [])
            continue

        if current_section is None:
            continue

        # EPANET comments start with ';'. Drop inline comments as well.
        no_comment = raw_line.split(";", 1)[0].strip()
        if not no_comment:
            continue
        sections[current_section].append(no_comment)

    return sections


def parse_options(sections: Dict[str, List[str]]) -> Dict[str, str]:
    """Extract key-value pairs from the ``[OPTIONS]`` section."""
    options: Dict[str, str] = {}
    for line in sections.get("OPTIONS", []):
        tokens = line.split()
        if len(tokens) < 2:
            continue
        key = " ".join(tokens[:-1]).strip().upper()
        value = tokens[-1].strip()
        options[key] = value
    return options


def summarize_epanet(path: Path | str) -> EpanetSummary:
    """Build a basic network summary for one EPANET file."""
    file_path = Path(path)
    sections = parse_epanet_sections(file_path)
    options = parse_options(sections)

    counts = {field: 0 for field in COUNT_SECTIONS.values()}
    for section_name, field_name in COUNT_SECTIONS.items():
        counts[field_name] = len(sections.get(section_name, []))

    return EpanetSummary(
        path=file_path,
        section_names=sorted(sections.keys()),
        units=options.get("UNITS"),
        headloss=options.get("HEADLOSS"),
        **counts,
    )


def summarize_many(paths: Sequence[Path | str]) -> List[EpanetSummary]:
    """Summarize multiple EPANET files in deterministic order."""
    summaries = [summarize_epanet(path) for path in paths]
    return sorted(summaries, key=lambda item: str(item.path))


def filter_existing_inp(paths: Iterable[Path | str]) -> List[Path]:
    """Return existing ``.inp`` files from a mixed path iterable."""
    existing: List[Path] = []
    for item in paths:
        candidate = Path(item)
        if candidate.suffix.lower() == ".inp" and candidate.is_file():
            existing.append(candidate)
    return sorted(set(existing))


# ---------------------------------------------------------------------------
# PyTorch Geometric graph construction (P11)
# ---------------------------------------------------------------------------

def extract_pipe_resistance_constants(benchmark_path: str | Path) -> Dict[str, float]:
    """Pre-compute Hazen-Williams resistance constants for all pipes.

    Computes the static per-pipe resistance coefficient used in the
    energy conservation (headloss) physics loss term:

        R_ij = 10.67 * L_ij / (C_ij^1.852 * D_ij^4.87)

    where L is pipe length [m], C is the Hazen-Williams roughness
    coefficient (dimensionless), and D is pipe diameter [m].

    These constants depend only on the network geometry and can be
    pre-computed once and cached to avoid recomputation during training.

    Args:
        benchmark_path: Path to an EPANET ``.inp`` file.

    Returns:
        Dictionary mapping pipe name → resistance constant R_ij [float].
        Returns an empty dict if WNTR is unavailable or no pipes exist.
    """
    if not _HAS_WNTR:
        import sys
        print("[warn] wntr not available; extract_pipe_resistance_constants returning empty dict", file=sys.stderr)
        return {}

    import wntr

    wn = wntr.network.WaterNetworkModel(str(benchmark_path))
    resistance: Dict[str, float] = {}

    for pipe_name in wn.pipe_name_list:
        try:
            link = wn.get_link(pipe_name)
            length = float(getattr(link, "length", 0.0) or 0.0)
            diameter = float(getattr(link, "diameter", 0.001) or 0.001)
            roughness = float(getattr(link, "roughness", 100.0) or 100.0)

            # Avoid division by zero / near-zero
            if diameter < 1e-6:
                diameter = 1e-6
            if roughness < 1e-3:
                roughness = 1e-3

            # Hazen-Williams resistance: R = 10.67 * L / (C^1.852 * D^4.87)
            r_ij = 10.67 * length / ((roughness ** 1.852) * (diameter ** 4.87))
            resistance[pipe_name] = float(r_ij)
        except Exception:
            resistance[pipe_name] = 0.0

    return resistance


def build_pyg_graph(
    benchmark_path: str | Path,
    *,
    simulation_results: Any | None = None,
    sensor_mask: List[str] | None = None,
) -> "Any":
    """Convert an EPANET network to a PyTorch Geometric graph representation.

    Constructs a graph where nodes correspond to junctions, tanks, and
    reservoirs and edges correspond to pipes (bidirectional).

    Node features (dimension 8 per node):
        0. pressure_norm: normalised pressure head [0,1] from simulation or 0.0
        1. demand_base: base demand from the INP file
        2. elevation: node elevation
        3. node_type: scalar encoding [junction=0.0, tank=0.5, reservoir=1.0]
        4. sensor_flag: 1.0 if this node appears in *sensor_mask*, else 0.0
        5. betweenness_centrality: graph centrality, normalised to [0,1]
        6. node_degree: number of connected pipes, normalised by network max
        7. demand_multiplier: 1.0 (static; updated if simulation available)

    Edge features (dimension 5 per directed edge pair):
        0. flow_norm: normalised absolute flow rate (simulation or 0.0)
        1. velocity_norm: normalised velocity (simulation or 0.0)
        2. headloss_norm: normalised headloss (simulation or 0.0)
        3. length_norm: normalised pipe length
        4. diameter_norm: normalised pipe diameter

    The returned object also exposes:
        ``pipe_resistance``: dict mapping pipe name → Hazen-Williams R_ij

    Args:
        benchmark_path: Path to an EPANET ``.inp`` file.
        simulation_results: Optional WNTR ``SimulationResults`` object.  When
            provided, time-averaged hydraulic quantities are used as node/edge
            features.  When ``None``, only static topology features are used.
        sensor_mask: Optional list of node names that carry real sensors.
            Sets ``sensor_flag=1`` for those nodes.

    Returns:
        A ``torch_geometric.data.Data`` object when PyTorch Geometric is
        available, otherwise a plain ``dict`` with the same arrays under keys
        ``"x"``, ``"edge_index"``, ``"edge_attr"``, and ``"pipe_resistance"``.
    """
    if not _HAS_WNTR:
        raise ImportError(
            "wntr is required for build_pyg_graph. "
            "Install it with: pip install wntr"
        )

    import wntr
    import numpy as np

    wn = wntr.network.WaterNetworkModel(str(benchmark_path))

    # ---- Build ordered node list ----------------------------------------
    junction_names: List[str] = list(wn.junction_name_list)
    tank_names: List[str] = list(wn.tank_name_list)
    reservoir_names: List[str] = list(wn.reservoir_name_list)
    all_node_names: List[str] = junction_names + tank_names + reservoir_names
    node_index: Dict[str, int] = {name: i for i, name in enumerate(all_node_names)}
    n_nodes = len(all_node_names)

    sensor_set = set(sensor_mask) if sensor_mask else set()

    # ---- Betweenness centrality (NetworkX) --------------------------------
    centrality: Dict[str, float] = {}
    if _HAS_NETWORKX:
        import networkx as nx
        G_nx = wn.get_graph().to_undirected()
        raw_centrality = nx.betweenness_centrality(G_nx)
        max_c = max(raw_centrality.values()) if raw_centrality else 1.0
        if max_c < 1e-12:
            max_c = 1.0
        centrality = {node: v / max_c for node, v in raw_centrality.items()}

    # ---- Node degree -------------------------------------------------------
    pipe_endpoint_count: Dict[str, int] = {name: 0 for name in all_node_names}
    for pipe_name in wn.pipe_name_list:
        try:
            link = wn.get_link(pipe_name)
            start = str(link.start_node_name)
            end = str(link.end_node_name)
            if start in pipe_endpoint_count:
                pipe_endpoint_count[start] += 1
            if end in pipe_endpoint_count:
                pipe_endpoint_count[end] += 1
        except Exception:
            pass
    max_degree = max(pipe_endpoint_count.values()) if pipe_endpoint_count else 1
    if max_degree < 1:
        max_degree = 1

    # ---- Simulation-derived node features ---------------------------------
    # Compute per-node time-averaged pressure and demand if simulation given.
    pressure_mean: Dict[str, float] = {}
    demand_mean: Dict[str, float] = {}
    if simulation_results is not None:
        try:
            pressure_df = simulation_results.node["pressure"]
            for col in pressure_df.columns:
                pressure_mean[col] = float(pressure_df[col].mean())
            demand_df = simulation_results.node["demand"]
            for col in demand_df.columns:
                demand_mean[col] = float(demand_df[col].mean())
        except Exception:
            pass

    # Normalise pressure to [0, 1]
    if pressure_mean:
        p_vals = list(pressure_mean.values())
        p_min, p_max = min(p_vals), max(p_vals)
        p_range = max(p_max - p_min, 1e-9)
        pressure_norm: Dict[str, float] = {
            name: (v - p_min) / p_range for name, v in pressure_mean.items()
        }
    else:
        pressure_norm = {}

    # ---- Build node feature matrix ----------------------------------------
    x = np.zeros((n_nodes, 8), dtype=np.float32)

    for i, node_name in enumerate(all_node_names):
        try:
            node = wn.get_node(node_name)
            elevation = float(getattr(node, "elevation", 0.0) or 0.0)
            # Base demand — only junctions have demand_timeseries_list
            base_demand = 0.0
            try:
                base_demand = float(node.demand_timeseries_list[0].base_value)
            except Exception:
                pass
        except Exception:
            elevation = 0.0
            base_demand = 0.0

        # node_type encoding
        if node_name in junction_names:
            node_type = 0.0
        elif node_name in tank_names:
            node_type = 0.5
        else:
            node_type = 1.0  # reservoir

        x[i, 0] = pressure_norm.get(node_name, 0.0)
        x[i, 1] = base_demand
        x[i, 2] = elevation
        x[i, 3] = node_type
        x[i, 4] = 1.0 if node_name in sensor_set else 0.0
        x[i, 5] = centrality.get(node_name, 0.0)
        x[i, 6] = pipe_endpoint_count.get(node_name, 0) / max_degree
        x[i, 7] = 1.0  # demand_multiplier default

    # ---- Build edge index and edge features --------------------------------
    pipe_resistance = extract_pipe_resistance_constants(benchmark_path)

    # Gather pipe static features for normalisation
    lengths: List[float] = []
    diameters: List[float] = []
    for pipe_name in wn.pipe_name_list:
        try:
            link = wn.get_link(pipe_name)
            lengths.append(float(getattr(link, "length", 0.0) or 0.0))
            diameters.append(float(getattr(link, "diameter", 0.001) or 0.001))
        except Exception:
            lengths.append(0.0)
            diameters.append(0.001)

    max_len = max(lengths) if lengths else 1.0
    max_diam = max(diameters) if diameters else 1.0
    if max_len < 1e-9:
        max_len = 1.0
    if max_diam < 1e-9:
        max_diam = 1.0

    # Simulation-derived edge features (time-averaged per pipe)
    flow_mean_edge: Dict[str, float] = {}
    headloss_mean_edge: Dict[str, float] = {}
    velocity_mean_edge: Dict[str, float] = {}
    if simulation_results is not None:
        try:
            flow_df = simulation_results.link["flowrate"]
            for col in flow_df.columns:
                flow_mean_edge[col] = float(flow_df[col].abs().mean())
        except Exception:
            pass
        try:
            headloss_df = simulation_results.link["headloss"]
            for col in headloss_df.columns:
                headloss_mean_edge[col] = float(headloss_df[col].abs().mean())
        except Exception:
            pass
        try:
            velocity_df = simulation_results.link["velocity"]
            for col in velocity_df.columns:
                velocity_mean_edge[col] = float(velocity_df[col].abs().mean())
        except Exception:
            pass

    # Normalise edge scalars
    max_flow = max(flow_mean_edge.values()) if flow_mean_edge else 1.0
    max_headloss = max(headloss_mean_edge.values()) if headloss_mean_edge else 1.0
    max_velocity = max(velocity_mean_edge.values()) if velocity_mean_edge else 1.0
    for d, fallback in [
        (max_flow, 1.0), (max_headloss, 1.0), (max_velocity, 1.0)
    ]:
        pass  # in-place done below with local references
    if max_flow < 1e-12:
        max_flow = 1.0
    if max_headloss < 1e-12:
        max_headloss = 1.0
    if max_velocity < 1e-12:
        max_velocity = 1.0

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_features: List[List[float]] = []

    for idx_p, pipe_name in enumerate(wn.pipe_name_list):
        try:
            link = wn.get_link(pipe_name)
            src_name = str(link.start_node_name)
            dst_name = str(link.end_node_name)
            if src_name not in node_index or dst_name not in node_index:
                continue
            src_i = node_index[src_name]
            dst_i = node_index[dst_name]

            flow_n = flow_mean_edge.get(pipe_name, 0.0) / max_flow
            vel_n = velocity_mean_edge.get(pipe_name, 0.0) / max_velocity
            hl_n = headloss_mean_edge.get(pipe_name, 0.0) / max_headloss
            len_n = lengths[idx_p] / max_len
            diam_n = diameters[idx_p] / max_diam

            feat = [flow_n, vel_n, hl_n, len_n, diam_n]

            # Add both directed edges (undirected pipe → bidirectional edges)
            edge_src.extend([src_i, dst_i])
            edge_dst.extend([dst_i, src_i])
            edge_features.extend([feat, feat])
        except Exception:
            continue

    edge_index_np = np.array([edge_src, edge_dst], dtype=np.int64)
    edge_attr_np = np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 5), dtype=np.float32)

    if _HAS_TORCH and _HAS_PYG:
        import torch
        from torch_geometric.data import Data

        graph = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index_np),
            edge_attr=torch.from_numpy(edge_attr_np),
        )
        graph.node_names = all_node_names  # type: ignore[assignment]
        graph.pipe_resistance = pipe_resistance  # type: ignore[assignment]
        graph.num_nodes = n_nodes
        return graph
    else:
        # Graceful fallback — return a plain dict with numpy arrays
        return {
            "x": x,
            "edge_index": edge_index_np,
            "edge_attr": edge_attr_np,
            "node_names": all_node_names,
            "pipe_resistance": pipe_resistance,
            "num_nodes": n_nodes,
        }
