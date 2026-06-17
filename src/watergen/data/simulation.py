"""WNTR-based scenario generation and smoke-experiment dataset creation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import wntr

# Optional NetworkX import — used by betweenness-centrality sensor placement.
# Presence is checked at call time so the module can be imported without it.
try:
    import networkx as _nx  # noqa: F401 — availability sentinel only

    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False


@dataclass(frozen=True)
class LeakScenarioSpec:
    """Description of one simulated baseline or leak scenario."""

    scenario_id: str
    benchmark_id: str
    benchmark_path: str
    split: str
    stage: str
    kind: str
    leak_node: str | None = None
    disturbance_target: str | None = None
    leak_area: float | None = None
    start_time: int | None = None
    end_time: int | None = None
    demand_seed: int | None = None


def _safe_base_demand(wn: wntr.network.WaterNetworkModel, node_name: str) -> float:
    node = wn.get_node(node_name)
    try:
        return float(node.demand_timeseries_list[0].base_value)
    except Exception:
        return 0.0


def choose_leak_candidates(benchmark_path: Path | str, *, max_candidates: int = 5) -> list[str]:
    """Choose deterministic leak candidate junctions with positive base demand."""
    wn = wntr.network.WaterNetworkModel(str(benchmark_path))
    scored = [(name, _safe_base_demand(wn, name)) for name in wn.junction_name_list]
    positive = [item for item in scored if item[1] > 0]
    ranked = sorted(positive, key=lambda item: (-item[1], item[0]))
    return [name for name, _ in ranked[:max_candidates]]


def default_leak_window(benchmark_path: Path | str, *, rng: np.random.Generator | None = None) -> tuple[int, int]:
    """Return a leak window based on simulation duration.

    If *rng* is provided, the start and end times are sampled uniformly
    within the simulation horizon.  Otherwise the deterministic legacy
    ``[duration/3, 2*duration/3]`` window is returned.
    """
    wn = wntr.network.WaterNetworkModel(str(benchmark_path))
    duration = int(wn.options.time.duration)
    report_step = int(wn.options.time.report_timestep or wn.options.time.hydraulic_timestep or 3600)

    if rng is not None:
        earliest_start = report_step
        latest_start = max(report_step, duration // 2)
        start_time = int(rng.integers(earliest_start, latest_start + 1))
        min_event_len = max(report_step, duration // 6)
        max_event_len = max(min_event_len, duration // 3)
        event_len = int(rng.integers(min_event_len, max_event_len + 1))
        end_time = min(duration - report_step, start_time + event_len)
        if end_time <= start_time:
            end_time = min(duration, start_time + report_step)
    else:
        start_time = max(report_step, duration // 3)
        end_time = min(duration - report_step, (2 * duration) // 3)
        if end_time <= start_time:
            end_time = min(duration, start_time + report_step)
    return start_time, end_time


def choose_pipe_candidates(wn: wntr.network.WaterNetworkModel, *, max_candidates: int = 3) -> list[str]:
    """Choose deterministic pipe candidates with the largest diameters."""
    scored = []
    for name in wn.pipe_name_list:
        link = wn.get_link(name)
        diameter = float(getattr(link, "diameter", 0.0) or 0.0)
        scored.append((name, diameter))
    ranked = sorted(scored, key=lambda item: (-item[1], item[0]))
    return [name for name, _ in ranked[:max_candidates]]


def choose_pump_candidates(wn: wntr.network.WaterNetworkModel, *, max_candidates: int = 3) -> list[str]:
    """Choose deterministic pump candidates (up to all pumps in network)."""
    return list(sorted(wn.pump_name_list))[:max_candidates]


def build_leak_scenarios(
    resolved_split_manifest: dict[str, Any],
    *,
    smoke_splits: Iterable[str],
    leak_area_values: Iterable[float],
    max_candidates_per_network: int,
    disturbance_types: Iterable[str] | None = None,
    include_baseline: bool = True,
) -> list[LeakScenarioSpec]:
    """Build baseline and leak scenario specs for selected split partitions."""
    selected_splits = {str(item) for item in smoke_splits}
    enabled_disturbances = {str(item) for item in (disturbance_types or ["leak"])}
    scenarios: list[LeakScenarioSpec] = []

    for split_name, entries in resolved_split_manifest.get("partitions", {}).items():
        if split_name not in selected_splits:
            continue
        for entry in entries:
            benchmark_id = str(entry["id"])
            benchmark_path = Path(entry["path"])
            stage = str(entry["stage"])

            try:
                wn = wntr.network.WaterNetworkModel(str(benchmark_path))
            except Exception:
                continue

            if include_baseline:
                scenarios.append(
                    LeakScenarioSpec(
                        scenario_id=f"{benchmark_id}__baseline",
                        benchmark_id=benchmark_id,
                        benchmark_path=str(benchmark_path),
                        split=split_name,
                        stage=stage,
                        kind="baseline",
                    )
                )

            duration = int(wn.options.time.duration)
            report_step = int(wn.options.time.report_timestep or wn.options.time.hydraulic_timestep or 3600)
            start_time = max(report_step, duration // 3)
            end_time = min(duration - report_step, (2 * duration) // 3)
            if end_time <= start_time:
                end_time = min(duration, start_time + report_step)

            if "leak" in enabled_disturbances:
                scored = [(name, _safe_base_demand(wn, name)) for name in wn.junction_name_list]
                positive = [item for item in scored if item[1] > 0]
                ranked = sorted(positive, key=lambda item: (-item[1], item[0]))
                leak_nodes = [name for name, _ in ranked[:max_candidates_per_network]]
                for leak_node in leak_nodes:
                    for leak_area in leak_area_values:
                        scenarios.append(
                            LeakScenarioSpec(
                                scenario_id=f"{benchmark_id}__leak__{leak_node}__{leak_area:g}",
                                benchmark_id=benchmark_id,
                                benchmark_path=str(benchmark_path),
                                split=split_name,
                                stage=stage,
                                kind="leak",
                                leak_node=leak_node,
                                disturbance_target=leak_node,
                                leak_area=float(leak_area),
                                start_time=int(start_time),
                                end_time=int(end_time),
                            )
                        )

            if "pipe_closure" in enabled_disturbances:
                for pipe_name in choose_pipe_candidates(wn, max_candidates=3):
                    scenarios.append(
                        LeakScenarioSpec(
                            scenario_id=f"{benchmark_id}__pipe_closure__{pipe_name}",
                            benchmark_id=benchmark_id,
                            benchmark_path=str(benchmark_path),
                            split=split_name,
                            stage=stage,
                            kind="pipe_closure",
                            disturbance_target=pipe_name,
                        )
                    )

            if "pump_outage" in enabled_disturbances:
                for pump_name in choose_pump_candidates(wn, max_candidates=3):
                    scenarios.append(
                        LeakScenarioSpec(
                            scenario_id=f"{benchmark_id}__pump_outage__{pump_name}",
                            benchmark_id=benchmark_id,
                            benchmark_path=str(benchmark_path),
                            split=split_name,
                            stage=stage,
                            kind="pump_outage",
                            disturbance_target=pump_name,
                        )
                    )

            if "demand_surge" in enabled_disturbances:
                for junction_name in choose_leak_candidates(benchmark_path, max_candidates=1):
                    scenarios.append(
                        LeakScenarioSpec(
                            scenario_id=f"{benchmark_id}__demand_surge__{junction_name}",
                            benchmark_id=benchmark_id,
                            benchmark_path=str(benchmark_path),
                            split=split_name,
                            stage=stage,
                            kind="demand_surge",
                            disturbance_target=junction_name,
                            start_time=int(start_time),
                            end_time=int(end_time),
                        )
                    )

            if "partial_valve_closure" in enabled_disturbances:
                for pipe_name in choose_pipe_candidates(wn, max_candidates=1):
                    scenarios.append(
                        LeakScenarioSpec(
                            scenario_id=f"{benchmark_id}__partial_valve__{pipe_name}",
                            benchmark_id=benchmark_id,
                            benchmark_path=str(benchmark_path),
                            split=split_name,
                            stage=stage,
                            kind="partial_valve_closure",
                            disturbance_target=pipe_name,
                        )
                    )

    return scenarios


def scenario_specs_to_records(scenarios: Iterable[LeakScenarioSpec]) -> list[dict[str, Any]]:
    """Convert scenario specs to serializable records."""
    return [asdict(item) for item in scenarios]


def _select_coverage_columns(columns: list[str], coverage: float) -> list[str]:
    """Select a deterministic subset of columns for a given sensor coverage level."""
    if not columns:
        return []
    bounded = max(0.0, min(1.0, float(coverage)))
    count = max(1, ceil(len(columns) * bounded))
    if count >= len(columns):
        return columns
    step = max(1, len(columns) // count)
    selected = columns[::step][:count]
    if len(selected) < count:
        selected = columns[:count]
    return selected


def _aggregate_view(
    pressure_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    head_df: pd.DataFrame | None,
    *,
    coverage: float,
) -> pd.DataFrame:
    """Aggregate a sensor view at a given coverage level."""
    pressure_cols = _select_coverage_columns(list(pressure_df.columns), coverage)
    demand_cols = _select_coverage_columns(list(demand_df.columns), coverage)
    flow_cols = _select_coverage_columns(list(flow_df.columns), coverage)
    head_cols = _select_coverage_columns(list(head_df.columns), coverage) if head_df is not None else []

    pressure_view = pressure_df[pressure_cols]
    demand_view = demand_df[demand_cols]
    flow_view = flow_df[flow_cols]
    head_view = head_df[head_cols] if head_df is not None and head_cols else None

    frame = pd.DataFrame(index=pressure_view.index.copy())
    frame["time_seconds"] = pressure_view.index.astype(float)
    frame["pressure_mean"] = pressure_view.mean(axis=1)
    frame["pressure_min"] = pressure_view.min(axis=1)
    frame["pressure_max"] = pressure_view.max(axis=1)
    frame["pressure_std"] = pressure_view.std(axis=1).fillna(0.0)
    frame["demand_mean"] = demand_view.mean(axis=1)
    frame["demand_sum"] = demand_view.sum(axis=1)
    frame["demand_std"] = demand_view.std(axis=1).fillna(0.0)
    frame["flow_mean"] = flow_view.mean(axis=1)
    frame["flow_abs_mean"] = flow_view.abs().mean(axis=1)
    frame["flow_std"] = flow_view.std(axis=1).fillna(0.0)

    if head_view is not None and not head_view.empty:
        frame["tank_head_mean"] = head_view.mean(axis=1)
        frame["tank_head_std"] = head_view.std(axis=1).fillna(0.0)
    else:
        frame["tank_head_mean"] = 0.0
        frame["tank_head_std"] = 0.0

    frame["sensor_coverage"] = float(coverage)
    frame["observed_junction_count"] = len(pressure_cols)
    frame["observed_link_count"] = len(flow_cols)

    # Temporal ramp features (detect monotonic trends — key signature of developing leaks)
    frame["pressure_trend"] = pressure_view.mean(axis=1).diff().fillna(0.0)  # First difference
    frame["pressure_trend_cumsum"] = pressure_view.mean(axis=1).diff().fillna(0.0).cumsum()
    frame["demand_trend"] = demand_view.mean(axis=1).diff().fillna(0.0)

    return frame


def _aggregate_view_with_placement(
    pressure_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    head_df: pd.DataFrame | None,
    *,
    node_names: list[str],
    coverage: float,
) -> pd.DataFrame:
    """Aggregate a sensor view for an explicit list of observed node names.

    Unlike :func:`_aggregate_view`, which derives the observed set from
    *coverage* via :func:`_select_coverage_columns`, this variant uses a
    caller-supplied list of junction node names as the observed sensor set.
    Flow links are still selected by coverage fraction (the flow topology is
    tied to junction placement but we have no direct node→link mapping here).

    Args:
        pressure_df: Full-network pressure time-series (columns = node names).
        demand_df: Full-network demand time-series (columns = node names).
        flow_df: Full-network flow time-series (columns = link names).
        head_df: Tank head time-series, or None if no tanks.
        node_names: Explicit list of junction node names to treat as the
            observed sensor set.  Names absent from *pressure_df* are silently
            ignored.
        coverage: Numeric coverage fraction recorded in the output frame and
            used to select the flow subset.

    Returns:
        Aggregated feature DataFrame with the same columns as
        :func:`_aggregate_view`.
    """
    # Intersect requested nodes with available columns
    available_nodes = set(pressure_df.columns)
    valid_nodes = [n for n in node_names if n in available_nodes]
    if not valid_nodes:
        # Fall back to full coverage if no requested nodes are in the data
        valid_nodes = list(pressure_df.columns)

    pressure_cols = valid_nodes
    demand_cols = [c for c in valid_nodes if c in demand_df.columns]
    # Flow links: select by coverage fraction (no explicit placement mapping)
    flow_cols = _select_coverage_columns(list(flow_df.columns), coverage)
    head_cols = (
        _select_coverage_columns(list(head_df.columns), coverage)
        if head_df is not None
        else []
    )

    pressure_view = pressure_df[pressure_cols]
    demand_view = demand_df[demand_cols] if demand_cols else demand_df.iloc[:, :0]
    flow_view = flow_df[flow_cols]
    head_view = head_df[head_cols] if head_df is not None and head_cols else None

    frame = pd.DataFrame(index=pressure_view.index.copy())
    frame["time_seconds"] = pressure_view.index.astype(float)
    frame["pressure_mean"] = pressure_view.mean(axis=1)
    frame["pressure_min"] = pressure_view.min(axis=1)
    frame["pressure_max"] = pressure_view.max(axis=1)
    frame["pressure_std"] = pressure_view.std(axis=1).fillna(0.0)
    frame["demand_mean"] = demand_view.mean(axis=1) if not demand_view.empty else 0.0
    frame["demand_sum"] = demand_view.sum(axis=1) if not demand_view.empty else 0.0
    frame["demand_std"] = (
        demand_view.std(axis=1).fillna(0.0) if not demand_view.empty else 0.0
    )
    frame["flow_mean"] = flow_view.mean(axis=1)
    frame["flow_abs_mean"] = flow_view.abs().mean(axis=1)
    frame["flow_std"] = flow_view.std(axis=1).fillna(0.0)

    if head_view is not None and not head_view.empty:
        frame["tank_head_mean"] = head_view.mean(axis=1)
        frame["tank_head_std"] = head_view.std(axis=1).fillna(0.0)
    else:
        frame["tank_head_mean"] = 0.0
        frame["tank_head_std"] = 0.0

    frame["sensor_coverage"] = float(coverage)
    frame["observed_junction_count"] = len(pressure_cols)
    frame["observed_link_count"] = len(flow_cols)
    return frame


def _build_raw_view(
    pressure_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    *,
    coverage: float,
) -> pd.DataFrame:
    """Build a raw time-series view (one row per timestep, all sensor columns) at a given coverage level."""
    pressure_cols = _select_coverage_columns(list(pressure_df.columns), coverage)
    demand_cols = _select_coverage_columns(list(demand_df.columns), coverage)
    flow_cols = _select_coverage_columns(list(flow_df.columns), coverage)

    pressure_view = pressure_df[pressure_cols].copy()
    pressure_view.columns = [f"pressure__{c}" for c in pressure_cols]

    demand_view = demand_df[demand_cols].copy()
    demand_view.columns = [f"demand__{c}" for c in demand_cols]

    flow_view = flow_df[flow_cols].copy()
    flow_view.columns = [f"flow__{c}" for c in flow_cols]

    frame = pd.DataFrame(index=pressure_view.index.copy())
    frame["time_seconds"] = pressure_view.index.astype(float)
    frame = pd.concat([frame, pressure_view, demand_view, flow_view], axis=1)
    frame["sensor_coverage"] = float(coverage)
    frame["observed_junction_count"] = len(pressure_cols)
    frame["observed_link_count"] = len(flow_cols)
    return frame


def compute_betweenness_centrality_placement(
    benchmark_path: str | Path,
    *,
    coverage: float,
    random_state: int = 42,
) -> list[str]:
    """Select sensor nodes using graph betweenness centrality.

    Nodes with highest betweenness centrality lie on the most hydraulic
    flow paths and provide maximum observability of network-wide events.

    Args:
        benchmark_path: Path to EPANET .inp file.
        coverage: Fraction of junction nodes to instrument (e.g. 0.25, 0.5,
            0.75, 1.0).
        random_state: Seed for tie-breaking when centrality scores are equal.

    Returns:
        List of junction node names sorted by descending betweenness
        centrality, truncated to ``coverage`` fraction of all junctions.
        Falls back to alphabetical ordering when NetworkX is unavailable.
    """
    wn = wntr.network.WaterNetworkModel(str(benchmark_path))

    if not _HAS_NETWORKX:
        # Graceful fallback: alphabetical ordering when NetworkX is absent
        junctions = sorted(wn.junction_name_list)
        k = max(1, int(len(junctions) * coverage))
        return junctions[:k]

    import networkx as nx

    G = wn.get_graph()  # Returns NetworkX DiGraph
    G_undirected = G.to_undirected()
    centrality = nx.betweenness_centrality(G_undirected)

    # Filter to junction nodes only (exclude tanks, reservoirs, etc.)
    junction_set = set(wn.junction_name_list)
    junction_centrality = {
        node: score for node, score in centrality.items() if node in junction_set
    }

    if not junction_centrality:
        # No junction nodes in centrality result — fall back to sorted list
        junctions = sorted(wn.junction_name_list)
        k = max(1, int(len(junctions) * coverage))
        return junctions[:k]

    # Sort by centrality descending, then by name for determinism on ties
    sorted_nodes = sorted(
        junction_centrality.keys(),
        key=lambda n: (-junction_centrality[n], n),
    )

    k = max(1, int(len(sorted_nodes) * coverage))
    return sorted_nodes[:k]


def compute_khop_observability(
    sensor_nodes: list[str],
    benchmark_path: str | Path,
    *,
    h: int = 2,
) -> float:
    """Compute k-hop observability coverage of a sensor placement.

    Coverage(S, h) = |{v ∈ V : min_{s ∈ S} d_G(v, s) ≤ h}| / |V|

    This metric captures what fraction of the network is within *h* pipe
    segments of at least one sensor node — more meaningful than raw node
    fraction because it accounts for network topology.

    Args:
        sensor_nodes: List of junction node names that are instrumented.
        benchmark_path: Path to EPANET .inp file.
        h: Maximum hop distance from a sensor to count a node as "covered".

    Returns:
        Fraction of all network nodes (junctions + tanks + reservoirs) that
        are within *h* hops of at least one sensor.  Returns 0.0 if there
        are no sensor nodes or if NetworkX is unavailable (falls back to the
        raw fraction of junctions instrumented).
    """
    wn = wntr.network.WaterNetworkModel(str(benchmark_path))
    all_nodes = (
        list(wn.junction_name_list)
        + list(wn.tank_name_list)
        + list(wn.reservoir_name_list)
    )
    total = len(all_nodes)
    if total == 0 or not sensor_nodes:
        return 0.0

    if not _HAS_NETWORKX:
        # Fallback: plain fraction of junctions instrumented
        junction_set = set(wn.junction_name_list)
        instrumented = sum(1 for n in sensor_nodes if n in junction_set)
        return instrumented / max(1, len(junction_set))

    import networkx as nx

    G = wn.get_graph().to_undirected()
    sensor_set = set(sensor_nodes)

    covered = set()
    for sensor in sensor_set:
        if sensor not in G:
            continue
        # BFS up to depth h from this sensor
        reachable = nx.single_source_shortest_path_length(G, sensor, cutoff=h)
        covered.update(reachable.keys())

    return len(covered) / total


def add_demand_noise(
    wn: "wntr.network.WaterNetworkModel",
    *,
    noise_fraction: float = 0.20,
    random_state: int = 42,
) -> "wntr.network.WaterNetworkModel":
    """Add stochastic demand perturbation to simulate real-world demand uncertainty.

    Real WDN demand is stochastic with ±15-25% node-level variation due to:
    - Household occupancy patterns
    - Industrial load schedules
    - Seasonal temperature effects

    Adding this noise makes the simulated problem more realistic and prevents
    artificially inflated F1 scores on clean simulated data.

    Args:
        wn: WaterNetworkModel to modify in place
        noise_fraction: Fraction of base demand to use as noise std (default 0.20 = ±20%)
        random_state: For reproducibility
    """
    rng = np.random.default_rng(random_state)
    for junction_name in wn.junction_name_list:
        junction = wn.get_node(junction_name)
        try:
            base = float(junction.demand_timeseries_list[0].base_value)
            if base > 0:
                noise = rng.normal(1.0, noise_fraction)
                noise = max(0.5, min(2.0, noise))  # Clip to [0.5, 2.0] multiplier
                junction.demand_timeseries_list[0].base_value = base * noise
        except (IndexError, AttributeError):
            pass
    return wn


def _ensure_eps_simulation(wn: "wntr.network.WaterNetworkModel") -> None:
    """Ensure the simulation uses Extended Period Simulation (EPS) mode.

    EPS simulates time-varying demands and produces temporal pressure/flow
    profiles — essential for learning disturbance ramp signatures.
    Steady-state snapshots destroy the temporal diagnostic signal.
    """
    duration = int(wn.options.time.duration)
    if duration <= 0:
        # Set to 24 hours if not configured
        wn.options.time.duration = 24 * 3600

    report_timestep = int(wn.options.time.report_timestep or 0)
    if report_timestep <= 0 or report_timestep > 3600:
        # Ensure at least hourly reporting
        wn.options.time.report_timestep = 3600

    hydraulic_timestep = int(wn.options.time.hydraulic_timestep or 0)
    if hydraulic_timestep <= 0:
        wn.options.time.hydraulic_timestep = 3600


def simulate_scenario(
    spec: LeakScenarioSpec,
    *,
    sensor_coverage_levels: Iterable[float] | None = None,
    return_raw: bool = False,
    placement_strategy: str = "random",
    sensor_nodes: list[str] | None = None,
    demand_noise: bool = False,
    demand_noise_fraction: float = 0.20,
) -> pd.DataFrame:
    """Run one WNTR simulation and convert it into a tabular time-series dataset.

    Parameters
    ----------
    spec:
        Scenario specification describing which network and disturbance to simulate.
    sensor_coverage_levels:
        Fraction of sensors to include.  Defaults to ``[1.0]`` (all sensors).
    return_raw:
        When ``True``, return one row per timestep with individual sensor columns
        (``pressure__<node>``, ``demand__<node>``, ``flow__<link>``) instead of the
        aggregated summary statistics produced by :func:`_aggregate_view`.
    placement_strategy:
        Sensor placement strategy to use for the aggregated view.  One of:

        - ``"random"`` *(default)*: existing behaviour — use
          :func:`_select_coverage_columns` to pick a deterministic subset by
          coverage fraction.
        - ``"centrality"``: use :func:`compute_betweenness_centrality_placement`
          to select nodes by graph betweenness centrality.
        - ``"explicit"``: use the caller-supplied *sensor_nodes* list directly.

        This parameter is ignored when ``return_raw=True``.
    sensor_nodes:
        Explicit list of junction node names to observe.  Only used when
        *placement_strategy* is ``"explicit"``.
    demand_noise:
        When ``True``, add stochastic demand perturbation before running the
        simulation.  See :func:`add_demand_noise` for details.
    demand_noise_fraction:
        Fraction of base demand to use as noise std (default 0.20 = ±20%).
        Only used when *demand_noise* is ``True``.
    """
    wn = wntr.network.WaterNetworkModel(spec.benchmark_path)
    _ensure_eps_simulation(wn)

    # Per-scenario seeded demand noise takes precedence: when a spec carries an
    # explicit demand_seed, apply reproducible noise with that seed. This lets a
    # caller generate many distinct normal-operation (baseline) realisations and
    # apply the SAME noise treatment to disturbance scenarios (no clean-vs-noisy
    # confound between the disturbance and baseline classes).
    if spec.demand_seed is not None:
        add_demand_noise(wn, noise_fraction=demand_noise_fraction, random_state=spec.demand_seed)
    elif demand_noise:
        add_demand_noise(wn, noise_fraction=demand_noise_fraction)

    if spec.kind == "leak" and spec.leak_node and spec.leak_area:
        junction = wn.get_node(spec.leak_node)
        junction.add_leak(
            wn,
            area=spec.leak_area,
            start_time=spec.start_time,
            end_time=spec.end_time,
        )
    elif spec.kind in {"pipe_closure", "pump_outage"} and spec.disturbance_target:
        link = wn.get_link(spec.disturbance_target)
        link.initial_status = wntr.network.base.LinkStatus.Closed

    elif spec.kind == "demand_surge" and spec.disturbance_target:
        # Apply 5x demand multiplier at the target node during the disturbance window
        junction = wn.get_node(spec.disturbance_target)
        try:
            base = float(junction.demand_timeseries_list[0].base_value)
            junction.demand_timeseries_list[0].base_value = base * 5.0
        except (IndexError, AttributeError):
            pass

    elif spec.kind == "partial_valve_closure" and spec.disturbance_target:
        # Set pipe to 50% resistance (simulate partial closure)
        try:
            link = wn.get_link(spec.disturbance_target)
            link.initial_status = wntr.network.base.LinkStatus.Open
            # Increase roughness to simulate partial closure headloss
            if hasattr(link, "roughness"):
                link.roughness = max(1.0, link.roughness * 0.3)
        except Exception:
            pass

    try:
        simulator = wntr.sim.WNTRSimulator(wn)
        results = simulator.run_sim()
    except Exception as exc:
        err_msg = str(exc).lower()
        if "pump speed" in err_msg or "pump speeds" in err_msg:
            # Fall back to EpanetSimulator for networks with variable-speed pumps
            import sys as _sys
            print(
                f"[info] WNTRSimulator failed (pump speed), retrying with EpanetSimulator: {exc}",
                file=_sys.stderr,
            )
            simulator = wntr.sim.EpanetSimulator(wn)
            results = simulator.run_sim()
        else:
            raise

    junctions = wn.junction_name_list
    pressure_df = results.node["pressure"][junctions]
    demand_df = results.node["demand"][junctions]
    leak_df = results.node.get("leak_demand")
    if leak_df is not None:
        leak_df = leak_df[junctions]
    flow_df = results.link["flowrate"]
    tank_names = wn.tank_name_list
    head_df = results.node["head"][tank_names] if tank_names else None

    coverage_levels = list(sensor_coverage_levels or [1.0])
    frames: list[pd.DataFrame] = []
    for coverage in coverage_levels:
        if return_raw:
            frame = _build_raw_view(
                pressure_df,
                demand_df,
                flow_df,
                coverage=float(coverage),
            )
        elif placement_strategy == "centrality":
            placed_nodes = compute_betweenness_centrality_placement(
                spec.benchmark_path,
                coverage=float(coverage),
            )
            frame = _aggregate_view_with_placement(
                pressure_df,
                demand_df,
                flow_df,
                head_df,
                node_names=placed_nodes,
                coverage=float(coverage),
            )
        elif placement_strategy == "explicit":
            explicit_nodes = list(sensor_nodes) if sensor_nodes else list(junctions)
            frame = _aggregate_view_with_placement(
                pressure_df,
                demand_df,
                flow_df,
                head_df,
                node_names=explicit_nodes,
                coverage=float(coverage),
            )
        else:
            # "random" — original behaviour
            frame = _aggregate_view(
                pressure_df,
                demand_df,
                flow_df,
                head_df,
                coverage=float(coverage),
            )
        frame["benchmark_id"] = spec.benchmark_id
        frame["scenario_id"] = spec.scenario_id
        frame["scenario_kind"] = spec.kind
        frame["split"] = spec.split
        frame["stage"] = spec.stage
        frame["leak_node"] = spec.leak_node or ""
        frame["disturbance_target"] = spec.disturbance_target or ""
        frame["leak_area"] = float(spec.leak_area or 0.0)
        frame["event_label"] = 0

        if spec.kind == "leak" and spec.start_time is not None and spec.end_time is not None:
            mask = (frame["time_seconds"] >= spec.start_time) & (frame["time_seconds"] <= spec.end_time)
            frame.loc[mask, "event_label"] = 1
        elif spec.kind != "baseline":
            frame["event_label"] = 1
        frames.append(frame.reset_index(drop=True))

    return pd.concat(frames, axis=0, ignore_index=True)


def simulate_scenario_raw(
    spec: LeakScenarioSpec,
    *,
    sensor_coverage_levels: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Run one WNTR simulation and return raw per-timestep sensor columns.

    Convenience wrapper around :func:`simulate_scenario` with ``return_raw=True``.
    Each row corresponds to one simulation timestep; sensor columns are named
    ``pressure__<node>``, ``demand__<node>``, and ``flow__<link>``.
    """
    return simulate_scenario(spec, sensor_coverage_levels=sensor_coverage_levels, return_raw=True)


def simulate_many(
    scenarios: Iterable[LeakScenarioSpec],
    *,
    sensor_coverage_levels: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Simulate multiple scenarios and concatenate the resulting rows."""
    frames = [
        simulate_scenario(spec, sensor_coverage_levels=sensor_coverage_levels)
        for spec in scenarios
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)
