"""Data access and dataset schema utilities for Plan A / Plan B."""

from .acwa import build_acwa_case_study_frame
from .epanet import (
    EpanetSummary,
    build_pyg_graph,
    extract_pipe_resistance_constants,
    parse_epanet_sections,
    parse_options,
    summarize_epanet,
    summarize_many,
)
from .io import discover_files, load_inp_paths
from .manifests import collect_benchmark_entries, load_yaml_file, resolve_path, resolve_split_manifest
from .schema import FeatureSchema, SimulationConfig, SplitConfig
from .simulation import (
    LeakScenarioSpec,
    build_leak_scenarios,
    compute_betweenness_centrality_placement,
    compute_khop_observability,
    scenario_specs_to_records,
    simulate_many,
    simulate_scenario,
    simulate_scenario_raw,
)

__all__ = [
    "EpanetSummary",
    "FeatureSchema",
    "LeakScenarioSpec",
    "SimulationConfig",
    "SplitConfig",
    "build_acwa_case_study_frame",
    "build_leak_scenarios",
    "build_pyg_graph",
    "collect_benchmark_entries",
    "compute_betweenness_centrality_placement",
    "compute_khop_observability",
    "discover_files",
    "extract_pipe_resistance_constants",
    "load_yaml_file",
    "load_inp_paths",
    "parse_epanet_sections",
    "parse_options",
    "resolve_path",
    "resolve_split_manifest",
    "scenario_specs_to_records",
    "simulate_many",
    "simulate_scenario",
    "simulate_scenario_raw",
    "summarize_epanet",
    "summarize_many",
]
