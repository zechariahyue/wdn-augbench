import json, importlib.util, pathlib, sys
here=pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))
spec=importlib.util.spec_from_file_location("rsa", here/"run_sensitivity_analysis.py")
rsa=importlib.util.module_from_spec(spec); spec.loader.exec_module(rsa)
jp=rsa.OUTPUT_DIR/"sensitivity_results.json"
print("loading existing results:", jp, "exists:", jp.exists())
data=json.load(open(jp))
rsa.plot_sensitivity_summary(data, rsa.OUTPUT_DIR)
print("re-plotted at 300 dpi (no recompute)")
