@echo off
rem Detached launcher (the agent tool caps commands at 10 minutes).
rem Variant A: expanded 8-network LOLO without leak_area (5 seeds, ~30 min each).
rem Variant B: same, additionally without the two raw network-size count features.
cd /d "C:\Users\Zachy\OneDrive\Desktop\LLM water"
python dev\active\scripts\run_leak_area_ablation.py --variant no_leak_area > dev\active\artifacts\leak_area_ablation.log 2>&1
echo variantA exit %ERRORLEVEL% >> dev\active\artifacts\leak_area_ablation.log
python dev\active\scripts\run_leak_area_ablation.py --variant no_leak_area_no_counts > dev\active\artifacts\leak_area_ablation_nocounts.log 2>&1
echo variantB exit %ERRORLEVEL% >> dev\active\artifacts\leak_area_ablation_nocounts.log
echo done > dev\active\artifacts\leak_area_ablation.done
