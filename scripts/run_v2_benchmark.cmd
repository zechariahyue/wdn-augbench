@echo off
rem Detached launcher for the v2 benchmark (S36 redesign). Resumable per seed.
cd /d "C:\Users\Zachy\OneDrive\Desktop\LLM water"
python -u dev\active\scripts\run_v2_benchmark.py --n-baseline 100 > dev\active\artifacts\v2_benchmark.log 2>&1
echo benchmark exit %ERRORLEVEL% >> dev\active\artifacts\v2_benchmark.log
python -u dev\active\scripts\analyze_v2.py --tag v2_benchmark --bootstrap 200 > dev\active\artifacts\v2_analysis.log 2>&1
echo analysis exit %ERRORLEVEL% >> dev\active\artifacts\v2_analysis.log
echo done > dev\active\artifacts\v2_benchmark.done
