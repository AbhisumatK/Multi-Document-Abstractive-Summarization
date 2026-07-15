@echo off
call conda activate GPU-pytorch
python evaluate_model.py
pause
