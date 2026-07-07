@echo off
cd /d "C:\Users\black\OneDrive\Desktop\WideBind"
echo [WideBind] Loading environment...
call conda activate base 2>nul || echo [WideBind] No conda, using system Python

echo [WideBind] Starting training...
echo   Data: C:\Users\black\OneDrive\Desktop\fcp
echo   Model: 41M params, 24 layers, B=2, L=128
echo   VRAM: ~1.9 GB peak
echo.

start /b /wait "" python train.py ^
    --data-dir "C:\Users\black\OneDrive\Desktop\fcp" ^
    --save-dir checkpoints ^
    --batch-size 2 ^
    --seq-len 128 ^
    --n-layers 24 ^
    --bottleneck 896 ^
    --bind-K 16 ^
    --lr 3e-4 ^
    --max-steps 500000 ^
    --warmup 1000 ^
    --log-interval 100 ^
    --eval-interval 1000 ^
    --save-interval 5000 ^
    --resume auto

echo.
echo [WideBind] Training finished.
pause
