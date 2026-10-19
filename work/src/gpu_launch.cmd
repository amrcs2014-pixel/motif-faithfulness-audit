@echo off
cd /d "E:\network research\new1\work\src"
set PYTHONPATH=E:\network research\new1\work\pylib
"C:\Users\amr\AppData\Local\Programs\Python\Python311\python.exe" -u -W ignore gpu_queue.py >> "E:\network research\new1\work\logs\gpu_queue_stdout.log" 2>&1
