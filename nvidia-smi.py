#!/usr/bin/env python3
# encoding: utf-8
import subprocess
import platform

if 'Win' in platform.system():
    nvidia_smi_path = 'C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe'
else:
    nvidia_smi_path = 'nvidia-smi'
subprocess.call(nvidia_smi_path, shell=True)