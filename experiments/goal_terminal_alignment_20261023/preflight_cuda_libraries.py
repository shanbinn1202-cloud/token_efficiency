"""Fail before model startup if a required CUDA runtime cannot load."""
import ctypes,json,os
for lib in ['libcudart.so.12','libcudart.so.13']:
 ctypes.CDLL(lib,mode=ctypes.RTLD_GLOBAL)
 print(json.dumps({'library':lib,'load_passed':True}),flush=True)
