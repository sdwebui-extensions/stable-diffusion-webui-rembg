import launch
import os
import shutil
try:
    from modules.paths_internal import models_path, shared_models_path
except:
    from modules.paths_internal import models_path 
    shared_models_path = None
from logger import logger

if not launch.is_installed("rembg"):
    launch.run_pip("install rembg==2.0.38 --no-deps", "rembg")

for dep in ['onnxruntime', 'pymatting', 'pooch']:
    if not launch.is_installed(dep):
        launch.run_pip(f"install {dep}", f"{dep} for REMBG extension")

save_dir = '/root/.u2net'
os.makedirs(save_dir, exist_ok=True)
models_rembg = os.path.join(models_path, 'rembg')
shared_models_rembg = os.path.join(shared_models_path, 'rembg') if shared_models_path else None
if os.path.exists(models_rembg) and os.path.isdir(models_rembg):
    for onnx_file in os.listdir(models_rembg):
        if onnx_file.endswith('.onnx'):
            onnx_path = os.path.join(models_rembg, onnx_file)
            shutil.copyfile(onnx_path, os.path.join(save_dir, onnx_file))
            logger.info(f'copy file from {onnx_path} to {save_dir}')
if shared_models_rembg and os.path.isdir(shared_models_rembg):
    for onnx_file in os.listdir(shared_models_rembg):
        if onnx_file.endswith('.onnx'):
            onnx_path = os.path.join(models_rembg, onnx_file)
            save_path = os.path.join(save_dir, onnx_file)
            if not os.path.exists(save_path):
                shutil.copyfile(onnx_path, save_path)
                logger.info(f'copy file from {onnx_path} to {save_dir}')
