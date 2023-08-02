from modules import scripts_postprocessing
import gradio as gr

from modules.ui_components import FormRow
import rembg

from modules.shared import cmd_opts, encode_image_to_base64
from modules.api.api import decode_base64_to_image
import requests
import json

import os
import shutil
try:
    from modules.paths_internal import models_path, shared_models_path
except:
    from modules.paths_internal import models_path 
    shared_models_path = None
from logger import logger

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


models = [
    "None",
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
]

class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Rembg"
    order = 20000
    model = None

    def ui(self):
        with FormRow():
            model = gr.Dropdown(label="Remove background", choices=models, value="None")
            return_mask = gr.Checkbox(label="Return mask", value=False)
            alpha_matting = gr.Checkbox(label="Alpha matting", value=False)

        with FormRow(visible=False) as alpha_mask_row:
            alpha_matting_erode_size = gr.Slider(label="Erode size", minimum=0, maximum=40, step=1, value=10)
            alpha_matting_foreground_threshold = gr.Slider(label="Foreground threshold", minimum=0, maximum=255, step=1, value=240)
            alpha_matting_background_threshold = gr.Slider(label="Background threshold", minimum=0, maximum=255, step=1, value=10)

        alpha_matting.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[alpha_matting],
            outputs=[alpha_mask_row],
        )

        return {
            "model": model,
            "return_mask": return_mask,
            "alpha_matting": alpha_matting,
            "alpha_matting_foreground_threshold": alpha_matting_foreground_threshold,
            "alpha_matting_background_threshold": alpha_matting_background_threshold,
            "alpha_matting_erode_size": alpha_matting_erode_size,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, model, return_mask, alpha_matting, alpha_matting_foreground_threshold, alpha_matting_background_threshold, alpha_matting_erode_size):
        if not model or model == "None":
            return
        
        if cmd_opts.just_ui:
            server_url = '/'.join([cmd_opts.server_path, 'rembg'])
            req_dict = dict(
                input_image = encode_image_to_base64(pp.image),
                model = model, 
                return_mask = return_mask, 
                alpha_matting = alpha_matting, 
                alpha_matting_foreground_threshold = alpha_matting_foreground_threshold, 
                alpha_matting_background_threshold = alpha_matting_background_threshold, 
                alpha_matting_erode_size = alpha_matting_erode_size
            )
            result = requests.post(server_url, json=req_dict)
            if result.status_code==200:
                pp.image = decode_base64_to_image(json.loads(result.text)['image'])
            else:
                raise Exception(f'failed to request server {server_url} with excepth {result.text}')
        else:
            pp.image = rembg.remove(
                pp.image,
                session=rembg.new_session(model),
                only_mask=return_mask,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
            )

        pp.info["Rembg"] = model
