import os
import json
import time
import requests
import aiohttp
import asyncio
from typing import Any, Dict, Optional

class ComfyUIAPI:
    def __init__(self, api_url: str, workflow_path: str, max_retries: int = 3, retry_delay: float = 2.0):
        self.api_url = api_url
        self.workflow_path = workflow_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def load_workflow(self) -> Dict[str, Any]:
        with open(self.workflow_path, 'r') as f:
            return json.load(f)

    def inject_prompts(self, workflow: Dict[str, Any], prompt: str, negative_prompt: str, sd_settings: Dict[str, Any], face_swap_path: Optional[str], enhance: bool, upscale: bool) -> Dict[str, Any]:
        # This function maintains upscaler integration by preserving the upscale flag and hires_fix setting
        # The upscale parameter controls the Upscale node, and hires_fix is maintained in sd_settings
        # This ensures that ComfyUI applies upscaling before the image is returned for face swapping
        # This function should be customized to match your workflow JSON structure.
        # Example: find nodes by type or label and set their parameters.
        for node in workflow.get('nodes', []):
            if node.get('type') == 'KSampler':
                node['inputs']['positive'] = prompt
                node['inputs']['negative'] = negative_prompt
                node['inputs']['steps'] = sd_settings.get('steps')
                node['inputs']['cfg'] = sd_settings.get('cfg')
                node['inputs']['sampler_name'] = sd_settings.get('sampler')
                node['inputs']['seed'] = sd_settings.get('seed')
                node['inputs']['width'] = sd_settings.get('width')
                node['inputs']['height'] = sd_settings.get('height')
                node['inputs']['hires_fix'] = sd_settings.get('hires_fix', False)
            if node.get('type') == 'CheckpointLoaderSimple':
                node['inputs']['ckpt_name'] = sd_settings.get('model')
            if node.get('type') == 'LoadImage' and face_swap_path:
                node['inputs']['image'] = face_swap_path
            if node.get('type') == 'FaceEnhance':
                node['inputs']['enabled'] = enhance
            if node.get('type') == 'Upscale':
                node['inputs']['enabled'] = upscale
        return workflow

    def post_workflow(self, workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, json=workflow, timeout=60)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"ComfyUI API error: {e}. Retry {attempt+1}/{self.max_retries}")
                time.sleep(self.retry_delay)
        return None

    async def post_workflow_async(self, workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.api_url, json=workflow, timeout=60) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as e:
                print(f"ComfyUI API error: {e}. Retry {attempt+1}/{self.max_retries}")
                await asyncio.sleep(self.retry_delay)
        return None 