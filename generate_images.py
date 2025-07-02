import os
import sys
import yaml
import typer
from dotenv import load_dotenv
from rich.progress import Progress
from comfyui_api import ComfyUIAPI
from PIL import Image

app = typer.Typer()

def load_config(config_path: str = 'config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_prompts(prompts_path: str):
    with open(prompts_path, 'r') as f:
        return yaml.safe_load(f)

def save_image(image_data, output_path):
    with open(output_path, 'wb') as f:
        f.write(image_data)

@app.command()
def main(prompts: str = typer.Option('promfy_prompts.yaml', help="Prompt YAML file")):
    load_dotenv()
    config = load_config()
    comfyui_api_url = os.getenv('COMFYUI_API_URL', config['comfyui_api_url'])
    workflow_path = config['workflow_path']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    api = ComfyUIAPI(comfyui_api_url, workflow_path)
    prompts_data = load_prompts(prompts)
    images = prompts_data.get('images', [])
    if not images:
        print("No images found in prompt file.")
        sys.exit(1)
    with Progress() as progress:
        task = progress.add_task("Generating images...", total=len(images))
        for img in images:
            name = img.get('name', 'output')
            prompt = img.get('prompt', '')
            negative_prompt = img.get('negative_prompt', '')
            sd_settings = config['sd_defaults'].copy()
            sd_settings.update(img.get('sd_settings', {}))
            face_swap_path = img.get('face_swap', {}).get('source')
            enhance = img.get('enhance', False)
            upscale = img.get('upscale', False)
            workflow = api.load_workflow()
            workflow = api.inject_prompts(workflow, prompt, negative_prompt, sd_settings, face_swap_path, enhance, upscale)
            result = api.post_workflow(workflow)
            if not result:
                print(f"Failed to generate image: {name}")
                progress.advance(task)
                continue
            # Save output image (assume result contains image bytes or path)
            # This part may need to be adapted to your workflow's output format
            output_path = os.path.join(output_dir, f"{name}.png")
            if 'image' in result:
                save_image(result['image'], output_path)
            elif 'output_path' in result:
                # Copy file from output_path
                from shutil import copyfile
                copyfile(result['output_path'], output_path)
            else:
                print(f"No image data found for {name}")
            progress.advance(task)
    print(f"\nAll images processed. Check the '{output_dir}' directory.")

if __name__ == "__main__":
    app() 