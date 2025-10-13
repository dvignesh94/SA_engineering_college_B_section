
import json
import requests
import base64
import io
from PIL import Image
from typing import Dict, List, Optional, Any, Union
import os
import time
import uuid
from dataclasses import dataclass
from enum import Enum
import sys
import argparse

# AssetManager dependency removed - files will be saved in current directory

class ImageFormat(Enum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"

class GenerationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted"
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    seed: Optional[int] = None
    sampler: str = "euler"
    scheduler: str = "normal"
    model: str = "sd_xl_base_1.0.safetensors"
    format: ImageFormat = ImageFormat.PNG
    batch_size: int = 1

@dataclass
class GenerationResult:
    id: str
    status: GenerationStatus
    images: List[Image.Image]
    metadata: Dict[str, Any]
    error: Optional[str] = None
    created_at: float = 0.0
    completed_at: Optional[float] = None

class ComfyUIImageGenerator:
    """
    Advanced image generator for ComfyUI with comprehensive features.
    """
    
    def __init__(self, 
                 server_url: str = "http://localhost:8188",
                 client_id: str = None,
                 timeout: int = 300):
        """
        Initialize the ComfyUI Image Generator.
        
        Args:
            server_url: ComfyUI server URL
            client_id: Client ID for tracking requests
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.client_id = client_id or str(uuid.uuid4())
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ComfyUI-ImageGenerator/1.0.0'
        })
        
        # Files will be saved in current directory instead of centralized asset management
        
    def _load_workflow_template(self, template_name: str) -> Dict[str, Any]:
        """Load a workflow template from the templates directory."""
        template_path = os.path.join(
            os.path.dirname(__file__), 
            'templates', 
            f'{template_name}.json'
        )
        
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Workflow template '{template_name}' not found")
            
        with open(template_path, 'r') as f:
            return json.load(f)
    
    def _create_workflow(self, request: GenerationRequest) -> Dict[str, Any]:
        """Create a workflow from a generation request."""
        # Load base workflow template
        template = self._load_workflow_template('simple_txt2img')
        workflow = template['workflow'].copy()
        
        # Update workflow parameters
        workflow['2']['inputs']['text'] = request.prompt
        workflow['3']['inputs']['text'] = request.negative_prompt
        workflow['4']['inputs']['width'] = request.width
        workflow['4']['inputs']['height'] = request.height
        workflow['5']['inputs']['steps'] = request.steps
        workflow['5']['inputs']['cfg'] = request.cfg_scale
        workflow['5']['inputs']['sampler_name'] = request.sampler
        workflow['5']['inputs']['scheduler'] = request.scheduler
        
        # Handle seed parameter
        if request.seed is not None and request.seed != -1:
            workflow['5']['inputs']['seed'] = request.seed
        else:
            workflow['5']['inputs']['seed'] = 0  # Use 0 as default instead of -1
        
        # Update model if specified
        if hasattr(request, 'model') and request.model:
            workflow['1']['inputs']['ckpt_name'] = request.model
        else:
            # Use default model
            workflow['1']['inputs']['ckpt_name'] = "sd_xl_base_1.0.safetensors"
            
        return workflow
    
    def _queue_workflow(self, workflow: Dict[str, Any]) -> str:
        """Queue a workflow for execution."""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        
        response = self.session.post(
            f"{self.server_url}/prompt",
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to queue workflow: {response.text}")
            
        result = response.json()
        return result['prompt_id']
    
    def _get_queue_status(self) -> Dict[str, Any]:
        """Get the current queue status."""
        response = self.session.get(f"{self.server_url}/queue")
        
        if response.status_code != 200:
            raise Exception(f"Failed to get queue status: {response.text}")
            
        return response.json()
    
    def _get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get the execution history for a prompt."""
        response = self.session.get(f"{self.server_url}/history/{prompt_id}")
        
        if response.status_code != 200:
            raise Exception(f"Failed to get history: {response.text}")
            
        return response.json()
    
    def _download_image(self, filename: str) -> Image.Image:
        """Download an image from the ComfyUI server."""
        response = self.session.get(f"{self.server_url}/view?filename={filename}")
        
        if response.status_code != 200:
            raise Exception(f"Failed to download image: {response.text}")
            
        return Image.open(io.BytesIO(response.content))
    
    def generate_image(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate an image based on the provided request.
        
        Args:
            request: Generation request parameters
            
        Returns:
            GenerationResult with the generated images
        """
        result_id = str(uuid.uuid4())
        created_at = time.time()
        
        try:
            # Create workflow from request
            workflow = self._create_workflow(request)
            
            # Queue the workflow
            prompt_id = self._queue_workflow(workflow)
            
            # Wait for completion
            images = []
            status = GenerationStatus.RUNNING
            error = None
            
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                history = self._get_history(prompt_id)
                
                if prompt_id in history:
                    prompt_data = history[prompt_id]
                    
                    if 'status' in prompt_data:
                        if prompt_data['status'].get('status_str') == 'success':
                            status = GenerationStatus.COMPLETED
                            
                            # Download generated images
                            outputs = prompt_data.get('outputs', {})
                            for node_id, node_output in outputs.items():
                                if 'images' in node_output:
                                    for image_info in node_output['images']:
                                        filename = image_info['filename']
                                        image = self._download_image(filename)
                                        images.append(image)
                            break
                            
                        elif prompt_data['status'].get('status_str') == 'error':
                            status = GenerationStatus.FAILED
                            error = prompt_data['status'].get('error', 'Unknown error')
                            break
                
                time.sleep(1)
            
            if status == GenerationStatus.RUNNING:
                status = GenerationStatus.FAILED
                error = "Generation timeout"
            
            return GenerationResult(
                id=result_id,
                status=status,
                images=images,
                metadata={
                    'prompt_id': prompt_id,
                    'request': request.__dict__,
                    'workflow': workflow
                },
                error=error,
                created_at=created_at,
                completed_at=time.time() if status == GenerationStatus.COMPLETED else None
            )
            
        except Exception as e:
            return GenerationResult(
                id=result_id,
                status=GenerationStatus.FAILED,
                images=[],
                metadata={'request': request.__dict__},
                error=str(e),
                created_at=created_at
            )
    
    def generate_batch(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        """
        Generate multiple images in batch.
        
        Args:
            requests: List of generation requests
            
        Returns:
            List of generation results
        """
        results = []
        
        for request in requests:
            result = self.generate_image(request)
            results.append(result)
            
        return results
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get information about the ComfyUI server."""
        try:
            response = self.session.get(f"{self.server_url}/system_stats")
            if response.status_code == 200:
                return response.json()
        except:
            pass
            
        return {'status': 'unknown', 'server_url': self.server_url}
    
    def test_connection(self) -> bool:
        """Test connection to the ComfyUI server."""
        try:
            response = self.session.get(f"{self.server_url}/system_stats", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def save_images(self, result: GenerationResult, output_dir: str = None) -> List[str]:
        """
        Save generated images to the local directory.
        
        Args:
            result: GenerationResult containing the images to save
            output_dir: Directory to save images (default: current directory)
            
        Returns:
            List of saved file paths
        """
        return save_result_images(result, output_dir)

# Example usage and utility functions
def create_simple_request(prompt: str, 
                         width: int = 512, 
                         height: int = 512,
                         steps: int = 20) -> GenerationRequest:
    """Create a simple generation request with default parameters."""
    return GenerationRequest(
        prompt=prompt,
        width=width,
        height=height,
        steps=steps
    )

def save_result_images(result: GenerationResult, output_dir: str = None):
    """
    Save generated images from a result to the local directory.
    
    Args:
        result: GenerationResult containing the images to save
        output_dir: Directory to save images (default: current directory)
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    
    for i, image in enumerate(result.images):
        # Create filename with timestamp and index
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"comfyui_generated_{timestamp}_{result.id}_{i}.png"
        
        # Create full filepath
        filepath = os.path.join(output_dir, filename)
        
        # Save the image
        image.save(filepath)
        
        # Create metadata file alongside the image
        metadata_filename = f"comfyui_generated_{timestamp}_{result.id}_{i}_metadata.json"
        metadata_filepath = os.path.join(output_dir, metadata_filename)
        
        metadata = {
            'generation_id': result.id,
            'image_index': i,
            'prompt': result.metadata.get('request', {}).get('prompt', ''),
            'negative_prompt': result.metadata.get('request', {}).get('negative_prompt', ''),
            'width': result.metadata.get('request', {}).get('width', 512),
            'height': result.metadata.get('request', {}).get('height', 512),
            'steps': result.metadata.get('request', {}).get('steps', 20),
            'cfg_scale': result.metadata.get('request', {}).get('cfg_scale', 7.0),
            'seed': result.metadata.get('request', {}).get('seed', None),
            'sampler': result.metadata.get('request', {}).get('sampler', 'euler'),
            'model': result.metadata.get('request', {}).get('model', ''),
            'created_at': result.created_at,
            'completed_at': result.completed_at,
            'status': result.status.value
        }
        
        # Save metadata as JSON file
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        saved_paths.append(filepath)
        print(f"✅ Saved image: {filepath}")
        print(f"✅ Saved metadata: {metadata_filepath}")
    
    return saved_paths

def parse_arguments():
    """Parse command-line arguments for the ComfyUI image generator."""
    parser = argparse.ArgumentParser(
        description="ComfyUI Image Generator with Local File Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comfyui_image_generator.py "Beautiful house with balloons"
  python comfyui_image_generator.py "A cat sitting on a chair" --width 768 --height 768
  python comfyui_image_generator.py "Sunset over mountains" --steps 30 --cfg-scale 8.0
  python comfyui_image_generator.py "Portrait of a person" --negative-prompt "blurry, low quality" --seed 42
        """
    )
    
    # Required positional argument for prompt
    parser.add_argument(
        'prompt',
        type=str,
        help='The text prompt for image generation'
    )
    
    # Optional arguments for generation parameters
    parser.add_argument(
        '--negative-prompt',
        type=str,
        default="blurry, low quality, distorted",
        help='Negative prompt to avoid certain elements (default: "blurry, low quality, distorted")'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=512,
        help='Image width in pixels (default: 512)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=512,
        help='Image height in pixels (default: 512)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=20,
        help='Number of denoising steps (default: 20)'
    )
    
    parser.add_argument(
        '--cfg-scale',
        type=float,
        default=7.0,
        help='CFG scale for prompt adherence (default: 7.0)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible generation (default: random)'
    )
    
    parser.add_argument(
        '--sampler',
        type=str,
        default="euler",
        choices=["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2m", "dpmpp_2m_ancestral", "dpmpp_sde", "dpmpp_sde_ancestral", "dpmpp_2m_sde", "dpmpp_2m_sde_ancestral", "ddim", "uni_pc", "uni_pc_bh2"],
        help='Sampling method (default: euler)'
    )
    
    parser.add_argument(
        '--scheduler',
        type=str,
        default="normal",
        choices=["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"],
        help='Scheduler type (default: normal)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default="sd_xl_base_1.0.safetensors",
        help='Model checkpoint name (default: sd_xl_base_1.0.safetensors)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=["png", "jpeg", "webp"],
        default="png",
        help='Output image format (default: png)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Number of images to generate (default: 1)'
    )
    
    parser.add_argument(
        '--server-url',
        type=str,
        default="http://localhost:8188",
        help='ComfyUI server URL (default: http://localhost:8188)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Request timeout in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving images to asset management system'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

def create_request_from_args(args):
    """Create a GenerationRequest from parsed arguments."""
    return GenerationRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        sampler=args.sampler,
        scheduler=args.scheduler,
        model=args.model,
        format=ImageFormat(args.format),
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize generator with custom server URL and timeout
    generator = ComfyUIImageGenerator(
        server_url=args.server_url,
        timeout=args.timeout
    )
    
    if args.verbose:
        print(f"🔧 Configuration:")
        print(f"   Server URL: {args.server_url}")
        print(f"   Timeout: {args.timeout}s")
        print(f"   Prompt: {args.prompt}")
        print(f"   Dimensions: {args.width}x{args.height}")
        print(f"   Steps: {args.steps}")
        print(f"   CFG Scale: {args.cfg_scale}")
        print(f"   Sampler: {args.sampler}")
        print(f"   Model: {args.model}")
        if args.seed is not None:
            print(f"   Seed: {args.seed}")
        print()
    
    # Test connection
    if generator.test_connection():
        print("✅ Connected to ComfyUI server")
        
        # Create request from command-line arguments
        request = create_request_from_args(args)
        
        if args.verbose:
            print(f"🎨 Generating image with prompt: '{args.prompt}'")
        else:
            print("🎨 Generating image...")
        
        # Generate image
        result = generator.generate_image(request)
        
        if result.status == GenerationStatus.COMPLETED:
            print(f"✅ Generated {len(result.images)} image(s)")
            
            if not args.no_save:
                # Save to local directory
                saved_paths = generator.save_images(result)
                print(f"📁 Images saved to local directory:")
                for path in saved_paths:
                    print(f"   - {path}")
            else:
                print("⚠️ Skipped saving images (--no-save flag used)")
                
        else:
            print(f"❌ Generation failed: {result.error}")
            sys.exit(1)
    else:
        print("❌ Could not connect to ComfyUI server")
        print(f"   Make sure ComfyUI is running at: {args.server_url}")
        sys.exit(1)
