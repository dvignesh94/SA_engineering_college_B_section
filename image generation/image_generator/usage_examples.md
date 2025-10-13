# ComfyUI Image Generator - Command Line Usage Examples

The ComfyUI Image Generator now supports command-line arguments for easy usage. All generated images are automatically saved to the current directory with full metadata tracking.

## 🚀 Quick Start

### Basic Usage (minimal command):
```bash
python comfyui_image_generator.py "Beautiful house with balloons"
```

### Get Help:
```bash
python comfyui_image_generator.py --help
```

## 📐 Image Size & Quality

### Custom dimensions:
```bash
python comfyui_image_generator.py "A cat sitting on a chair" --width 768 --height 768
```

### High-resolution generation:
```bash
python comfyui_image_generator.py "Detailed architectural drawing" --width 1024 --height 1024
```

### High-quality generation with more steps:
```bash
python comfyui_image_generator.py "Sunset over mountains" --steps 30 --cfg-scale 8.0
```

### Ultra-high quality:
```bash
python comfyui_image_generator.py "Masterpiece painting" --steps 50 --cfg-scale 7.5 --width 1024 --height 1024
```

## 🎯 Prompt Control

### With specific seed for reproducibility:
```bash
python comfyui_image_generator.py "Portrait of a person" --seed 42
```

### Custom negative prompt:
```bash
python comfyui_image_generator.py "Beautiful landscape" --negative-prompt "blurry, low quality, distorted, ugly"
```

### Detailed negative prompt:
```bash
python comfyui_image_generator.py "Portrait of a woman" --negative-prompt "blurry, low quality, distorted, ugly, bad anatomy, deformed, disfigured, extra limbs, missing limbs, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad proportions, extra limbs, cloned face, disfigured, out of frame, ugly, extra limbs, bad anatomy, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck"
```

## ⚙️ Advanced Settings

### Different sampler and scheduler:
```bash
python comfyui_image_generator.py "Abstract art" --sampler dpmpp_2m --scheduler karras
```

### Fast generation with DPM++:
```bash
python comfyui_image_generator.py "Quick sketch" --sampler dpm_fast --steps 10
```

### High-quality with Euler:
```bash
python comfyui_image_generator.py "Detailed illustration" --sampler euler --steps 40 --cfg-scale 8.0
```

### Different model:
```bash
python comfyui_image_generator.py "Anime character" --model "anime_model.safetensors"
```

### Custom model with specific settings:
```bash
python comfyui_image_generator.py "Realistic portrait" --model "realistic_model.safetensors" --steps 25 --cfg-scale 7.0
```

## 🔄 Batch & Multiple Images

### Generate multiple images:
```bash
python comfyui_image_generator.py "Flower field" --batch-size 4
```

### Multiple high-quality images:
```bash
python comfyui_image_generator.py "Fantasy castle" --batch-size 3 --steps 30 --width 768 --height 768
```

## 🌐 Server Configuration

### Custom server URL:
```bash
python comfyui_image_generator.py "Space scene" --server-url "http://192.168.1.100:8188"
```

### Remote server with longer timeout:
```bash
python comfyui_image_generator.py "Complex scene" --server-url "http://remote-server:8188" --timeout 600
```

## 📊 Output & Debugging

### Verbose output (see all settings):
```bash
python comfyui_image_generator.py "Detailed prompt" --verbose
```

### Skip saving to local directory:
```bash
python comfyui_image_generator.py "Test image" --no-save
```

### Different output format:
```bash
python comfyui_image_generator.py "Web image" --format jpeg
```

## 🎨 Real-World Use Cases

### Portrait Photography:
```bash
python comfyui_image_generator.py "Professional headshot of a businesswoman, studio lighting, high quality" \
  --negative-prompt "blurry, low quality, distorted, ugly, bad anatomy, deformed" \
  --width 768 --height 1024 --steps 30 --cfg-scale 7.5 --seed 42
```

### Landscape Art:
```bash
python comfyui_image_generator.py "Serene mountain lake at golden hour, misty atmosphere, photorealistic" \
  --negative-prompt "blurry, low quality, distorted, ugly, oversaturated" \
  --width 1024 --height 768 --steps 40 --cfg-scale 8.0
```

### Anime/Illustration:
```bash
python comfyui_image_generator.py "Anime girl with long hair, detailed eyes, vibrant colors, high quality" \
  --negative-prompt "blurry, low quality, distorted, ugly, bad anatomy, deformed" \
  --width 512 --height 768 --steps 25 --cfg-scale 7.0 --model "anime_model.safetensors"
```

### Abstract Art:
```bash
python comfyui_image_generator.py "Abstract geometric patterns, vibrant colors, modern art style" \
  --negative-prompt "blurry, low quality, distorted, ugly" \
  --width 1024 --height 1024 --steps 35 --cfg-scale 8.5 --sampler dpmpp_2m
```

### Product Photography:
```bash
python comfyui_image_generator.py "Modern smartphone on white background, product photography, clean lighting" \
  --negative-prompt "blurry, low quality, distorted, ugly, bad lighting, shadows" \
  --width 1024 --height 1024 --steps 30 --cfg-scale 7.0 --format jpeg
```

## 🔧 Complete Example with All Options:
```bash
python comfyui_image_generator.py "A majestic dragon flying over a medieval castle at sunset" \
  --negative-prompt "blurry, low quality, distorted, ugly, bad anatomy" \
  --width 1024 \
  --height 1024 \
  --steps 50 \
  --cfg-scale 7.5 \
  --seed 12345 \
  --sampler dpmpp_2m \
  --scheduler karras \
  --model "v1-5-pruned-emaonly-fp16.safetensors" \
  --format png \
  --batch-size 1 \
  --timeout 600 \
  --verbose
```

## 📋 All Available Arguments

### Required:
- `prompt` - The text prompt for image generation

### Image Parameters:
- `--negative-prompt` - Negative prompt (default: "blurry, low quality, distorted")
- `--width` - Image width in pixels (default: 512)
- `--height` - Image height in pixels (default: 512)
- `--format` - Output image format: png, jpeg, webp (default: png)

### Generation Settings:
- `--steps` - Number of denoising steps (default: 20)
- `--cfg-scale` - CFG scale for prompt adherence (default: 7.0)
- `--seed` - Random seed for reproducible generation (default: random)
- `--sampler` - Sampling method (default: euler)
- `--scheduler` - Scheduler type (default: normal)
- `--model` - Model checkpoint name (default: v1-5-pruned-emaonly-fp16.safetensors)
- `--batch-size` - Number of images to generate (default: 1)

### Server Configuration:
- `--server-url` - ComfyUI server URL (default: http://localhost:8188)
- `--timeout` - Request timeout in seconds (default: 300)

### Output Control:
- `--no-save` - Skip saving images to local directory
- `--verbose` - Enable verbose output

## 🆘 Getting Help
To see all available options and their descriptions:
```bash
python comfyui_image_generator.py --help
```

## 📁 Output & File Storage
All generated images are automatically saved to the current directory with the following naming pattern:
`comfyui_generated_YYYYMMDD_HHMMSS_[generation_id]_[index].png`

Each image includes a corresponding metadata JSON file:
`comfyui_generated_YYYYMMDD_HHMMSS_[generation_id]_[index]_metadata.json`

The metadata file contains:
- Generation parameters (prompt, negative prompt, dimensions, steps, etc.)
- Model and sampler information
- Timestamps and generation status
- Unique generation ID for tracking

## 💡 Tips & Best Practices

### For High Quality:
- Use 30-50 steps for detailed images
- CFG scale 7.0-8.5 for good prompt adherence
- Higher resolution (768x768 or 1024x1024) for detail
- Use `dpmpp_2m` or `euler` samplers

### For Speed:
- Use 10-20 steps for quick generation
- Lower resolution (512x512) for faster processing
- Use `dpm_fast` sampler
- Reduce CFG scale to 6.0-7.0

### For Reproducibility:
- Always use `--seed` with a specific number
- Keep the same model and settings
- Use `--verbose` to see exact parameters used

### For Batch Processing:
- Use `--batch-size` for multiple variations
- Consider using different seeds for variety
- Monitor disk space for large batches

