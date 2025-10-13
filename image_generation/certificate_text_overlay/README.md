# Add Vignesh to Certificate

Simple script to add "Vignesh" to certificate with predefined settings.

## Features

- Adds "Vignesh" to certificate with exact positioning
- Predefined settings: x=200, y=620, font_size=50, color=#000000
- High-quality output (JPEG with 95% quality)
- Simple one-click execution

## Files

- `add_vignesh_to_certificate.py` - Main script with predefined settings
- `README.md` - This documentation file

## Quick Start

```bash
cd "/Users/vignesh/Documents/GitHub/Generative AI/ComfyUI/certificate_text_overlay"
python add_vignesh_to_certificate.py
```

This will add "Vignesh" to the certificate at `/Users/vignesh/Documents/GitHub/Generative AI/Datasets/Certificate.jpeg` and save it as `Certificate_with_name.jpeg`.

## Predefined Settings

- **Text**: "Vignesh"
- **Position**: x=200, y=620
- **Font Size**: 50
- **Color**: #000000 (black)

## Output

The modified certificate will be saved as:
`/Users/vignesh/Documents/GitHub/Generative AI/Datasets/Certificate_with_name.jpeg`

## Requirements

- Python 3.7+
- PIL (Pillow)

## Notes

- Uses PIL for reliable text overlay
- Font fallback system ensures compatibility across different systems
- High-quality JPEG output preserves certificate quality
