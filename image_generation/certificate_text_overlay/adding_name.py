"""
Add Vignesh to Certificate
Simple script to add "Vignesh" to certificate with predefined settings.
Saves output to the current working directory (no external asset manager).
"""

from PIL import Image, ImageDraw, ImageFont
import os

def add_vignesh_to_certificate():
    """Add 'Vignesh' to the certificate with predefined settings."""
    
    # Predefined settings
    name = "Vignesh Dhanasekaran"
    x = 200
    y = 620
    font_size = 50
    text_color = "#000000"
    
    # Use provided absolute input image path and save output in current directory
    certificate_path = \
        "/Users/vignesh/Documents/GitHub/SA_engineering_college_B_section/image_generation/Jeevisoft.png"
    output_path = os.path.join(os.getcwd(), "Jeevisoft_with_name.png")
    
    print(f"   Adding '{name}' to certificate...")
    print(f"   Position: ({x}, {y})")
    print(f"   Font size: {font_size}")
    print(f"   Color: {text_color}")
    
    # Load the certificate image
    image = Image.open(certificate_path)
    
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Convert hex color to RGB
    if text_color.startswith('#'):
        text_color = text_color[1:]
    rgb_color = tuple(int(text_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Add text to the image
    draw.text((x, y), name, fill=rgb_color, font=font)
    
    # Save the image (PNG to preserve transparency/format)
    image.save(output_path, "PNG")
    
    print(f"✅ Successfully added '{name}' to certificate!")
    print(f"📁 Output saved to: {output_path}")
    
    # Saved locally in current directory; no asset registry used

if __name__ == "__main__":
    add_vignesh_to_certificate()
