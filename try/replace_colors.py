import os
import numpy as np
from PIL import Image

def replace_colors_tolerance(image_path, color_mappings, tolerance):
    """
    Replace colors in an image with tolerance for similar colors and overwrite the original image.

    Args:
        image_path (str): Path to input image
        color_mappings (dict): Dictionary mapping source RGB tuples to target RGB tuples
        tolerance (int): Color tolerance (0-255)
    """
    # Load image
    img = Image.open(image_path)

    # Handle different image modes
    if img.mode == 'RGBA':
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[-1])
        img_array = np.array(rgb_img)
    elif img.mode == 'P':
        img = img.convert('RGB')
        img_array = np.array(img)
    else:
        img_array = np.array(img)

    modified_array = img_array.copy()

    for source_color, target_color in color_mappings.items():
        if len(source_color) != 3 or len(target_color) != 3:
            print(f"Warning: Skipping color mapping {source_color} -> {target_color}")
            continue

        color_diff = np.sqrt(np.sum((img_array - source_color) ** 2, axis=-1))
        mask = color_diff <= tolerance
        modified_array[mask] = target_color

    # Overwrite original image
    result_img = Image.fromarray(modified_array)
    result_img.save(image_path)
    print(f"Overwritten image: {image_path}")

if __name__ == "__main__":
    folder_path = "C:/Users/Sharon/Desktop/Cambridge/Mphil/writeup/renewed_panels"
    """
    (91, 94, 215): (252, 95, 64),
    (236, 192, 68): (252, 95, 64),
    (76, 168, 73): (252, 95, 64),
    (222, 89, 76): (252, 95, 64),
    (152, 92, 150): (252, 95, 64),

    (81, 93, 186): (252, 95, 64),
    (235, 177, 48): (252, 95, 64),
    (82, 79, 200): (252, 95, 64),
    (87, 63, 169): (252, 95, 64),
    (235, 177, 48): (252, 95, 64),
    (75, 78, 199): (252, 95, 64),
    (59, 151, 56): (252, 95, 64),
    (206, 73, 61): (252, 95, 64),

    (129, 175, 51): (252, 95, 64),
    (135, 75, 133): (252, 95, 64),
    (223, 166, 36): (252, 95, 64),
    (172, 153, 94): (252, 95, 64),
    (70, 84, 176): (252, 95, 64),
    (115, 137, 135): (252, 95, 64),
    (71, 67, 189): (252, 95, 64),
    
        (166, 147, 89): (252, 95, 64),
        (72, 66, 195): (252, 95, 64),
        (225, 165, 42): (252, 95, 64),
        (219, 175, 51): (252, 95, 64),
        
        
        (73, 154, 47): (252, 95, 64),
        (218, 179, 62): (252, 95, 64),
        (127, 71, 138): (252, 95, 64),
        (69, 156, 56): (252, 95, 64),
        (142, 79, 134): (252, 95, 64),
        (208, 172, 58): (252, 95, 64),
        

        
        (218, 185, 78): (252, 95, 64),
        (67, 88, 230): (252, 95, 64),
    """

    """
            (74, 144, 193): (252, 95, 64),
        (244, 142, 52): (252, 95, 64),
        (84, 177, 84): (252, 95, 64),

        (252, 95, 64): (252, 95, 64),
        (251, 169, 96): (252, 95, 64),
        (116, 190, 116): (252, 95, 64),
        
                (224, 111, 34): (252, 95, 64),
        (45, 104, 166): (252, 95, 64),
        
        """
    color_map_tolerance = {

        (242,93,47): (252, 95, 64),


    }
    tolerance = 15

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            full_path = os.path.join(folder_path, filename)
            try:
                replace_colors_tolerance(full_path, color_map_tolerance, tolerance)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
