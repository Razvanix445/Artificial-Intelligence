import os
from PIL import Image

def remove_corrupted_images(directory):
    removed_count = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            try:
                with Image.open(os.path.join(directory, filename)) as img:
                    img.verify()
            except (IOError, SyntaxError):
                os.remove(os.path.join(directory, filename))
                print(f'Removed corrupted image: {filename}')
                removed_count += 1
    print(f'Total corrupted images removed: {removed_count}')

remove_corrupted_images('images')
remove_corrupted_images('sepia_images')