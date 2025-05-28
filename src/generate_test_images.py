import numpy as np
import cv2
import os

def generate_test_image(sample_id, is_healthy=True):
    # Create a 224x224 RGB image
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Generate different colors based on health status
    if is_healthy:
        # Healthy coral: vibrant colors
        img[:, :, 0] = np.random.randint(100, 255, (224, 224))  # Blue
        img[:, :, 1] = np.random.randint(100, 255, (224, 224))  # Green
        img[:, :, 2] = np.random.randint(100, 255, (224, 224))  # Red
    else:
        # Bleached coral: white/pale colors
        img[:, :, 0] = np.random.randint(200, 255, (224, 224))  # Blue
        img[:, :, 1] = np.random.randint(200, 255, (224, 224))  # Green
        img[:, :, 2] = np.random.randint(200, 255, (224, 224))  # Red
    
    # Add some texture
    noise = np.random.normal(0, 25, (224, 224, 3)).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Save the image
    output_dir = "data/raw/imagery"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{sample_id}.jpg"), img)

# Generate images for all samples
samples = {
    "sample_001": True,  # healthy
    "sample_002": False,  # bleached
    "sample_003": True,  # healthy
    "sample_004": False,  # bleached
    "sample_005": True,  # healthy
}

for sample_id, is_healthy in samples.items():
    generate_test_image(sample_id, is_healthy) 