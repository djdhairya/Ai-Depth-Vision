import cv2
import torch 
import matplotlib.pyplot as plt
import numpy as np

# Download the MiDaS model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Input transform pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transforms = transforms.small_transform

# OpenCV video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Transform input for MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transforms(img).to('cpu')

    # Make prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()
        
        output = prediction.cpu().numpy()

    # Display the depth map and original frame using matplotlib
    plt.subplot(1, 2, 1)
    plt.imshow(output, cmap='plasma')  # Depth map
    plt.title('Depth Map')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img)  # Original frame
    plt.title('Original Frame')

    plt.draw()  # Update the figure
    plt.pause(0.00001)  # Short pause to render the next frame

    # To manually break the loop, use keyboard input
    if plt.waitforbuttonpress(timeout=0.01):
        break

cap.release()
plt.show()
