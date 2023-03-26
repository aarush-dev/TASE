import cv2
import torch
import time
import numpy as np
from playsound import playsound

# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") 
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


cap = cv2.VideoCapture(1)

while cap.isOpened():

    success, img = cap.read()

    start = time.time()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)


    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

    dim = (192*3, 108*4)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('Image', img)
    cv2.imshow('Depth Map', depth_map)

    hsvFrame = cv2.cvtColor(depth_map, cv2.COLOR_BGR2HSV)

    red_lower = np.array([0, 100, 60], np.uint8)
    red_upper = np.array([35, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    
    kernal = np.ones((5, 5), "uint8")
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(depth_map, depth_map,
							mask = red_mask)
                            
    contours, hierarchy = cv2.findContours(red_mask,
										cv2.RETR_TREE,
										cv2.CHAIN_APPROX_SIMPLE)
                                        
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 18500):
            x, y, w, h = cv2.boundingRect(contour)
            depth_map = cv2.rectangle(depth_map, (x, y),
									(x + w, y + h),
									(0, 0, 255), 2)
            cv2.putText(depth_map, str(x), (x, y),
						cv2.FONT_HERSHEY_SIMPLEX, 1.0,
						(0, 0, 255))
            if x < 140.0 and w<=500:
                playsound(r"C:\Users\Aarus\PycharmProjects\TASE-2.1\Left.wav")
            elif x >= 140.0 and x<=280:
                playsound(r"C:\Users\Aarus\PycharmProjects\TASE-2.1\Front.wav")
            elif x > 280.0:	
                playsound(r"C:\Users\Aarus\PycharmProjects\TASE-2.1\Right.wav")
            elif w>500:
                playsound(r"C:\Users\Aarus\PycharmProjects\TASE-2.1\Front.wav")     
		
                        
    cv2.imshow("Obstacle Map", depth_map)

    if cv2.waitKey(5) and 0xFF == 27:
        break


cap.release()