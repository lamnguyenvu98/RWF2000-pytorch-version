import cv2
import numpy as np
from src.utils import getOpticalFlow, uniform_sampling

test_video = "/home/pep/drive/PCLOUD/Dataset/RWF2000_Dataset/RWF-2000/train/Fight/3kpviz7lAMY_4.avi"

cap = cv2.VideoCapture(test_video)

_, first_frame = cap.read()

prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

queue = []

def calc_flow(queue):
    gray_video = []
    for i in range(len(queue)):
        gray = cv2.cvtColor(queue[i], cv2.COLOR_BGR2GRAY)
        gray_video.append(gray)
    
    flows = []
    for i in range(len(gray_video) - 1):
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i+1], None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # subtract the mean in order to eliminate the movement of camera
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        # normalize each component in optical flow
        flow[..., 0] = cv2.normalize(flow[..., 0],None,0,255,cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1],None,0,255,cv2.NORM_MINMAX)
        flows.append(flow)
    
    # Padding the last frame as empty array
    flows.append(np.zeros((flows[0].shape[0], flows[0].shape[1], 2)))
      
    return np.array(flows, dtype=np.float32)



while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (224, 224))
    if len(queue) <= 0:
        for _ in range(150):
            queue.append(frame)
    
    queue.pop(0)
    queue.append(frame)

    v = np.array(queue)
    samp = uniform_sampling(v.copy())


    flows = getOpticalFlow(samp)

    result = np.zeros((len(flows), 224, 224, 5))
    result[...,:3] = samp
    result[...,3:] = flows
    
    opt_flows = result[..., 3]
    magnitude = np.sum(opt_flows, axis=0)
    thresh = np.mean(magnitude)
    magnitude[magnitude < thresh] = 0
    # calculate center of gravity of magnitude map and adding 0.001 to avoid empty value
    x_pdf = np.sum(magnitude, axis=1) + 0.001
    y_pdf = np.sum(magnitude, axis=0) + 0.001
    # normalize PDF of x and y so that the sum of probs = 1
    x_pdf /= np.sum(x_pdf)
    y_pdf /= np.sum(y_pdf)
    # randomly choose some candidates for x and y 
    x_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=x_pdf)
    y_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=y_pdf)
    # get the mean of x and y coordinates for better robustness
    x = int(np.mean(x_points))
    y = int(np.mean(y_points))
    # print(x, y)
    # avoid to beyond boundaries of array
    x = max(56,min(x,167))
    y = max(56,min(y,167))
    cropped_frame = result[:, x-56:x+56,y-56:y+56, :]
    # magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    frame = result[-1, ..., :3].astype(np.uint8)
    heatmap = cv2.applyColorMap(magnitude.astype(np.uint8), cv2.COLORMAP_INFERNO)
    merge_img = cv2.addWeighted(heatmap, 0.3, frame, 1, 0)
    # cv2.imshow('Frame', frame)
    # cv2.imshow('Magnitude', magnitude)
    cropped_frame = cropped_frame[-1, ..., :3].astype(np.uint8)
    merge_img = cv2.circle(merge_img, (x, y), 3, (0, 0, 255), 3)
    cv2.imshow('heatmap', merge_img)
    cv2.imshow("cropped", cropped_frame)
    # cv2.imshow('Flows', flows[-1, ..., 0])
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
