from imutils.video import FPS
import numpy as np
import cv2
import time

use_gpu = 1
webcam = 1
expected_confidence = 0.3
show_output = 1
save_output = 1

weightsPath = "mask-rcnn-coco/frozen_inference_graph.pb"
configPath = "mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

if use_gpu:
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

print("[INFO] accessing video stream...")
cap = cv2.VideoCapture(0) if webcam else cv2.VideoCapture('humans.mp4')

writer = None
fps = FPS().start()

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('output.avi', fourcc, 10, (int(cap.get(3)), int(cap.get(4))), True)

# Performance metrics
total_frames = 0
true_positive_count = 0
false_positive_count = 0
latency_times = []
processing_times = []

# Main loop
while True:
    grabbed, frame = cap.read()
    if not grabbed:
        break
    
    start_time = time.time()  # Start latency timer
    cv2.imshow('org', frame)

    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

    detection_made = False  # Flag to track true positives
    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        if classID != 0:
            continue  # Ignore non-person classes
        
        confidence = boxes[0, 0, i, 2]
        if confidence > expected_confidence:
            detection_made = True
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Draw the bounding box on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"Person: {confidence:.2f}"
            cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update performance metrics
    if detection_made:
        true_positive_count += 1
    else:
        false_positive_count += 1

    total_frames += 1
    latency_times.append(time.time() - start_time)  # Frame latency

    # Show the output
    if show_output:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            break

    # Save the output
    if save_output:
        writer.write(frame)

    fps.update()
    processing_times.append(time.time() - start_time)  # Processing time for frame

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Calculate metrics
avg_latency = sum(latency_times) / total_frames
avg_processing_time = sum(processing_times) / total_frames
accuracy = true_positive_count / total_frames
precision = true_positive_count / (true_positive_count + false_positive_count)

# Print the metrics
print("[INFO] Average Latency per frame: {:.4f} seconds".format(avg_latency))
print("[INFO] Average Processing Time per frame: {:.4f} seconds".format(avg_processing_time))
print("[INFO] Detection Accuracy: {:.2%}".format(accuracy))
print("[INFO] Detection Precision: {:.2%}".format(precision))

# Release resources
cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()