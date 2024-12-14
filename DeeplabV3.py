import tensorflow as tf
import numpy as np
import cv2
import time

# Load the frozen inference graph
model_path = "deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb"
print("[INFO] Loading DeepLabv3 model...")

# Load the frozen graph
def load_graph(model_file):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    return graph

# Load the model
graph = load_graph(model_path)

# Define the input and output tensors
input_tensor_name = "ImageTensor:0"
output_tensor_name = "SemanticPredictions:0"

# Define a color map for the PASCAL VOC dataset
def create_color_map():
    return np.array([
        [0, 0, 0],        # Background
        [128, 0, 0],      # Aeroplane
        [0, 128, 0],      # Bicycle
        [128, 128, 0],    # Bird
        [0, 0, 128],      # Boat
        [128, 0, 128],    # Bottle
        [0, 128, 128],    # Car
        [128, 128, 128],  # Chair
        [64, 0, 0],       # Cow
        [192, 0, 0],      # Dining Table
        [64, 128, 0],     # Dog
        [192, 128, 0],    # Horse
        [64, 0, 128],     # Motorbike
        [192, 0, 128],    # Person
        [64, 128, 128],   # Potted Plant
        [192, 128, 128],  # Sheep
        [0, 64, 0],       # Sofa
        [128, 64, 0],     # Train
        [0, 192, 0],      # TV/Monitor
        [255, 255, 255],  # Ignore class
    ], dtype=np.uint8)

# Setup Video Capture
cap = cv2.VideoCapture(0)

# Performance metrics
total_frames = 0
true_positive_count = 0
false_positive_count = 0
false_negative_count = 0
latency_times = []

# Color map for visualization
color_map = create_color_map()

with tf.compat.v1.Session(graph=graph) as sess:
    def run_deeplabv3_inference(frame):
        # Preprocess frame for DeepLabv3 input
        input_frame = cv2.resize(frame, (513, 513))
        input_frame = np.expand_dims(input_frame, axis=0)

        # Run model
        output = sess.run(output_tensor_name, feed_dict={input_tensor_name: input_frame})

        # Post-process the output to get the segmentation mask
        segmentation_mask = output[0]

        return segmentation_mask

    # Main loop
    fps_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Run inference
        segmentation_mask = run_deeplabv3_inference(frame)

        # Clip segmentation mask to valid indices
        segmentation_mask = np.clip(segmentation_mask, 0, len(color_map) - 1)

        # Convert segmentation mask to color
        colored_mask = color_map[segmentation_mask]

        # Resize colored_mask to match the original frame size
        colored_mask_resized = cv2.resize(colored_mask, (frame.shape[1], frame.shape[0]))

        # Overlay the colored mask on the original frame
        output_frame = cv2.addWeighted(frame, 0.7, colored_mask_resized, 0.3, 0)

        # Display the output frame
        cv2.imshow("DeepLabv3 Segmentation", output_frame)
        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
            break

        # Update metrics for accuracy and precision calculation
        total_frames += 1
        true_positive_count += np.sum(segmentation_mask == 15)  # Count pixels classified as "person"
        
        # Count pixels incorrectly classified as "person" (false positives)
        false_positive_count += np.sum(segmentation_mask > 15)
        
        # Count false negatives (missing "person" pixels)
        # Here we can define a ground truth mask for comparison if available
        # For example, assume we have a way to get the ground truth (gt_mask)
        # gt_mask = (some method to get ground truth for the current frame)
        # false_negative_count += np.sum((gt_mask == 15) & (segmentation_mask != 15))

        latency_times.append(time.time() - start_time)

    # Calculate FPS and metrics
    fps_end = time.time()
    fps = total_frames / (fps_end - fps_start)
    avg_latency = sum(latency_times) / total_frames

    if total_frames > 0:
        accuracy = true_positive_count / (true_positive_count + false_negative_count) if (true_positive_count + false_negative_count) > 0 else 0
        precision = true_positive_count / (true_positive_count + false_positive_count) if (true_positive_count + false_positive_count) > 0 else 0
    else:
        accuracy = 0
        precision = 0

    # Print metrics
    print(f"[INFO] FPS: {fps:.2f}")
    print(f"[INFO] Average Latency per frame: {avg_latency:.4f} seconds")
    print(f"[INFO] Detection Accuracy: {accuracy:.2%}")
    print(f"[INFO] Detection Precision: {precision:.2%}")

# Release resources
cap.release()
cv2.destroyAllWindows()