import cv2
import numpy as np
from tracker import Tracker
import tensorflow as tf


def main(input_video, output_video):
    # Load the TensorFlow SavedModel.
    model_path = "/home/modesto/PycharmProjects/compressed_tracking/TFOD/pretrained_models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/saved_model"
    detector = tf.saved_model.load(model_path)
    infer = detector.signatures['serving_default']
    print(infer)
    # Initialize the DeepSort tracker.
    tracker = Tracker()

    # Open the input video.
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo d'entrée.")
        return

    # Get the video dimensions and create a VideoWriter object for the output video.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Read the input video, detect objects, and update the tracking.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame.astype(np.uint8)
        frame_batch = np.expand_dims(frame, axis=0)

        # Run detection on the object detection model.
        input_tensor = tf.convert_to_tensor(frame_batch)
        detections = infer(input_tensor)

        # Get the number of detections.
        print(detections)
        print(detections['num_detections'])
        print(detections['num_detections'][0])
        print(tf.get_static_value(detections['num_detections']))
        num_detections = int(detections['num_detections'][0])

        detection_boxes = detections['detection_boxes'][0][:num_detections]
        detection_scores = detections['detection_scores'][0][:num_detections]

        # Filter out detections with low confidence.
        min_confidence = 0.5
        indices = tf.where(detection_scores >= min_confidence)

        filtered_boxes = tf.gather(detection_boxes, indices)
        filtered_scores = tf.gather(detection_scores, indices)

        h, w, _ = frame.shape
        absolute_boxes = filtered_boxes * tf.constant([h, w, h, w], dtype=tf.float32)

        formatted_detections = tf.concat([absolute_boxes, tf.expand_dims(filtered_scores, axis=1)], axis=1)

        # Update the DeepSort tracker with the detections.
        tracker.update(frame, formatted_detections.numpy())

        # Draw the detection and tracking boxes on the frame.
        for track in tracker.tracks:
            bbox = track.bbox.astype(int)
            id = track.track_id
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, str(id), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Add the frame to the output video.
        out.write(frame)

        # Display the frame in a window.
        cv2.imshow('Output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close windows.
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    input_video = '/home/modesto/PycharmProjects/compressed_tracking/TFOD/videos/video_1.mp4'
    output_video = '/home/modesto/PycharmProjects/compressed_tracking/TFOD/outputs_videos/video_1.mp4'
    main(input_video, output_video)

