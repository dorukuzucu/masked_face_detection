import numpy as np
from utils import *


# import trained cascades
face_cascade = load_cascade("face")
left_eye_cascade = load_cascade("left-eye")

message_config = {
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_color" : (255, 255, 255),
    "font_scale": 1,
    "mask_message": "Thank you for wearing a mask",
    "no_mask_message": "Please wear a mask!"
}

cap = get_capture((1200,820))

def main():
    try:
        # acquire frames
        while True:

            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Initial Operations on Image
            processed_frame = preprocess_frame(frame)


            # detect left eye using left eye cascade
            left_eyes = left_eye_cascade.detectMultiScale(processed_frame, 1.3, 5)
            left_list, weights = cv2.groupRectangles(list(left_eyes), 1, 2)

            # draw bounding boxes
            for (x, y, w, h) in left_list:
                [x_new, y_new, w_new, h_new] = [x - w, y - h, w * 3, h * 5]
                frame = cv2.rectangle(frame, (x_new, y_new), (x_new + w_new, y_new + h_new), (255, 0, 0), 2)

                # Convert Gray Image to Black and White
                face_area = processed_frame[y_new:y_new + h_new, x_new:x_new + w_new]
                roi_bw = cv2.threshold(face_area[int(h_new * 0.7):h_new, 0:w_new], 100, 255, cv2.THRESH_BINARY)[1]

                if np.mean(roi_bw) > 150:
                    cv2.putText(frame, message_config["mask_message"], (x_new - 30, y_new - 5),
                                message_config["font"],
                                message_config["font_scale"], message_config["font_color"], 2,
                                cv2.LINE_4)
                elif np.mean(roi_bw) < 150:
                    cv2.putText(frame, message_config["no_mask_message"], (x_new - 30, y_new - 5), message_config["font"],
                                message_config["font_scale"], message_config["font_color"], 2,
                                cv2.LINE_4)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            cv2.waitKey(10)

    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()