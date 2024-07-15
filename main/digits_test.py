from tensorflow.keras import models
import cv2
import numpy as np
import time
import func


func.create_my_sess(0.5)

drawing = False
pt1_x, pt1_y = None, None

# mouse callback function


def line_drawing(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            cv2.line(img, (pt1_x, pt1_y), (x, y),
                     color=(255, 255, 255), thickness=15)
            pt1_x, pt1_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt1_x, pt1_y), (x, y),
                 color=(255, 255, 255), thickness=15)


img = np.zeros((256, 256, 3), np.uint8)
image_text = np.zeros((256, 256, 3), np.uint8)

cv2.namedWindow('test draw')
cv2.namedWindow('test digits')

cv2.setMouseCallback('test draw', line_drawing)

model = models.load_model('mnist.h5')

past_digit = None
present_digit = None

while True:
    img_test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_test = cv2.resize(img_test, (28, 28))
    img_test = img_test.reshape((1, 28, 28, 1))
    img_test = img_test.astype('float32') / 255.

    prediction = np.array(model.predict(img_test))

    cv2.imshow(('test draw'), img)

    if np.max(img) == 0:
        image_text = np.zeros((256, 256, 3), np.uint8)
    else:
        present_digit = np.argmax(prediction[0])

        if present_digit != past_digit:
            image_text = np.zeros((256, 256, 3), np.uint8)

            cv2.putText(image_text, f'{np.argmax(prediction[0])}',
                        (image_text.shape[0] // 2 - 50,
                         image_text.shape[1] // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 5)
            pass_digit = present_digit

    cv2.imshow('test digits', image_text)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
        break
    if k == ord('d'):
        img = np.zeros((256, 256, 3), np.uint8)

    time.sleep(0.1)

cv2.destroyAllWindows()
