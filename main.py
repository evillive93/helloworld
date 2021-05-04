import threading
from functools import partial
import cv2
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from utils import backbone
import tensorflow as tf
import csv
import cv2
import numpy as np
from utils import visualization_utils as vis_util

# Variables
total_passed_objects = 0  # using it to count objects
from api import object_counting_api

class MainScreen(Screen):
    pass


class Manager(ScreenManager):
    pass


Builder.load_string('''
<MainScreen>:
    name: "Test"

    FloatLayout:
        Label:
            text: "Webcam from OpenCV?"
            pos_hint: {"x":0.0, "y":0.8}
            size_hint: 1.0, 0.2

        Image:
            # this is where the video will show
            # the id allows easy access
            id: vid
            size_hint: 1, 0.6
            allow_stretch: True  # allow the video image to be scaled
            keep_ratio: True  # keep the aspect ratio so people don't look squashed
            pos_hint: {'center_x':0.5, 'top':0.8}

        Button:
            text: 'Stop Video'
            pos_hint: {"x":0.0, "y":0.0}
            size_hint: 1.0, 0.2
            font_size: 50
            on_release: app.stop_vid()
''')


class Main(App):
    def build(self):

        # start the camera access code on a separate thread
        # if this was done on the main thread, GUI would stop
        # daemon=True means kill this thread when app stops
        threading.Thread(target=self.doit, daemon=True).start()

        sm = ScreenManager()
        self.main_screen = MainScreen()
        sm.add_widget(self.main_screen)
        return sm

    def doit(self):
        # this code is run in a separate thread
        self.do_vid = True  # flag to stop loop

        # make a window for use by cv2
        # flags allow resizing without regard to aspect ratio
        cv2.namedWindow('Hidden', cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

        # resize the window to (0,0) to make it invisible
        cv2.resizeWindow('Hidden', 0, 0)
        cam = cv2.VideoCapture(0)
        detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28',
                                                             'mscoco_label_map.pbtxt')

        is_color_recognition_enabled = False  # set it to true for enabling the color prediction for the detected objects


        # start processing loop
        while (self.do_vid):

            color = "waiting..."
            with detection_graph.as_default():
                with tf.compat.v1.Session(graph=detection_graph) as sess:
                    # Definite input and output Tensors for detection_graph
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                    # Each box represents a part of the image where a particular object was detected.
                    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    #cap = cv2.VideoCapture(0)
                    (ret, frame) = cam.read()

                    # for all the frames that are extracted from input video
                    while True:
                        # Capture frame-by-frame
                        (ret, frame) = cam.read()

                        if not ret:
                            print("end of the video file...")
                            break

                        input_frame = frame

                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(input_frame, axis=0)

                        # Actual detection.
                        (boxes, scores, classes, num) = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})

                        # insert information text to video frame
                        font = cv2.FONT_HERSHEY_SIMPLEX

                        # Visualization of the results of a detection.
                        counter, csv_line, counting_result = vis_util.visualize_boxes_and_labels_on_image_array(
                            cam.get(1),
                            input_frame,
                            is_color_recognition_enabled,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=4)
                        if (len(counting_result) == 0):
                            cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0, 255, 255), 2,
                                        cv2.FONT_HERSHEY_SIMPLEX)
                        else:
                            cv2.putText(input_frame, counting_result, (10, 35), font, 0.8, (0, 255, 255), 2,
                                        cv2.FONT_HERSHEY_SIMPLEX)

                       # cv2.imshow('object counting', input_frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                            # ...
                            # more code
                            # ...

                            # send this frame to the kivy Image Widget
                            # Must use Clock.schedule_once to get this bit of code
                            # to run back on the main thread (required for GUI operations)
                            # the partial function just says to call the specified method with the provided argument (Clock adds a time argument)
                        Clock.schedule_once(partial(self.display_frame, input_frame))

                        cv2.imshow('Hidden', input_frame)
                        cv2.waitKey(1)



                    cam.release()
                    cv2.destroyAllWindows()

    def stop_vid(self):
        # stop the video capture loop
        self.do_vid = False

    def display_frame(self, frame, dt):
        # display the current video frame in the kivy Image widget

        # create a Texture the correct size and format for the frame
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')

        # copy the frame data into the texture
        texture.blit_buffer(frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')

        # flip the texture (otherwise the video is upside down
        texture.flip_vertical()

        # actually put the texture in the kivy Image widget
        self.main_screen.ids.vid.texture = texture


if __name__ == '__main__':
    Main().run()