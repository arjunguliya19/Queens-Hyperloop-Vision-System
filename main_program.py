"""
COMMENTS:
We tried just importing the object detection as a function, but it led to low framerate because it kept reinitizalizing the network
Instead we just put the code in this file.
Also, a lot of this stuff can be made into classes and objects but we are C programmers so :(
"""

def object_detect(imported_frame):
    """
    FOR MORE INFORMATION ON OBJECT DETECTION, CHECK YOLO-3-CAMERA.py IN "object_detection/YOLO-3-OpenCV"
    """
    h, w = None, None
    while True:
        _, frame = imported_frame.read()
        if w is None or h is None:
            h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        network.setInput(blob)
        output_from_network = network.forward(layers_names_output)
        bounding_boxes = []
        confidences = []
        class_numbers = []
        for result in output_from_network:
            for detected_objects in result:
                scores = detected_objects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]
                if confidence_current > probability_minimum:
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))
                    bounding_boxes.append([x_min, y_min,
                                           int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                   probability_minimum, threshold)
        if len(results) > 0:
            for i in results.flatten():
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                colour_box_current = colours[class_numbers[i]].tolist()
                cv2.rectangle(frame, (x_min, y_min),
                              (x_min + box_width, y_min + box_height),
                              colour_box_current, 2)
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                       confidences[i])
                cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
        return frame, text_box_current

if __name__ == '__main__':
    ##LIBRARY IMPORTS AND GLOBAL VARIABLES
    import os
    import datetime
    import time
    import detection
    import math
    import cv2
    import numpy as np
    from tkinter import *
    from PIL import Image, ImageTk
    from pathlib import Path
    global network, labels, layers_names_all, layers_names_output, probability_minimum, threshold, colours, start, critical
    count = 160

    #timer for object detection
    start = None
    if start == None:
        start = time.time()

    #intitializing object detection network
    with open('object_detection/YOLO-3-OpenCV/yolo-coco-data/coco.names') as f:
        labels = [line.strip() for line in f]
    network = cv2.dnn.readNetFromDarknet('object_detection/YOLO-3-OpenCV/yolo-coco-data/yolov4.cfg',
                                         'object_detection/YOLO-3-OpenCV/yolo-coco-data/yolov4.weights')

    """
    ENABLE THIS IF YOU HAVE OPENCV BUILT WITH CUDA
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    """

    layers_names_all = network.getLayerNames()
    layers_names_output = \
        [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
    probability_minimum = 0.5
    threshold = 0.3
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    """
    FOR MORE INFORMATION ON OBJECT DETECTION, CHECK YOLO-3-CAMERA.py IN "object_detection_sample_code/YOLO-3-OpenCV"
    """

    #setting the dictionary to the current folder
    curr_directory = str(os.getcwd())
    OUTPUT_PATH = Path(__file__).parent
    ASSETS_PATH = OUTPUT_PATH / Path(curr_directory+"\\frame")

    #Original Design Dimensions
    DEFAULT_WIDTH = 1920
    DEFAULT_HEIGHT = 1080

    #Create a Tkinter Window
    window = Tk()

    #Gets User's Screen Size
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    #Scale Factor Calculation
    w_scale_factor = screen_width / DEFAULT_WIDTH
    h_scale_factor = screen_height / DEFAULT_HEIGHT

    #Scales Based on Smallest Sidelength
    scale_factor = min(w_scale_factor, h_scale_factor)

    #Canvas Size Calculation
    canvas_width = int(scale_factor * DEFAULT_WIDTH)
    canvas_height = int(scale_factor * DEFAULT_HEIGHT)

    #Creates UI Canvas
    window.geometry(f"{canvas_width}x{canvas_height}")
    window.configure(bg = "#1F1C25")

    canvas = Canvas(
        window,
        bg = "#1F1C25",
        height = canvas_height,
        width = canvas_width,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )
    canvas.pack()

    #Function returining the absolute path for a file relative to ASSETS_PATH
    #Was generated from Parth Jadhav's application
    def relative_to_assets(path: str) -> Path:
        return str(Path(ASSETS_PATH / Path(path)).absolute())

    #Function for Scaling Position
    def scale_value(value: float) -> float:
        return value * scale_factor

    #Places Canvas Top Left Corner of Parent Widget
    canvas.place(x = 0, y = 0)

    #Resizing Image Setup
    image_files = ["image_1.png", "image_2.png", "image_3.png", "image_4.png", "image_5.png", "image_6.png", "image_7.png", "image_8.png", "image_9.png", "image_10.png", "image_11.png"]
    images = []
    for file in image_files:
        resized_image = Image.open(relative_to_assets(file))
        resized_image = resized_image.resize((int(resized_image.width * scale_factor), int(resized_image.height * scale_factor)))
        images.append(ImageTk.PhotoImage(resized_image))

    #Creates Rectangle for Top Header
    canvas.create_rectangle(
        scale_value(0.0),
        scale_value(15.24444580078125),
        scale_value(1920.0),
        scale_value(127.24444580078125),
        fill="#383442",
        outline="")

    #Creates Text for Top Header
    canvas.create_text(
        scale_value(454.0),
        scale_value(44.24444580078125),
        anchor="nw",
        text="Hyperloop Pod Vision Sensor System Panel",
        fill="#FFFFFF",
        font=("Alata Regular", int(48 * -1 * scale_factor)))

    #Creates Hyperloop Logo
    canvas.create_image(scale_value(111.0), scale_value(71.24444580078125),image = images[0])

    #Creates Images for Live Video Feed Panel
    canvas.create_image(scale_value(670.0), scale_value(172.24444580078125), image = images[1])
    canvas.create_image(scale_value(670.0), scale_value(924.2444458007812), image = images[2])
    canvas.create_image(scale_value(55.497314453125), scale_value(172.599609375), image = images[5])

    #Creates Rectangle for Live Video Feed Panel
    canvas.create_rectangle(
        scale_value(30.0),
        scale_value(198.24444580078125),
        scale_value(1310.0),
        scale_value(918.2444458007812),
        fill="#383443",
        outline="")

    #Live Video Feed Panel Text
    canvas.create_text(
        scale_value(82),
        scale_value(160),
        anchor="nw",
        text="Live Video Feed",
        fill="#FFFFFF",
        font=("Commissioner Bold", int(25 * -1 * scale_factor)))

    #Creates Images for Object Detection Panel
    canvas.create_image(scale_value(1615.0), scale_value(172.24444580078125), image = images[3])
    canvas.create_image(scale_value(1615.0), scale_value(924.2444458007812), image = images[4])

    #Create Rectangles for Object Detection Panel
    canvas.create_rectangle(
        scale_value(1341.0),
        scale_value(198.24444580078125),
        scale_value(1890.0),
        scale_value(918.2444458007812),
        fill="#383442",
        outline="")

    canvas.create_rectangle(
        scale_value(1881.0),
        scale_value(207.24444580078125),
        scale_value(1888.0),
        scale_value(913.2444458007812),
        fill="#2D2936",
        outline="")

    #Object Detection Panel Text
    canvas.create_text(
        scale_value(1362.0),
        scale_value(160.0),
        anchor="nw",
        text="Objects Detected",
        fill="#FFFFFF",
        font=("Commissioner Bold", int(25 * -1 * scale_factor)))

    #Create Status Message Box
    #Status Message Box(1018)
    canvas.create_image(scale_value(959), scale_value(992.4703979492188), image = images[6])

    #Create Warning Status Message Images
    warning_1 = canvas.create_image(scale_value(363), scale_value(986),image = images[7])
    warning_2 = canvas.create_image(scale_value(1553), scale_value(986), image = images[8])

    #Create Alert Status Message Image
    alert_1 = canvas.create_image(scale_value(363), scale_value(986), image = images[9])
    alert_2 = canvas.create_image(scale_value(1553), scale_value(986), image = images[10])

    #Creates Status Message Text
    status_text = canvas.create_text(
        scale_value(511),
        scale_value(960),
        anchor="nw",
        text="Status: On Track Towards Destination",
        fill="#FFFFFF",
        font=("Commissioner Bold", int(51 * -1 * scale_factor)))

    object_alert_message = canvas.create_text(
        scale_value(412),
        scale_value(960),
        anchor="nw",
        text="CAUTION: OBJECT DETECTED ON TRACK ",
        fill="#FFD42A",
        font=("Commissioner Bold", int(54 * -1 * scale_factor)))

    pod_warning_message = canvas.create_text(
        scale_value(411),
        scale_value(960),
        anchor="nw",
        text="WARNING: POD DEVIATING FROM TRACK",
        fill="#ED1C24",
        font=("Commissioner Bold", int(54 * -1 * scale_factor)))

    object_warning_message = canvas.create_text(
        scale_value(419),
        scale_value(960),
        anchor="nw",
        text="WARNING: OBJECT OBSTRUCTING PATH",
        fill="#ED1C24",
        font=("Commissioner Bold", int(54 * -1 * scale_factor)))

    #temporary Status Message Toggle Function
    def toggle_text(text_item, show):
        state = "normal" if show else "hidden"
        canvas.itemconfigure(text_item, state=state)

    #toggle Warning Texts
    #set to "True" to display text and "False" to hide text
    toggle_text(status_text, False)
    toggle_text(object_alert_message, False)
    toggle_text(pod_warning_message, False)
    toggle_text(object_warning_message, False)

    #toggle Warning Images
    #set to "normal" to display image and "hidden" to hide image
    canvas.itemconfig(alert_1, state = "normal")
    canvas.itemconfig(alert_2, state = "normal")
    canvas.itemconfig(warning_1, state = "hidden")
    canvas.itemconfig(warning_2, state = "hidden")

    #capturing camera video, change brackets to (0) for camera feed or ("videofilename.mp4") for pre-recorded video
    cap = cv2.VideoCapture(0)

    #creating widget for video
    video_widget = Label(window)
    video_widget.place(bordermode=OUTSIDE, x=scale_value(30.0),y=scale_value(198.24444580078125))

    #creating scaling for video widget, rounding b/c tkinter only takes integer sizes
    video_width=math.floor(scale_value(1310.0)-scale_value(30.0))
    video_height=math.ceil(scale_value(918.2444458007812)-scale_value(198.24444580078125))

    def video_feed():
        ##CALLING LANE DETECTION
        try:
            #standard lane detection input
            lane_frame_bright, is_obstructed = detection.videoCapture(cap)

            #displaying alert message if something is obstructing the lane
            if is_obstructed is True:
                lane_frame = cv2.convertScaleAbs(lane_frame_bright, 1, 0)
                critical = True
                toggle_text(status_text, False)
                toggle_text(object_alert_message, False)
                toggle_text(pod_warning_message, False)
                toggle_text(object_warning_message, True)
                canvas.itemconfig(warning_1, state="normal")
                canvas.itemconfig(warning_2, state="normal")
                canvas.itemconfig(alert_1, state="hidden")
                canvas.itemconfig(alert_2, state="hidden")

            #else, displays no alert message, pod resumes as normal
            else:
                lane_frame = lane_frame_bright
                critical = False
                toggle_text(object_alert_message, False)
                toggle_text(object_warning_message, False)
                toggle_text(pod_warning_message, False)
                toggle_text(status_text, True)
                canvas.itemconfig(alert_1, state="hidden")
                canvas.itemconfig(alert_2, state="hidden")
                canvas.itemconfig(warning_1, state="hidden")
                canvas.itemconfig(warning_2, state="hidden")

        except:
            #turns on critical error message if lane cannot be detected
            critical = True
            _, lane_frame_bright = cap.read()
            lane_frame = cv2.convertScaleAbs(lane_frame_bright, 0.8, 0)
            toggle_text(object_warning_message, False)
            toggle_text(status_text, False)
            toggle_text(object_alert_message, False)
            toggle_text(pod_warning_message, True)
            canvas.itemconfig(warning_1, state="normal")
            canvas.itemconfig(warning_2, state="normal")
            canvas.itemconfig(alert_1, state="hidden")
            canvas.itemconfig(alert_2, state="hidden")

        ##CALLING OBJECT DETECTION
        try:
            object_frame, object_info = object_detect(cap)

            #detects if object was returned
            if object_info != "":
                #sorting information
                global image_storage
                image_storage = True
                object_name = object_info.split()[0].replace(":", "").capitalize()
                percent_accuracy = float((object_info.split()[1])) * 100
                date = str(datetime.datetime.now().replace(microsecond=0))
                file_date = date.split()[0]
                object_text = f"{object_name}     {percent_accuracy:1.2f}%    {date}"

                #toggling object detection text
                if critical is False:
                    toggle_text(status_text, False)
                    toggle_text(object_alert_message, True)
                    canvas.itemconfig(alert_1, state="normal")
                    canvas.itemconfig(alert_2, state="normal")

                #timer w/ intervals to make sure right hand side isn't crowded
                global start
                end = time.time()

                #INCREASE OR DECREASE THIS VALUE TO CHANGE DATA INTERVAL
                if (end - start) > 1:
                    #resetting time
                    start = time.time()

                    #iterating count, count determines the object text placement
                    global count
                    count += 50

                    #clearing canvas if text goes beyond rectangle, resetting count
                    if count > 860:
                        count = 210
                        canvas.delete('object_text')

                    #creating text
                    canvas.create_text(
                        scale_value(1362.0),
                        scale_value(count),
                        anchor="nw",
                        text=object_text,
                        fill="#FFFFFF",
                        font=("Commissioner Bold", int(25 * -1 * scale_factor)),
                        tags=('object_text')
                        )

                    #saving data to a log file
                    with open(f'logs/{file_date}.txt', 'a') as f:
                        f.write(f'{object_text} \n')

        #if object detection does not have any information, toggle messages
            else:
                if critical is False:
                    toggle_text(object_alert_message, False)
                    toggle_text(object_warning_message, False)
                    toggle_text(pod_warning_message, False)
                    toggle_text(status_text, True)
                    canvas.itemconfig(alert_1, state="hidden")
                    canvas.itemconfig(alert_2, state="hidden")

        except:
            #cannot detect object, setting frame to video frame
            _, object_frame_bright = cap.read()
            object_frame = cv2.convertScaleAbs(object_frame_bright, 0.8, 1)

            #toggling status messages
            canvas.itemconfig(alert_1, state="hidden")
            canvas.itemconfig(alert_2, state="hidden")

            #checking if pod is in critical state
            if critical is False:
                toggle_text(object_alert_message, False)
                toggle_text(object_warning_message, False)
                toggle_text(pod_warning_message, False)
                toggle_text(status_text, True)
                canvas.itemconfig(warning_1, state="hidden")
                canvas.itemconfig(warning_2, state="hidden")

        ##IMAGE REFORMATTING
        #converts the images to IRL colors, BGR spectrum to RGB
        try:
            converted_image_object = cv2.cvtColor(object_frame, cv2.COLOR_BGR2RGB)
            converted_image_lane = cv2.cvtColor(lane_frame, cv2.COLOR_BGR2RGB)
        except:
            video_feed()
            return

        #resizes images regardless of webcam resolution
        resized_image_object = cv2.resize(converted_image_object, (video_width, video_height))
        resized_image_lane = cv2.resize(converted_image_lane, (video_width, video_height))

        #superimposing images
        blended_image = cv2.addWeighted(resized_image_lane, 1, resized_image_object, 1, 1)

        #creating image from blended image array
        capture_image = Image.fromarray(blended_image)

        #creates a tkinter image with img
        photo_image = ImageTk.PhotoImage(image=capture_image)

        #inserts tkinter image into widget
        video_widget.imgtk = photo_image
        video_widget.configure(borderwidth=0, image=photo_image)

        #logs object detection image
        if image_storage == True:
            #had to replace b/c files cannot have ':'
            date_list = date.replace(":", "-")
            date_update = date_list.split()
            cv2.imwrite(f'image_logs/{date_update[0]}_{date_update[1]}_{object_name}.png', object_frame)
            image_storage = False

        #recalls function after 16ms (matches 60fps, approx 16ms per frame) to update videofeed, change value to match any camera framerate
        video_widget.after(16, video_feed)

    video_feed()
    window.resizable(True, True)
    window.mainloop()
