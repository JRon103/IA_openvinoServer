from flask import Flask, request, jsonify
import os



import os
from pathlib import Path
import numpy as np
import math
import cv2
import tarfile
import matplotlib.pyplot as plt
import openvino as ov

# Fetch `notebook_utils` module
import requests




app = Flask(__name__)
UPLOAD_FOLDER = 'C:/Users/villa/IA/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Crear la carpeta de subidas si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/subir', methods=['POST'])
def openvino():

    import platform

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )

    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file, segmentation_map_to_image


    # ## Prepare the Model and Test Image
    # Download PPYOLOv2 and DeepLabV3P pre-trained models from PaddlePaddle community.

    MODEL_DIR = "model"
    DATA_DIR = "data"
    DET_MODEL_LINK = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/meter-reader/meter_det_model.tar.gz"
    SEG_MODEL_LINK = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/meter-reader/meter_seg_model.tar.gz"
    DET_FILE_NAME = DET_MODEL_LINK.split("/")[-1]
    SEG_FILE_NAME = SEG_MODEL_LINK.split("/")[-1]
    #IMG_LINK = "https://user-images.githubusercontent.com/91237924/170696219-f68699c6-1e82-46bf-aaed-8e2fc3fa5f7b.jpg"
    #IMG_FILE_NAME = IMG_LINK.split("/")[-1]
    #IMG_PATH = Path(f"{DATA_DIR}/{IMG_FILE_NAME}")
    #print("path antes:", IMG_PATH)
    #IMG_PATH = Path("C:/Users/villa/IA/data/170696219-f68699c6-1e82-46bf-aaed-8e2fc3fa5f7b.jpg")
    #print("path des:", IMG_PATH)
    IMG_PATH = Path("C:/Users/villa/IA/uploads/image.jpg")
    # In[8]:





    # In[9]:


    os.makedirs(MODEL_DIR, exist_ok=True)

    download_file(DET_MODEL_LINK, directory=MODEL_DIR, show_progress=True)
    file = tarfile.open(f"model/{DET_FILE_NAME}")
    res = file.extractall("model")
    if not res:
        print(f'Detection Model Extracted to "./{MODEL_DIR}".')
    else:
        print("Error Extracting the Detection model. Please check the network.")

    download_file(SEG_MODEL_LINK, directory=MODEL_DIR, show_progress=True)
    file = tarfile.open(f"model/{SEG_FILE_NAME}")
    res = file.extractall("model")
    if not res:
        print(f'Segmentation Model Extracted to "./{MODEL_DIR}".')
    else:
        print("Error Extracting the Segmentation model. Please check the network.")

    # download_file(IMG_LINK, directory=DATA_DIR, show_progress=True)
    # if IMG_PATH.is_file():
    #     print(f'Test Image Saved to "./{DATA_DIR}".')
    # else:
    #     print("Error Downloading the Test Image. Please check the network.")


    # ## Configuration
    # [back to top ⬆️](#Table-of-contents:)
    # Add parameter configuration for reading calculation.

    # In[10]:


    METER_SHAPE = [512, 512]
    CIRCLE_CENTER = [256, 256]
    CIRCLE_RADIUS = 250
    PI = math.pi
    RECTANGLE_HEIGHT = 120
    RECTANGLE_WIDTH = 1570
    TYPE_THRESHOLD = 40
    COLORMAP = np.array([[28, 28, 28], [238, 44, 44], [250, 250, 250]])

    # There are 2 types of meters in test image datasets
    METER_CONFIG = [
        {"scale_interval_value": 25.0 / 50.0, "range": 25.0, "unit": "(MPa)"},
        {"scale_interval_value": 1.6 / 32.0, "range": 1.6, "unit": "(MPa)"},
    ]

    SEG_LABEL = {"background": 0, "pointer": 1, "scale": 2}


    # ## Load the Models
    # [back to top ⬆️](#Table-of-contents:)
    # Define a common class for model loading and inference

    # In[11]:


    # Initialize OpenVINO Runtime
    core = ov.Core()


    class Model:
        """
        This class represents a OpenVINO model object.

        """

        def __init__(self, model_path, new_shape, device="CPU"):
            """
            Initialize the model object

            Param:
                model_path (string): path of inference model
                new_shape (dict): new shape of model input

            """
            self.model = core.read_model(model=model_path)
            self.model.reshape(new_shape)
            self.compiled_model = core.compile_model(model=self.model, device_name=device)
            self.output_layer = self.compiled_model.output(0)

        def predict(self, input_image):
            """
            Run inference

            Param:
                input_image (np.array): input data

            Retuns:
                result (np.array)): model output data
            """
            result = self.compiled_model(input_image)[self.output_layer]
            return result


    # ## Data Process
    # [back to top ⬆️](#Table-of-contents:)
    # Including the preprocessing and postprocessing tasks of each model.

    # In[12]:


    def det_preprocess(input_image, target_size):
        """
        Preprocessing the input data for detection task

        Param:
            input_image (np.array): input data
            size (int): the image size required by model input layer
        Retuns:
            img.astype (np.array): preprocessed image

        """
        img = cv2.resize(input_image, (target_size, target_size))
        img = np.transpose(img, [2, 0, 1]) / 255
        img = np.expand_dims(img, 0)
        img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        img -= img_mean
        img /= img_std
        return img.astype(np.float32)


    def filter_bboxes(det_results, score_threshold):
        """
        Filter out the detection results with low confidence

        Param：
            det_results (list[dict]): detection results
            score_threshold (float)： confidence threshold

        Retuns：
            filtered_results (list[dict]): filter detection results

        """
        filtered_results = []
        for i in range(len(det_results)):
            if det_results[i, 1] > score_threshold:
                filtered_results.append(det_results[i])
        return filtered_results


    def roi_crop(image, results, scale_x, scale_y):
        """
        Crop the area of detected meter of original image

        Param：
            img (np.array)：original image。
            det_results (list[dict]): detection results
            scale_x (float): the scale value in x axis
            scale_y (float): the scale value in y axis

        Retuns：
            roi_imgs (list[np.array]): the list of meter images
            loc (list[int]): the list of meter locations

        """
        roi_imgs = []
        loc = []
        for result in results:
            bbox = result[2:]
            xmin, ymin, xmax, ymax = [
                int(bbox[0] * scale_x),
                int(bbox[1] * scale_y),
                int(bbox[2] * scale_x),
                int(bbox[3] * scale_y),
            ]
            sub_img = image[ymin : (ymax + 1), xmin : (xmax + 1), :]
            roi_imgs.append(sub_img)
            loc.append([xmin, ymin, xmax, ymax])
        return roi_imgs, loc


    def roi_process(input_images, target_size, interp=cv2.INTER_LINEAR):
        """
        Prepare the roi image of detection results data
        Preprocessing the input data for segmentation task

        Param：
            input_images (list[np.array])：the list of meter images
            target_size (list|tuple)： height and width of resized image， e.g [heigh,width]
            interp (int)：the interp method for image reszing

        Retuns：
            img_list (list[np.array])：the list of processed images
            resize_img (list[np.array]): for visualization

        """
        img_list = list()
        resize_list = list()
        for img in input_images:
            img_shape = img.shape
            scale_x = float(target_size[1]) / float(img_shape[1])
            scale_y = float(target_size[0]) / float(img_shape[0])
            resize_img = cv2.resize(img, None, None, fx=scale_x, fy=scale_y, interpolation=interp)
            resize_list.append(resize_img)
            resize_img = resize_img.transpose(2, 0, 1) / 255
            img_mean = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
            img_std = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
            resize_img -= img_mean
            resize_img /= img_std
            img_list.append(resize_img)
        return img_list, resize_list


    def erode(seg_results, erode_kernel):
        """
        Erode the segmentation result to get the more clear instance of pointer and scale

        Param：
            seg_results (list[dict])：segmentation results
            erode_kernel (int): size of erode_kernel

        Return：
            eroded_results (list[dict])： the lab map of eroded_results

        """
        kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
        eroded_results = seg_results
        for i in range(len(seg_results)):
            eroded_results[i] = cv2.erode(seg_results[i].astype(np.uint8), kernel)
        return eroded_results


    def circle_to_rectangle(seg_results):
        """
        Switch the shape of label_map from circle to rectangle

        Param：
            seg_results (list[dict])：segmentation results

        Return：
            rectangle_meters (list[np.array])：the rectangle of label map

        """
        rectangle_meters = list()
        for i, seg_result in enumerate(seg_results):
            label_map = seg_result

            # The size of rectangle_meter is determined by RECTANGLE_HEIGHT and RECTANGLE_WIDTH
            rectangle_meter = np.zeros((RECTANGLE_HEIGHT, RECTANGLE_WIDTH), dtype=np.uint8)
            for row in range(RECTANGLE_HEIGHT):
                for col in range(RECTANGLE_WIDTH):
                    theta = PI * 2 * (col + 1) / RECTANGLE_WIDTH

                    # The radius of meter circle will be mapped to the height of rectangle image
                    rho = CIRCLE_RADIUS - row - 1
                    y = int(CIRCLE_CENTER[0] + rho * math.cos(theta) + 0.5)
                    x = int(CIRCLE_CENTER[1] - rho * math.sin(theta) + 0.5)
                    rectangle_meter[row, col] = label_map[y, x]
            rectangle_meters.append(rectangle_meter)
        return rectangle_meters


    def rectangle_to_line(rectangle_meters):
        """
        Switch the dimension of rectangle label map from 2D to 1D

        Param：
            rectangle_meters (list[np.array])：2D rectangle OF label_map。

        Return：
            line_scales (list[np.array])： the list of scales value
            line_pointers (list[np.array])：the list of pointers value

        """
        line_scales = list()
        line_pointers = list()
        for rectangle_meter in rectangle_meters:
            height, width = rectangle_meter.shape[0:2]
            line_scale = np.zeros((width), dtype=np.uint8)
            line_pointer = np.zeros((width), dtype=np.uint8)
            for col in range(width):
                for row in range(height):
                    if rectangle_meter[row, col] == SEG_LABEL["pointer"]:
                        line_pointer[col] += 1
                    elif rectangle_meter[row, col] == SEG_LABEL["scale"]:
                        line_scale[col] += 1
            line_scales.append(line_scale)
            line_pointers.append(line_pointer)
        return line_scales, line_pointers


    def mean_binarization(data_list):
        """
        Binarize the data

        Param：
            data_list (list[np.array])：input data

        Return：
            binaried_data_list (list[np.array])：output data。

        """
        batch_size = len(data_list)
        binaried_data_list = data_list
        for i in range(batch_size):
            mean_data = np.mean(data_list[i])
            width = data_list[i].shape[0]
            for col in range(width):
                if data_list[i][col] < mean_data:
                    binaried_data_list[i][col] = 0
                else:
                    binaried_data_list[i][col] = 1
        return binaried_data_list


    def locate_scale(line_scales):
        """
        Find location of center of each scale

        Param：
            line_scales (list[np.array])：the list of binaried scales value

        Return：
            scale_locations (list[list])：location of each scale

        """
        batch_size = len(line_scales)
        scale_locations = list()
        for i in range(batch_size):
            line_scale = line_scales[i]
            width = line_scale.shape[0]
            find_start = False
            one_scale_start = 0
            one_scale_end = 0
            locations = list()
            for j in range(width - 1):
                if line_scale[j] > 0 and line_scale[j + 1] > 0:
                    if not find_start:
                        one_scale_start = j
                        find_start = True
                if find_start:
                    if line_scale[j] == 0 and line_scale[j + 1] == 0:
                        one_scale_end = j - 1
                        one_scale_location = (one_scale_start + one_scale_end) / 2
                        locations.append(one_scale_location)
                        one_scale_start = 0
                        one_scale_end = 0
                        find_start = False
            scale_locations.append(locations)
        return scale_locations


    def locate_pointer(line_pointers):
        """
        Find location of center of pointer

        Param：
            line_scales (list[np.array])：the list of binaried pointer value

        Return：
            scale_locations (list[list])：location of pointer

        """
        batch_size = len(line_pointers)
        pointer_locations = list()
        for i in range(batch_size):
            line_pointer = line_pointers[i]
            find_start = False
            pointer_start = 0
            pointer_end = 0
            location = 0
            width = line_pointer.shape[0]
            for j in range(width - 1):
                if line_pointer[j] > 0 and line_pointer[j + 1] > 0:
                    if not find_start:
                        pointer_start = j
                        find_start = True
                if find_start:
                    if line_pointer[j] == 0 and line_pointer[j + 1] == 0:
                        pointer_end = j - 1
                        location = (pointer_start + pointer_end) / 2
                        find_start = False
                        break
            pointer_locations.append(location)
        return pointer_locations


    def get_relative_location(scale_locations, pointer_locations):
        """
        Match location of pointer and scales

        Param：
            scale_locations (list[list])：location of each scale
            pointer_locations (list[list])：location of pointer

        Return：
            pointed_scales (list[dict])： a list of dict with:
                                        'num_scales': total number of scales
                                        'pointed_scale': predicted number of scales

        """
        pointed_scales = list()
        for scale_location, pointer_location in zip(scale_locations, pointer_locations):
            num_scales = len(scale_location)
            pointed_scale = -1
            if num_scales > 0:
                for i in range(num_scales - 1):
                    if scale_location[i] <= pointer_location < scale_location[i + 1]:
                        pointed_scale = i + (pointer_location - scale_location[i]) / (scale_location[i + 1] - scale_location[i] + 1e-05) + 1
            result = {"num_scales": num_scales, "pointed_scale": pointed_scale}
            pointed_scales.append(result)
        return pointed_scales


    def calculate_reading(pointed_scales):
        """
        Calculate the value of meter according to the type of meter

        Param：
            pointed_scales (list[list])：predicted number of scales

        Return：
            readings (list[float])： the list of values read from meter

        """
        readings = list()
        batch_size = len(pointed_scales)
        for i in range(batch_size):
            pointed_scale = pointed_scales[i]
            # find the type of meter according the total number of scales
            if pointed_scale["num_scales"] > TYPE_THRESHOLD:
                reading = pointed_scale["pointed_scale"] * METER_CONFIG[0]["scale_interval_value"]
            else:
                reading = pointed_scale["pointed_scale"] * METER_CONFIG[1]["scale_interval_value"]
            readings.append(reading)
        return readings


    # ## Main Function
    # [back to top ⬆️](#Table-of-contents:)
    # 

    # ### Initialize the model and parameters.
    # [back to top ⬆️](#Table-of-contents:)
    # 
    # 
    # select device from dropdown list for running inference using OpenVINO

    # In[13]:


    import ipywidgets as widgets

    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="AUTO",
        description="Device:",
        disabled=False,
    )

    device


    # The number of detected meter from detection network can be arbitrary in some scenarios, which means the batch size of segmentation network input is a [dynamic dimension](https://docs.openvino.ai/2024/openvino-workflow/running-inference/dynamic-shapes.html), and it should be specified as ```-1``` or the ```ov::Dimension()``` instead of a positive number used for static dimensions. In this case, for memory consumption optimization, we can specify the lower and/or upper bounds of input batch size.

    #img_file = f"{DATA_DIR}/{IMG_FILE_NAME}"
    det_model_path = f"{MODEL_DIR}/meter_det_model/model.pdmodel"
    det_model_shape = {
        "image": [1, 3, 608, 608],
        "im_shape": [1, 2],
        "scale_factor": [1, 2],
    }
    seg_model_path = f"{MODEL_DIR}/meter_seg_model/model.pdmodel"
    seg_model_shape = {"image": [ov.Dimension(1, 2), 3, 512, 512]}

    erode_kernel = 4
    score_threshold = 0.5
    seg_batch_size = 2
    input_shape = 608

    # Intialize the model objects
    detector = Model(det_model_path, det_model_shape, device.value)
    segmenter = Model(seg_model_path, seg_model_shape, device.value)


    # Load and visualize input image
    image = cv2.imread(str(IMG_PATH))
    #rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #plt.imshow(rgb_image)


    # ### Run meter detection model

    # Prepare the input data for meter detection model
    im_shape = np.array([[input_shape, input_shape]]).astype("float32")
    scale_factor = np.array([[1, 2]]).astype("float32")
    input_image = det_preprocess(image, input_shape)
    inputs_dict = {"image": input_image, "im_shape": im_shape, "scale_factor": scale_factor}

    # Run meter detection model
    det_results = detector.predict(inputs_dict)

    # Filter out the bounding box with low confidence
    filtered_results = filter_bboxes(det_results, score_threshold)

    # Prepare the input data for meter segmentation model
    scale_x = image.shape[1] / input_shape * 2
    scale_y = image.shape[0] / input_shape

    # Create the individual picture for each detected meter
    roi_imgs, loc = roi_crop(image, filtered_results, scale_x, scale_y)
    roi_imgs, resize_imgs = roi_process(roi_imgs, METER_SHAPE)

    # Create the pictures of detection results
    roi_stack = np.hstack(resize_imgs)

    if cv2.imwrite(f"{DATA_DIR}/detection_results.jpg", roi_stack):
        print('The detection result image has been saved as "detection_results.jpg" in data')
        plt.imshow(cv2.cvtColor(roi_stack, cv2.COLOR_BGR2RGB))
    print("datadir: "+DATA_DIR)

    # ### Run meter segmentation model
    
    seg_results = list()
    mask_list = list()
    num_imgs = len(roi_imgs)

    # Run meter segmentation model on all detected meters
    for i in range(0, num_imgs, seg_batch_size):
        batch = roi_imgs[i : min(num_imgs, i + seg_batch_size)]
        seg_result = segmenter.predict({"image": np.array(batch)})
        seg_results.extend(seg_result)
    results = []
    for i in range(len(seg_results)):
        results.append(np.argmax(seg_results[i], axis=0))
    seg_results = erode(results, erode_kernel)

    # Create the pictures of segmentation resultsdata:image/png;base64,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
    for i in range(len(seg_results)):
        mask_list.append(segmentation_map_to_image(seg_results[i], COLORMAP))
    mask_stack = np.hstack(mask_list)

    if cv2.imwrite(f"{DATA_DIR}/segmentation_results.jpg", cv2.cvtColor(mask_stack, cv2.COLOR_RGB2BGR)):
        print('The segmentation result image has been saved as "segmentation_results.jpg" in data')
        plt.imshow(mask_stack)


    # ### Postprocess the models result and calculate the final readings
   # Use OpenCV function to find the location of the pointer in a scale map.

    # In[18]:


    # Find the pointer location in scale map and calculate the meters reading
    rectangle_meters = circle_to_rectangle(seg_results)
    line_scales, line_pointers = rectangle_to_line(rectangle_meters)
    binaried_scales = mean_binarization(line_scales)
    binaried_pointers = mean_binarization(line_pointers)
    scale_locations = locate_scale(binaried_scales)
    pointer_locations = locate_pointer(binaried_pointers)
    pointed_scales = get_relative_location(scale_locations, pointer_locations)
    meter_readings = calculate_reading(pointed_scales)




    print(meter_readings)
    return meter_readings


def subir_imagen():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo en la solicitud'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        print(filepath)
        file.save(UPLOAD_FOLDER+"image.jpg")
        res= openvino
        return jsonify({'resultados': res, 'filepath': "image"}), 200

if __name__ == '__main__':
    app.run(debug=False)
