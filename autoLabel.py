from ultralytics import YOLO
import cv2
import os

model_path = "weights/vehicles&road_block_yolov8/best.pt"
image_path = "pole柱狀"

# Load the model
model = YOLO(model_path)

# def coordinatesNormalize(coordinates):


def find_files(directory, subTitle=""):
    returnFiles = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(subTitle):
                returnFiles.append(os.path.join(root, file))
                print(file)
    return returnFiles


def autoLabel(imagePath):
    # Load an image
    img = cv2.imread(imagePath)

    # Predict
    results = model.predict(img)

    # create label file
    labelFile = os.path.splitext(imagePath)[0] + ".txt"
    print(labelFile)

    # Open the output file
    with open(labelFile, "w") as f:
        # Get bounding box coordinates
        for r in results:
            boxes = r.boxes
            for box in boxes:
                coordinate = box.xyxy.tolist()[0]
                # Convert to YOLO format
                # print(coordinate)

                x1 = coordinate[0]
                y1 = coordinate[1]
                x2 = coordinate[2]
                y2 = coordinate[3]

                # width = abs(x2 - x1)
                # height = abs(y2 - y1)
                # x_center = abs(x1 + width / 2)
                # y_center = abs(y1 + height / 2)

                # Normalize the coordinates
                # img_height, img_width = img.shape[:2]
                # x_center /= img_width
                # y_center /= img_height
                # width /= img_width
                # height /= img_height

                # Get the class id and confidence value
                class_id = int(box.cls.item())
                conf = box.conf.item()

                # write to disk
                # f.write(f"{r.names[class_id]} {x_center} {y_center} {width} {height}\n")
                # print(f"label: {r.names[class_id]} {x_center} {y_center} {width} {height}\n")
                f.write(f"{r.names[class_id]} {x1} {y1} {x2} {y2}\n")
                print(f"label: {r.names[class_id]} {x1} {y1} {x2} {y2}\n")

            resultName=os.path.splitext(imagePath)[0]+"_result"+".jpg"
            print(resultName)
            r.save(resultName)


if __name__ == "__main__":
    images = find_files(image_path, "jpg")
    for image in images:
        autoLabel(image)
    # autoLabel('frames/MOV_0012/MOV_0012_frame_0390.jpg')
