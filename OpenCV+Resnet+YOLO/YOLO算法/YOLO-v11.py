from ultralytics import YOLO

# Load a model
model = YOLO(r"C:\Users\86135\Desktop\AIGC\第三讲\YOLO算法\runs\detect\train8\weights\best.pt")  # load an official model
#model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model

while True:
    results = model.predict(source=0,show=True)  # predict on an image
    # Access the results
    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-riht-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box
