# Load YOLOv8n-seg, train it on COCO128-seg for 3 epochs and predict an image with it
from ultralytics import YOLO
import pdb
import numpy as np

model = YOLO('yolov8n-seg.pt')  # load a pretrained YOLOv8n segmentation model
# model.train(data='coco128-seg.yaml', epochs=3)  # train the model
# model('https://ultralytics.com/images/bus.jpg',save=True)  # predict on an image
# model('rgb_input0_0000000005.png',save=True,save_txt=True)  # predict on an image

results = model('zidane.jpg',save=False,save_txt=False)  # predict on an image

result = results[0]
# print(result)
# print(result.masks)
# print(result.boxes)
# print(result.names)
# # Print out the bounding boxes
# for box in result.boxes:
#     print(box.xyxy)  # This should print the bounding box coordinates in the [x_min, y_min, x_max, y_max] format
# Print out the bounding boxes

for i in result.boxes:
    print(i.cls)
    pdb.set_trace()
    if i.cls.cpu().item() == 0:
        print(f'人間の範囲は{i.xyxy}だよ')
    # if i.cls == 0:
    #     print(i.cls)  # This should print the bounding box coordinates in the [x_min, y_min, x_max, y_max] format
        # print("data.shpae:",i.data.shape)
        # print("xy:",np.shape(i.xy))
    # print(mask.segments)  # This should print the bounding box coordinates in the [x_min, y_min, x_max, y_max] format
# pdb.set_trace()

#画像のxyxy範囲を囲って出力する
import cv2
import numpy as np




# import numpy as np
# import cv2

# def get_depth_within_coordinates(depth_map, coordinates):
#     """
#     Get the depth information within the specified coordinates.

#     Parameters:
#     - depth_map: A 2D numpy array containing depth information.
#     - coordinates: A list of tuples representing the coordinates (x1, y1, ..., x6, y6).

#     Returns:
#     - A numpy array containing the depth values within the specified coordinates.
#     """
#     # Assuming coordinates are [(x1, y1), (x2, y2), ..., (x6, y6)]
#     # Convert coordinates to a numpy array
#     points = np.array(coordinates, dtype=np.int32)

#     # Create a mask where the polygon defined by the coordinates is filled
#     mask = np.zeros_like(depth_map, dtype=np.uint8)
#     cv2.fillPoly(mask, [points], 255)

#     # Use the mask to select the depth information
#     depth_within_coordinates = cv2.bitwise_and(depth_map, depth_map, mask=mask)

#     return depth_within_coordinates

# # 仮定の深度マップを読み込む（実際にはセンサーから取得するか、深度推定アルゴリズムを使用）
# depth_map = cv2.imread('path_to_depth_map.png', cv2.IMREAD_UNCHANGED)

# # セグメンテーションから得られた車の座標
# # 例: [(x1, y1), (x2, y2), ..., (x6, y6)]
# car_coordinates = [(100, 200), (150, 200), (150, 250), (100, 250), (75, 225), (125, 225)]

# # 座標内の深度情報を取得
# depth_info = get_depth_within_coordinates(depth_map, car_coordinates)

# # 深度情報を使用する（例：平均深度を計算）
# average_depth = np.mean(depth_info[depth_info > 0])  # 0より大きい値のみを考慮
# print(f"The average depth within the car coordinates is: {average_depth}")