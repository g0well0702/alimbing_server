from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import logging
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import torch

app = Flask(__name__)
CORS(app)

# 로그 설정
logging.basicConfig(level=logging.DEBUG)

# 모델 설정 및 로딩
def setup_model():
    cfg = get_cfg()
    cfg.merge_from_file("experiment_config.yml")
    cfg.MODEL.WEIGHTS = "model_final.pth"
    cfg.MODEL.DEVICE = 'cpu'
    MetadataCatalog.get("meta").thing_classes = ["hold", "volume"]
    metadata = MetadataCatalog.get("meta")
    return DefaultPredictor(cfg), metadata

predictor, metadata = setup_model()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        logging.error("No image file in request")
        return "No image file", 400

    file = request.files['image']
    if file.filename == '':
        logging.error("No selected file")
        return "No selected file", 400

    try:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        logging.debug("Image loaded successfully")

        outputs = predictor(img)
        logging.debug("Prediction completed")

        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes
        pred_masks = instances.pred_masks
        pred_scores = instances.scores

        hold_class_idx = metadata.thing_classes.index("hold")

        filtered_masks = pred_masks[(pred_classes == hold_class_idx) & (pred_scores > 0.9)]

        if filtered_masks.size(0) > 0:
            for i in range(filtered_masks.size(0)):
                mask = filtered_masks[i].numpy().astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (0, 0, 255), 2)  # Red color for contours

        # 결과 이미지 저장
        result_path = "segmented_result.jpg"
        cv2.imwrite(result_path, img)
        logging.debug("Segmented result image saved successfully")

        return send_file(result_path, mimetype='image/jpeg')
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
