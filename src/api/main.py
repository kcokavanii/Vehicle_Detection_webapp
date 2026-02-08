from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import time
from typing import Optional, List, Dict, Any
from ultralytics import YOLO
from PIL import Image
import io
import os
import base64

app = FastAPI(title="Vehicle Detection API")


MODEL_PATHS = {
    "yolov8n": "../models/yolov8nano.pt",
    "yolov8m": "../models/yolov8m.pt",
}

models_dict = {}

@app.on_event("startup")
async def load_models():
    try:
        models_dict["yolov8n"] = YOLO(MODEL_PATHS["yolov8n"])
        models_dict["yolov8m"] = YOLO(MODEL_PATHS["yolov8m"])
        
    except Exception as e:
        print(f"Ошибка при загрузке моделей: {e}")
        # Используем предобученные модели по умолчанию
        models_dict["yolov8n"] = YOLO("yolov8n.pt")
        models_dict["yolov8m"] = YOLO("yolov8m.pt")


# Класс для детекции автотранспорта
class VehicleDetector:
    VEHICLE_CLASSES = {
        0: "car",
        1: "van",
        2: "truck",
        3: "tricycle",
        4: "awning-tricycle",
        5: "bus",
        6: "motor"
    }    
    def __init__(self, model_name: str = "yolov8n"):
        self.model_name = model_name
        self.model = models_dict.get(model_name)
        
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.25) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError(f"Модель {self.model_name} не найдена")
        
        start_time = time.time()
        results = self.model.predict(image, conf=confidence_threshold, verbose=False)        
        processing_time = time.time() - start_time
        
        vehicles = []
        total_vehicles = 0        
        for box in results[0].boxes.cpu().numpy():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = list(self.VEHICLE_CLASSES.values())[class_id]            
            total_vehicles += 1
            
            vehicles.append({
                "class": class_name,
                "class_id": class_id,
                "confidence": confidence,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "bbox_normalized": [
                    float(x1 / image.shape[1]),
                    float(y1 / image.shape[0]),
                    float(x2 / image.shape[1]),
                    float(y2 / image.shape[0])
                ]
            })
        
        # Группировка по типам транспортных средств
        vehicle_counts = {}
        for vehicle in vehicles:
            vehicle_type = vehicle["class"]
            vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1
        
        return {
            "model": self.model_name,
            "total_vehicles": total_vehicles,
            "vehicle_counts": vehicle_counts,
            "vehicles": vehicles,
            "processing_time": round(processing_time, 3),
            "image_size": {
                "width": image.shape[1],
                "height": image.shape[0],
                "channels": image.shape[2] if len(image.shape) > 2 else 1
            }
        }


# Маршрут для проверки работы
@app.get("/")
async def read_root():
    return {
        "message": "Vehicle Detection API работает!",
        "available_models": list(MODEL_PATHS.keys()),
        "model_descriptions": {
            "yolov8n": "Быстрая модель (nano version)",
            "yolov8m": "Точная модель (medium version)"
        },
        "endpoints": {
            "root": "/",
            "health_check": "/health",
            "detect": "/detect/",
            "detect_with_params": "/detect/?model_name=yolov8n&conf_threshold=0.25"
        }
    }

# Маршрут для проверки здоровья сервера и моделей (что все работает правильно)
@app.get("/health")
async def health_check():
    status = {
        "status": "healthy",
        "models_loaded": list(models_dict.keys()),
        "timestamp": time.time()
    }    
    for model_name, model in models_dict.items():
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            _ = model.predict(test_img, verbose=False)
            status[f"{model_name}_status"] = "working"
        except:
            status[f"{model_name}_status"] = "error"
    return JSONResponse(content=status)

# Маршрут для детекции объектов
@app.post("/detect/")
async def detect_vehicles(
    file: UploadFile = File(...),
    model_name: str = "yolov8n",
    conf_threshold: float = 0.25
):
    try:
        # Проверка поддерживаемой модели
        if model_name not in models_dict:
            available_models = list(models_dict.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Модель '{model_name}' не найдена. Доступные модели: {available_models}"
            )
        
        # Проверка типа файла
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Файл должен быть изображением"
            )
        
        contents = await file.read()
        
        try:
            image = Image.open(io.BytesIO(contents))
            image = image.convert('RGB')
            img_array = np.array(image)            
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except:
            nparr = np.frombuffer(contents, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_array is None:
            raise HTTPException(
                status_code=400,
                detail="Не удалось прочитать изображение"
            )
        
        # Создаем детектор и выполняем детекцию
        detector = VehicleDetector(model_name)
        results = detector.detect(img_array, confidence_threshold=conf_threshold)
        
        results["filename"] = file.filename
        results["file_size"] = len(contents)
        results["conf_threshold"] = conf_threshold
        
        
        # Рисуем bounding boxes на изображении
        output_image = img_array.copy()        
        for vehicle in results["vehicles"]:
            x1, y1, x2, y2 = vehicle["bbox"]
            label = f"{vehicle['class']}: {vehicle['confidence']:.2f}"

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(output_image, label, (x1, y1-10), 
                        font, 0.5, (0, 255, 0), 2)        
        _, buffer = cv2.imencode('.jpg', output_image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        results["processed_image"] = f"data:image/jpeg;base64,{img_str}"        
        return JSONResponse(content=results)        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке изображения: {str(e)}"
        )
    


# Маршрут для сравнения моделей
@app.post("/compare/")
async def compare_models(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_array is None:
            raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")
        
        results = {}
        
        # Тестируем каждую модель
        for model_name in ["yolov8n", "yolov8m"]:
            start_time = time.time()            
            detector = VehicleDetector(model_name)
            model_results = detector.detect(img_array, confidence_threshold=conf_threshold)
            
            processing_time = time.time() - start_time
            model_results["total_processing_time"] = round(processing_time, 3)
            
            # Рисуем bounding boxes на изображении
            output_image = img_array.copy()
            
            for vehicle in model_results["vehicles"]:
                x1, y1, x2, y2 = vehicle["bbox"]
                label = f"{vehicle['class']}: {vehicle['confidence']:.2f}"

                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(output_image, label, (x1, y1-10), 
                            font, 0.5, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', output_image)
            img_str = base64.b64encode(buffer).decode('utf-8')
            model_results["processed_image"] = f"data:image/jpeg;base64,{img_str}"            
            results[model_name] = model_results
        
        # Сравниваем результаты
        comparison = {
            "filename": file.filename,
            "image_size": f"{img_array.shape[1]}x{img_array.shape[0]}",
            "conf_threshold": conf_threshold,
            "results": results,
            "comparison": {
                "speed_ratio": round(
                    results["yolov8m"]["total_processing_time"] / 
                    results["yolov8n"]["total_processing_time"], 
                    2
                ),
                "count_difference": 
                    results["yolov8m"]["total_vehicles"] - 
                    results["yolov8n"]["total_vehicles"]
            }
        }        
        return JSONResponse(content=comparison)        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при сравнении моделей: {str(e)}"
        )

# Маршрут для теста без загрузки файлов
@app.get("/test/")
async def test_detection():
    return {
        "status": "success",
        "example_request": {
            "endpoint": "POST /detect/",
            "parameters": {
                "model_name": "yolov8n или yolov8m",
                "conf_threshold": "0.25",
                "file": "изображение в формате JPEG/PNG"
            }
        },
        "example_response": {
            "model": "yolov8n",
            "total_vehicles": 3,
            "vehicle_counts": {"car": 2, "truck": 1},
            "processing_time": 0.15,
            "vehicles": [
                {
                    "class": "car",
                    "confidence": 0.85,
                    "bbox": [100, 150, 300, 400]
                }
            ]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)