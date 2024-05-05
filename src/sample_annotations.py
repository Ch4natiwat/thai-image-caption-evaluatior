from pathlib import Path
from glob import glob
from tqdm import tqdm
import random as rd
from attacut import Tokenizer
import shutil
import json
import os


ANNOTATIONS_PATH = os.path.join(os.path.dirname(__file__), "data", "annotations")

ORIGINAL_SET = "full_train"
SAMPLE_SIZE = 5000
TARGET_SET = f"sample_{SAMPLE_SIZE}_train"

ORIGINAL_SET_PATH = os.path.join(ANNOTATIONS_PATH, ORIGINAL_SET)
TARGET_SET_PATH = os.path.join(ANNOTATIONS_PATH, TARGET_SET)
ORIGINAL_SET_IMAGES_PATH = os.path.join(ORIGINAL_SET_PATH, "images")
TARGET_SET_IMAGES_PATH = os.path.join(TARGET_SET_PATH, "images")

ORIGINAL_SET_JSON_PATH = glob(os.path.join(ORIGINAL_SET_PATH, "*.json"))[0]

atta = Tokenizer(model="attacut-sc")


Path(TARGET_SET_PATH).mkdir(parents=True, exist_ok=True)
for item in os.listdir(TARGET_SET_PATH):
    item_path = os.path.join(TARGET_SET_PATH, item)
    try:
        os.remove(item_path)
    except IsADirectoryError:
        shutil.rmtree(item_path)
Path(TARGET_SET_IMAGES_PATH).mkdir()


with open(ORIGINAL_SET_JSON_PATH, "r") as annotation_json_file:
    annotation_json = json.load(annotation_json_file)

images = annotation_json["images"]
rd.shuffle(images)

annotations = annotation_json["annotations"]
annotations_map = {}
for annotation in annotations:
    image_id = int(annotation["image_id"])
    if image_id in annotations_map:
        annotations_map[image_id].append(annotation)
    else:
        annotations_map[image_id] = [annotation]  

sampled_images = []
sampled_annotations = []

count = 0

with tqdm(total=SAMPLE_SIZE) as t:
    
    for image in images:
        
        image_id = image["id"]
        
        if image_id not in annotations_map:
            continue
        
        image_annotations = annotations_map[image_id]    
        annotation = max(image_annotations, key=lambda x: len(atta.tokenize(x["caption_thai"])))
        
        file_name = image["file_name"]
        image_path = os.path.join(ORIGINAL_SET_IMAGES_PATH, file_name)
        new_image_path = os.path.join(TARGET_SET_IMAGES_PATH, file_name)
        shutil.copyfile(image_path, new_image_path)
        
        sampled_images.append(image)
        sampled_annotations.append(annotation)
        
        t.update()
            
        if len(sampled_images) >= SAMPLE_SIZE:
            break
    
# print(annotation_json.keys())
# print(annotation_json["licenses"])
    
new_annotation_json = {
    "info": {
        "description": "COCO 2017 Sampled Dataset", 
        "url": "", 
        "version": "1.0", 
        "year": 2024, 
        "contributor": "COCO Consortium & Chanatiwat", 
        "date_created": "2024/05/05"
    },
    "licenses": annotation_json["licenses"],
    "images": sampled_images,
    "annotations": sampled_annotations
}

with open(os.path.join(TARGET_SET_PATH, "annotations.json"), "w") as json_file:
    json.dump(new_annotation_json, json_file, ensure_ascii=False)