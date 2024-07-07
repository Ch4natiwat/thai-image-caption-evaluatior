from pathlib import Path
import argparse
import json
import sys
import os

sys.path.append(os.path.abspath(Path(__file__).parents[1]))

from thai_image_caption_subtle_discriminator.dataset.create_similarity_split import create_similarity_splits


def get_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--annotation_json_path", 
        type=str, 
        required=True, 
        help="A path to annotation JSON file"
    )
    parser.add_argument(
        "--image_dir_path", 
        type=str, 
        required=True, 
        help="A path to image directory"
    )
    parser.add_argument(
        "--total_images", 
        type=int, 
        default=None, 
        help="An optional float argument with a default value"
    )
    parser.add_argument(
        "--splits", 
        type=str, 
        default=None, 
        help="An optional float argument with a default value"
    )
    parser.add_argument(
        "--number_of_similar_choices",
        type=int,
        default=20,
        help=""
    )
    parser.add_argument(
        "--output_csv_names",
        type=str,
        default=None,
        help=""
    )
    
    return parser


def main(args=None):
    
    parser = get_parser()
    args = parser.parse_args(args)
    
    annotation_json_path = args.annotation_json_path
    with open(annotation_json_path, "r") as annotation_file:
        annotations = json.load(annotation_file)
    image_dir_path = args.image_dir_path
        
    total_images = args.total_images
    splits = args.splits
    output_csv_names = args.output_csv_names
    
    if splits is None:
        splits = [1.]
    else:
        splits = [float(split_size) for split_size in splits.split(",")]
        
    if output_csv_names is None:
        output_csv_names = []
    else:
        output_csv_names = [output_name for output_name in output_csv_names.split(",")]
        
    output_csv_names.extend([f"unnamed_{index}.csv" for index in range(len(splits))])
    output_csv_names = output_csv_names[: len(splits)]
    
    number_of_similar_choices = args.number_of_similar_choices
    
    output_dir = os.path.dirname(annotation_json_path)
    output_paths = [os.path.join(output_dir, output_name) for output_name in output_csv_names]
    create_similarity_splits(annotations, image_dir_path, output_paths, total_images, splits, number_of_similar_choices)   
    
    
if __name__ == "__main__":
    
    main()