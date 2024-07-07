from pythainlp.corpus.common import thai_words
from pythainlp.tag import pos_tag_sents
from pythainlp.util import dict_trie
from pythainlp import Tokenizer
from typing import List
from tqdm import tqdm
import pandas as pd
import random as rd
import csv
import os


DEFAULT_SIMILARITY_SPLITS = [0.7, 0.1, 0.2]
DROP_THAI_CAPTIONS_WARNING_TEXT = "WARNING: total of {} thai captions dropped. (Image ID not found: {} | Image file not found: {})"
UNEQUAL_CAPTIONS_IMAGES_ERROR_TEXT = "Unequal number of captions and number of image file names. Found {} and {}."
TOO_MANY_CANDIDATES_ERROR_TEXT = "Number of candidates exceeds number of captions/images. ({} > {})"


THAI_TOKENIZER = Tokenizer(
    custom_dict=dict_trie(dict_source=set(thai_words())), 
    engine="newmm"
)


def create_similarity_candidates(captions: List[str], file_names: List[str], output_path: str, number_of_similar_choices: int=10, verbose: bool=True) -> pd.DataFrame:
    """_summary_

    Args:
        captions (List[str]): _description_
        file_names (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    
    if len(captions) != len(file_names):
        raise ValueError(UNEQUAL_CAPTIONS_IMAGES_ERROR_TEXT.format(len(captions), len(file_names)))
    
    if number_of_similar_choices >= len(captions):
        raise ValueError(TOO_MANY_CANDIDATES_ERROR_TEXT.format(number_of_similar_choices, len(captions)))
    
    def iou(text1, text2):
        
        total_words = text1.union(text2)
        
        if not len(total_words):
            return 0
        
        return len(text1.intersection(text2)) / len(total_words)
    
    # stored_similarity = {}
    
    captions = [caption.replace(",", "") for caption in captions]
    
    sents = [THAI_TOKENIZER.word_tokenize(caption) for caption in captions]
    sents_with_pos = pos_tag_sents(sents, engine="perceptron", corpus="orchid")
    all_sent_words = [set(words) for words in sents]
    all_noun_sent_words = [set([word[0] for word in words if word[1].startswith("N")]) for words in sents_with_pos]
    
    header_names = ["correct_caption", "correct_image"]
    header_names += [f"similar_caption_{num}" for num in range(1, number_of_similar_choices + 1)]
    header_names += [f"similar_image_{num}" for num in range(1, number_of_similar_choices + 1)]
    
    with open(output_path, "w") as csv_file:
        csv_file.writelines(",".join(header_names) + "\n")
        
    for target_index, target_caption in enumerate(tqdm(captions)):
        
        target_all_words = all_sent_words[target_index]
        target_all_noun_words = all_noun_sent_words[target_index]
        
        top_comparisons = []
        for comparison_index, comparison_caption in enumerate(captions):
            
            if comparison_index == target_index:
                continue
            
            if file_names[target_index] == file_names[comparison_index]:
                similarity = (-1, -1)
                # stored_similarity[(target_index, comparison_index)] = similarity
            # elif (comparison_index, target_index) in stored_similarity:
            #     similarity = stored_similarity[(comparison_index, target_index)]
            else:
                comparison_all_words = all_sent_words[comparison_index]
                comparison_all_noun_words = all_noun_sent_words[comparison_index]
                all_iou = iou(target_all_words, comparison_all_words)
                noun_iou = iou(target_all_noun_words, comparison_all_noun_words)
                similarity = (noun_iou, all_iou)
                # stored_similarity[(target_index, comparison_index)] = similarity
                
            if len(top_comparisons) < number_of_similar_choices:
                top_comparisons.append((comparison_caption, file_names[comparison_index], similarity))
            elif similarity > top_comparisons[-1][2]:
                top_comparisons = sorted(
                    top_comparisons + [(comparison_caption, file_names[comparison_index], similarity)], 
                    key=lambda comparision: comparision[2],
                    reverse=True
                )
                top_comparisons = top_comparisons[: number_of_similar_choices]
                
        row = [target_caption, file_names[target_index]]
        row += [comparison[0] for comparison in top_comparisons]
        row += [comparison[1] for comparison in top_comparisons]
            
        with open(output_path, "a") as csv_file:
            csv_file.writelines(",".join(row) + "\n")
    
    
def create_similarity_splits(
        annotations: dict, 
        image_dir_path: str, 
        output_paths: List[str],
        total_images: int=None,
        splits: List[float]=None,
        number_of_similar_choices: int=10
    ) -> List[pd.DataFrame]:
    """_summary_

    Args:
        annotations (dict): _description_
        image_dir_path (str): _description_
        output_path (List[str]): _description_
        total_images (int, optional): _description_. Defaults to None.
        splits (List[float], optional): _description_. Defaults to None.

    Returns:
        List[pd.DataFrame]: _description_
    """
    
    image_id_to_file_name = {}
    for image_data in annotations["images"]:
        
        image_id, file_name = int(image_data["id"]), image_data["file_name"]
        
        image_id_to_file_name[image_id] = file_name
        
    n_dropped_captions_missing_image_ids = 0
    n_dropped_captions_missing_image_files = 0
        
    image_file_name_to_captions = {}
    for caption_data in annotations["annotations"]:
        
        image_id, caption = int(caption_data["image_id"]), caption_data["caption_thai"]
        
        if image_id not in image_id_to_file_name:
            n_dropped_captions_missing_image_ids += 1
            continue
        
        file_name = image_id_to_file_name[image_id]
        
        if not os.path.exists(os.path.join(image_dir_path, file_name)):
            n_dropped_captions_missing_image_files += 1
            continue
        
        if file_name in image_file_name_to_captions:
            image_file_name_to_captions[file_name].append(caption)
        else:
            image_file_name_to_captions[file_name] = [caption]      
            
    if n_dropped_captions_missing_image_ids or n_dropped_captions_missing_image_files:
        print(DROP_THAI_CAPTIONS_WARNING_TEXT.format(
            n_dropped_captions_missing_image_ids + n_dropped_captions_missing_image_files,
            n_dropped_captions_missing_image_ids,
            n_dropped_captions_missing_image_files
        ))
        
    if total_images is not None:
        image_file_name_to_captions = {
            file_name: captions 
            for file_name, captions in rd.sample(image_file_name_to_captions.items(), total_images)
        }
        
    image_file_names = list(image_file_name_to_captions.keys())
    rd.shuffle(image_file_names)
    
    sum_splits = sum(splits)
    splits = [split_size / sum_splits for split_size in splits]
    
    no_split = splits is None
    if no_split:
        splits = [1]
    
    split_start = 0
    for split_index, (split_size, output_path) in enumerate(zip(splits, output_paths)):
        
        if split_index == len(splits) - 1:
            split_image_file_names = image_file_names[split_start :]
        else:
            split_end = split_start + round(split_size * len(image_file_names))
            split_image_file_names = image_file_names[split_start : split_end]
            
        full_file_names = []
        full_captions = []
        for file_name in split_image_file_names:
            captions = image_file_name_to_captions[file_name]
            for caption in captions:
                full_file_names.append(file_name)
                full_captions.append(caption)
                
        create_similarity_candidates(full_captions, full_file_names, output_path, number_of_similar_choices)       
        
        split_start = split_end     