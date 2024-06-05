from pythainlp.corpus.common import thai_words
from pythainlp.tag import pos_tag_sents
from pythainlp.util import dict_trie
from pythainlp import Tokenizer
from typing import List
from tqdm import tqdm
import pandas as pd
import random as rd
import os


DEFAULT_SIMILARITY_SPLITS = [0.7, 0.1, 0.2]
DROP_THAI_CAPTIONS_WARNING_TEXT = "WARNING: total of {} thai captions dropped. (Image ID not found: {} | Image file not found: {})"
UNEQUAL_CAPTIONS_IMAGES_ERROR_TEXT = "Unequal number of captions and number of image file names. Found {} and {}."
TOO_MANY_CANDIDATES_ERROR_TEXT = "Number of candidates exceeds number of captions/images. ({} > {})"


THAI_TOKENIZER = Tokenizer(
    custom_dict=dict_trie(dict_source=set(thai_words())), 
    engine="newmm"
)


def create_similarity_candidates(captions: List[str], file_names: List[str], number_of_candidates: int=10, verbose: bool=True) -> pd.DataFrame:
    """_summary_

    Args:
        captions (List[str]): _description_
        file_names (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    
    if len(captions) != len(file_names):
        raise ValueError(UNEQUAL_CAPTIONS_IMAGES_ERROR_TEXT.format(len(captions), len(file_names)))
    
    if number_of_candidates - 1 > len(captions):
        raise ValueError(TOO_MANY_CANDIDATES_ERROR_TEXT.format(number_of_candidates, len(captions)))
    
    def iou(text1, text2):
        
        total_words = text1.union(text2)
        
        if not len(total_words):
            return 0
        
        return len(text1.intersection(text2)) / len(total_words)
    
    
    words = [THAI_TOKENIZER.word_tokenize(caption) for caption in captions]
    words_with_pos = pos_tag_sents(words, engine="perceptron", corpus="orchid")

    processed_texts = [
        (set(sent), set([s[0] for s in sent_with_pos if s[1].startswith("N")]))
        for sent, sent_with_pos in zip(words, words_with_pos)
    ]
    
    similarity_summary = {}
    for i in range(len(processed_texts) - 1):
        for j in range(i + 1, len(processed_texts)):
            if file_names[i] == file_names[j]:
                similarity = (-1, -1)
                # similarity = -1
            else:
                text_1 = processed_texts[i]
                text_2 = processed_texts[j]
                all_iou = iou(text_1[0], text_2[0])
                noun_iou = iou(text_1[1], text_2[1])
                similarity = (noun_iou, all_iou)
                # similarity = all_iou
            similarity_summary[(i, j)] = similarity
            
    rows = [] 
        
    caption_indices = range(len(captions))
    if verbose:
        caption_indices = tqdm(caption_indices)
        
    for caption_index in caption_indices:
        similarities = [(key, similarity) for key, similarity in similarity_summary.items() if caption_index in key]
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[: number_of_candidates - 1]
        
        similar_captions = []
        similar_images = []
        for key, similarity in similarities:
            key = [index for index in key if index != caption_index][0]
            caption = captions[key]
            image = file_names[key]
            similar_captions.append(caption)
            similar_images.append(image)
            
        correct_caption = captions[caption_index]
        correct_image = file_names[caption_index]
        
        row = {
            "correct_caption": correct_caption,
            "correct_image": correct_image
        }
        
        for num, caption in enumerate(similar_captions, 1):
            row.update({f"similar_caption_{num}": caption})
        for num, image in enumerate(similar_images, 1):
            row.update({f"similar_image_{num}": image})
        rows.append(row)
        
    similarity_candidates_df = pd.DataFrame.from_dict(rows)
    
    return similarity_candidates_df
    
    
def create_similarity_splits(
        annotations: dict, 
        image_dir_path: str, 
        total_images: int=None,
        splits: List[float]=None,
        number_of_candidates: int=10
    ) -> List[pd.DataFrame]:
    """_summary_

    Args:
        annotations (dict): _description_
        image_dir_path (str): _description_
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
        
    full_similarity_candidates = []
    
    split_start = 0
    for split_index, split_size in enumerate(splits):
        
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
                
        similarity_candidates = create_similarity_candidates(full_captions, full_file_names, number_of_candidates)       
        full_similarity_candidates.append(similarity_candidates)
        
        split_start = split_end
        
    if no_split:
        return full_similarity_candidates[0]
    
    return full_similarity_candidates           