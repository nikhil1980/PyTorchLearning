#!/bin/python

"""
This module select key frames from a video that are
unique and truly represent the given video using advanced
clustering and image quality filtering.

 -- STAGE 1 CLUSTERING:
      Deploys 3 level pipeline with HDBSCAN clustering.
 -- STAGE 2 QUALITY:
      Quality control is done via BRISQUE and BLUR
 -- STAGE 3 SELECTION:
      Key Frame selection done on sorting Image Quality metric

"""

import av
import numpy as np
import torch
from transformers import utils, AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

import sys, os
sys.path.append('/raid/home/nikhilb/research/ExpansionNet_v2')

from argparse import Namespace
from ExpansionNet_v2.models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from ExpansionNet_v2.utils.image_utils import preprocess_image
from ExpansionNet_v2.utils.language_utils import tokens2description

import pickle

from strsimpy.cosine import Cosine
import operator

import secrets
import time
from math import ceil
import glob
import numpy as np
from shutil import rmtree, copyfile
from pathlib import Path
import cv2
import os
import extract_frames as ef
import cluster_frames as cf
from image_metrics import IQM, process_file
from key_frame_selection import KeyFrameSelector

from collections import defaultdict

utils.logging.set_verbosity_error()

""""""""""""""""" 
" Global Settings 
"""""""""""""""""
""" Config """
BASE_PATH = "/raid/home/nikhilb/research/thumbnail"  # Update base path accordingly
KFS = "/raid/home/nikhilb/research/thumbnail/kfs"
TOP_K = 3  # Number of target thumbnails


# Path to temporary files
APP_FS_TEMP = os.path.join(BASE_PATH, 'tmp')
APP_FS_OUTPUT = os.path.join(BASE_PATH, 'output_frame_selection')
APP_MODEL = os.path.join(os.path.join(KFS, 'model'), 'brisque')

""""""""""""""""""""""""""""""""""""""""""""""""
# Configuration parameters for HDBSCAN algorithm
""""""""""""""""""""""""""""""""""""""""""""""""


""" HDBSCAN """
BLUR_SCORE_THRESHOLD = 100.0
BRISQUE_SCORE_THRESHOLD = 40.0


def default_dict_with_keys_from_list(list1):
    return defaultdict(lambda: list1)


def read_file(path):
    with open(path, "r") as f:
        # Read the entire file
        file_contents = f.read()
        return file_contents


def compute_similarity(file_name):
    """
    Method to compute similarity metric of video
    caption w.r.t to each candidate thumbnail

    :param file_name: Name of the video file
    :type file_name: str

    :return: status of the function, dict containing name of final thumbnail and its score
    :rtype: tuple

    """
    path = os.path.join(os.path.realpath(__file__), 'DEMO')
    vid_base = os.path.join(os.path.join(path, Path(file_name).stem), 'thumbnail')
    thumb_files = [f for f in os.listdir(vid_base) if f.endswith('.png')]
    # print(thumb_files)

    cosine = Cosine(TOP_K + 1)

    if os.path.exists(os.path.join(path, (Path(file_name).stem + '.txt'))):
        print(f"Processing {file_name}")
        s0 = read_file(os.path.join(path, (Path(file_name).stem + '.txt')))
        p0 = cosine.get_profile(s0)
        sim_score_dict = {}

        print(os.path.join(vid_base, (Path(thumb_files[0]).stem + '.txt')))
        if os.path.exists(os.path.join(vid_base, (Path(thumb_files[0]).stem + '.txt'))):
            s1 = read_file(os.path.join(vid_base, (Path(thumb_files[0]).stem + '.txt')))
            p1 = cosine.get_profile(s1)
            sim_score_dict[thumb_files[0]] = round(cosine.similarity_profiles(p0, p1), 3)

        if os.path.exists(os.path.join(vid_base, (Path(thumb_files[1]).stem + '.txt'))):
            s2 = read_file(os.path.join(vid_base, (Path(thumb_files[1]).stem + '.txt')))
            p2 = cosine.get_profile(s2)
            sim_score_dict[thumb_files[1]] = round(cosine.similarity_profiles(p0, p2), 3)

        if os.path.exists(os.path.join(vid_base, (Path(thumb_files[1]).stem + '.txt'))):
            s3 = read_file(os.path.join(vid_base, (Path(thumb_files[2]).stem + '.txt')))
            p3 = cosine.get_profile(s3)
            sim_score_dict[thumb_files[2]] = round(cosine.similarity_profiles(p0, p3), 3)

        if len(sim_score_dict) != 0:
            sorted_dict = dict(sorted(sim_score_dict.items(), key=operator.itemgetter(1), reverse=True))
            print(f"Scores: {sorted_dict}")

            winner = next(iter(sorted_dict))
            status = True
        else:
            winner = {}
            status = False

        return status, winner


def caption_image(file_name):
    """
    Generate caption for a image file and write it in a predefined path

    :param file_name: name of video file
    :type file_name: str

    :return status : caption for each image in list, status  of the function
    :rtype: tuple

    """
    path = os.path.join(os.path.realpath(__file__), 'DEMO')
    vid_base = os.path.join(os.path.join(path, Path(file_name).stem), 'thumbnail')
    thumb_files = [f for f in os.listdir(vid_base) if f.endswith('.png')]

    args = {'description': 'Demo',
            'model_dim': 512,
            'N_enc': 3,
            'N_dec': 3,
            'max_seq_len': 74,
            'load_path': '/raid/home/nikhilb/research/ExpansionNet_v2/downloads/rf_model.pth',
            'image_paths': thumb_files,
            'beam_size': 5}

    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)

    model_args = Namespace(model_dim=args['model_dim'],
                           N_enc=args['N_enc'],
                           N_dec=args['N_dec'],
                           dropout=0.0,
                           drop_args=drop_args)

    with open('/raid/home/nikhilb/research/ExpansionNet_v2/demo_material/demo_coco_tokens.pickle', 'rb') as f:
        coco_tokens = pickle.load(f)
        sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
        eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

    print("Dictionary loaded ...")

    img_size = 384
    model = End_ExpansionNet_v2(swin_img_size=img_size,
                                swin_patch_size=4,
                                swin_in_chans=3,
                                swin_embed_dim=192,
                                swin_depths=[2, 2, 18, 2],
                                swin_num_heads=[6, 12, 24, 48],
                                swin_window_size=12,
                                swin_mlp_ratio=4.,
                                swin_qkv_bias=True,
                                swin_qk_scale=None,
                                swin_drop_rate=0.0,
                                swin_attn_drop_rate=0.0,
                                swin_drop_path_rate=0.0,
                                swin_norm_layer=torch.nn.LayerNorm,
                                swin_ape=False,
                                swin_patch_norm=True,
                                swin_use_checkpoint=False,
                                final_swin_dim=1536,
                                d_model=model_args.model_dim,
                                N_enc=model_args.N_enc,
                                N_dec=model_args.N_dec,
                                num_heads=8,
                                ff=2048,
                                num_exp_enc_list=[32, 64, 128, 256, 512],
                                num_exp_dec=16,
                                output_word2idx=coco_tokens['word2idx_dict'],
                                output_idx2word=coco_tokens['idx2word_list'],
                                max_seq_len=args['max_seq_len'],
                                drop_args=model_args.drop_args,
                                rank='cpu')

    checkpoint = torch.load(args['load_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded ...")

    input_images = []
    for path in args['image_paths']:
        input_images.append(preprocess_image(path, img_size))

    print("Generating captions ...\n")
    caption_list = []

    for i in range(len(input_images)):
        caption = {}
        path = args['image_paths'][i]
        image = input_images[i]
        beam_search_kwargs = {'beam_size': args['beam_size'],
                              'beam_max_seq_len': args['max_seq_len'],
                              'sample_or_max': 'max',
                              'how_many_outputs': 1,
                              'sos_idx': sos_idx,
                              'eos_idx': eos_idx}
        with torch.no_grad():
            pred, _ = model(enc_x=image,
                            enc_x_num_pads=[0],
                            mode='beam_search',
                            **beam_search_kwargs)

        pred = tokens2description(pred[0][0], coco_tokens['idx2word_list'], sos_idx, eos_idx)
        print(path + ' \n\tDescription: ' + pred + '\n')
        with open(os.path.join(vid_base, (Path(path).stem + '.txt')), 'w') as f:
            f.write(caption)

    print("---> End of Image caption run <---")
    status = True

    return status


def caption_video(file_name):
    """
    Generate caption for a video file and write it in a predefined path

    :param file_name: name of the file
    :type file_name: str

    :return status : status  of the function
    :rtype: bool

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load pretrained processor, tokenizer, and paligemma-weights
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

    # load video
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), file_name)
    container = av.open(file_path)

    # extract evenly spaced frames from video
    seg_len = container.streams.video[0].frames
    clip_len = model.config.encoder.num_frames
    indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))

    # generate caption
    gen_kwargs = {
        "min_length": 10,
        "max_length": 20,
        "num_beams": 8,
    }
    pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
    tokens = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    print(caption)

    # Write caption in DEMO folder
    input_path = os.path.join(os.path.join(os.path.realpath(__file__), 'DEMO'), Path(file_name).stem)

    with open(os.path.join(input_path, (Path(file_name).stem + '.txt')), 'w') as f:
        f.write(caption)

    status = True

    print("---> End of Video caption run <---")
    return status


def generate_candidate_thumbnail(file_name):
    """
    Generate Candidate Thumbnails

    :param file_name: name of the file
    :type file_name: str

    :return status : status  of the function
    :rtype: bool

    """
    """ Create object for Base Key Frame Selector """
    kfs = KeyFrameSelector(None, None)

    # Create the IQM object
    iqm = IQM()

    """ Set the file name in base class """
    kfs.file_name = file_name

    """ Create dataset path """
    input_path = os.path.join(os.path.realpath(__file__), 'DEMO')
    os.makedirs(input_path, exist_ok=True)

    hdbscancluster_fs = HDBSCANClusterFrameSelector(file_name, input_path, 'hdbscanCluster')

    start_time = time.time()
    _, frame_count = hdbscancluster_fs.extract_key_frames_hdbscancluster()
    total_time = time.time() - start_time

    print(f"Key Frame Selector method: {hdbscancluster_fs.method_name} working on file: "
          f"{hdbscancluster_fs.file_name} took {total_time:.3f} seconds to find {frame_count} frames")

    """ Final Selection using IQM """
    folder_path = os.path.join(os.path.join(input_path, Path(file_name).stem), 'kfs')

    final_list = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    # print(final_list)

    final_file_dict = default_dict_with_keys_from_list(final_list)
    print(final_file_dict)

    for file in final_list:
        iqm_score = process_file(iqm, os.path.join(folder_path, file))
        final_file_dict[file] = iqm_score

    final_file_dict = sorted(final_file_dict.items(), key=lambda kv: kv[1], reverse=True)
    # Temporary path for this file
    folder_thumbnail_path = os.path.join(os.path.join(input_path, Path(file_name).stem), 'thumbnail')

    # Create a fresh folder thumbnail
    os.makedirs(folder_thumbnail_path, exist_ok=True)

    counter = 0
    print(final_file_dict)
    for file in final_file_dict:
        # Write TOP_K files as Thumbnail
        print(f"Writing {file[0]}, {folder_path}, {folder_thumbnail_path}")
        new_file_name = Path(file[0]).stem + '_TOP_' + str(counter + 1) + Path(file[0]).suffix
        print(new_file_name)
        copyfile(os.path.join(folder_path, file[0]), os.path.join(folder_thumbnail_path, new_file_name))
        counter = counter + 1
        if counter == TOP_K:
            break

    """ Delete KFS folder as we got our thumbnails """
    folder_path = os.path.join(input_path, Path(file_name).stem)
    folder_path_new = os.path.join(folder_path, 'kfs')
    rmtree(folder_path_new)
    print(f"{folder_path_new} deleted successfully!!!")

    status = True
    return status


def main(file_name):
    """
    Main method to execute demo for Thumbnail creation

    :param file_name: name of the file
    :type file_name: str

    :return: final status of operation
      -- True for success
      -- False for failure

    :rtype: bool

    """
    """ 1. Generate candidate Thumbnails """
    status = generate_candidate_thumbnail(file_name)

    if status:
        """ 2. Generate video caption """
        status = caption_video(file_name)
        if status:
            """ 3. Generate thumbnail image caption """
            status = caption_image(file_name)
            if status:

                """ 4. Computer similarity caption """
                status, winner = compute_similarity(file_name)
                if status:
                    # Display the final thumbnail with score embedded in it
                    print()
                    status = True

                else:
                    print(f"\n-->Fail to display final thumbnail for video: {file_name}!!! <---\n")
                    status = False
    else:
        print(f"\n-->Fail to generate candidate thumbnails from video: {file_name}!!! <---\n")
        status = False

    return status


if __name__ == "__main__":
    print(f"\n--->Welcome to demo for Thumbnail/Poster Generator<---\n")
    file_name = str(input("\nEnter the name of the video file including extn:\n"))

    status = main(file_name)

    if status:
        print(f"\nSUCCESS\n")
        # show the final thumbnail
    else:
        print(f"\n---> FAILURE!!! <---")



