import csv
import os
import pandas as pd
import spacy
import re
import string
from models import sort_videos, NUM_VIDEOS, DIRPATH
from common import CSV_FILEPATH
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load('en_core_web_lg')


def compare_texts():
	with open(CSV_FILEPATH) as f:
		reader = csv.DictReader(f)
		videos = {}
		for row in reader:
		    hearing_date = row["Hearing date"]

		    if hearing_date[-2:] == "18":  # only get videos from 2018 hearing
		        video_link = row["Link to video"]
		        video_id = os.path.basename(video_link).split(".mp4")[0]
		        full_video_link = video_link.split("#t=")[0]
		        if video_id not in videos:
		            start_time = int(row["Start"])
		            videos[video_id] = {"video_link": full_video_link, "start_time": start_time}
		        end_time = int(row["End"])
		        videos[video_id]["end_time"] = end_time

		videos = sort_videos(videos)
		video_count = 0

		for info in videos:  # change me
		    if video_count <= NUM_VIDEOS:
		        vid_id = info["video_id"]
		        run_metrics(vid_id)
		    video_count += 1

def run_metrics(vid_id: str):

	groundtruth_file = f"{DIRPATH}/groundtruth/{vid_id}_truth.txt"
	assemblyai_file = f"{DIRPATH}/assembly/{vid_id}_assembly.txt"
	deepgram_file = f"{DIRPATH}/deepgram/{vid_id}_deepgram.txt"
	
	with open(groundtruth_file, "r") as f:
		groundtruth = f.read().replace('\n', '')

	with open(assemblyai_file, "r") as f:
		assemblyai = f.read().replace('\n', '')

	with open(deepgram_file, "r") as f:
		deepgram = f.read().replace('\n', '')

	# breakpoint()
	metrics(vid_id, groundtruth, assemblyai, deepgram)

def num_words(text: str, model: str):
	num_words = len(text.split(' '))
	print(f"   {model}: ", num_words)
	return num_words

def metrics(video_id: str, truth: str, assemblyai: str, deepgram: str):

	print(f"Video ID: {video_id}")
	print("Num words:")
	n1 = num_words(truth, "groundtruth")
	n2 = num_words(assemblyai, "assemblyai")
	n3 = num_words(deepgram, "deepgram")

	print("Cosine similarity:")
	c1, c2 = get_cosine_similarity(truth, assemblyai, "groundtruth and assemblyai")
	c3, c4 = get_cosine_similarity(truth, deepgram, "groundtruth and deepgram")
	c5, c6 = get_cosine_similarity(assemblyai, deepgram, "assemblyai and deepgram")

	print(n1)
	print(n2)
	print(n3)
	print(c1)
	print(c2)
	print(c3)
	print(c4)
	print(c5)
	print(c6)
	print("-----------------------------")
	get_names(truth)
	get_names(assemblyai)
	get_names(deepgram)

	get_orgs(truth)
	get_orgs(assemblyai)
	get_orgs(deepgram)

def get_names(text):
	spacy_parser = nlp(text)
	for entity in spacy_parser.ents:
		if entity.label_ == 'PERSON':
			print(entity.text)

def get_orgs(text):
	spacy_parser = nlp(text)
	for entity in spacy_parser.ents:
		if entity.label_ == 'ORG':
			print(entity.text)


def compute_cosine_similarity(text1, text2):
	"""
	Uses TFIDF vectorizere and cosine sim from sklearn
	"""
    # stores text in a list
    list_text = [text1, text2]
    
    # converts text into vectors with the TF-IDF 
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit_transform(list_text)
    tfidf_text1, tfidf_text2 = vectorizer.transform([list_text[0]]), vectorizer.transform([list_text[1]])
    
    # computes the cosine similarity
    cs_score = cosine_similarity(tfidf_text1, tfidf_text2)
    return cs_score[0][0]


def get_cosine_similarity(truth: str, test: str, model: str):

	truth_text = nlp(truth)
	test_text = nlp(test)

	# spacy has it's own similarity function
	similarity_spacy = truth_text.similarity(test_text)
	similarity_sklearn = compute_cosine_similarity(truth, test)

	print(f"   {model} Spacy: {similarity_spacy}")
	print(f"   {model} Sklearn: {similarity_sklearn}")
	return similarity_spacy, similarity_sklearn


if __name__ == "__main__":

	compare_texts()

