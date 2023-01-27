import csv
import os
import pandas as pd
import spacy
import re
import string
# from assemblyai import run_assemblyai_model
# from deepgram import run_deepgram_model





# nlp = spacy.load('en_core_web_lg')


# AssemblyAI links to full text
# 7f5uDPxTBXo https://www.assemblyai.com/playground/transcript/rj9268sd9s-cb9d-4102-9fe2-ff2ed2d4c300


# def pipeline():
# 	with open(FILEPATH) as f:
# 		reader = csv.DictReader(f)
# 		for row in reader:
# 			video_link = row["Link to video"]
# 			full_video_link = video_link.split("#t=")[0]
# 			video_id = os.path.basename(video_link).split(".mp4")[0]

# 			# assemblyai_file = f"{DIRPATH}/assembly/{vid_id}_assembly.txt"
# 			# deepgram_file = f"{DIRPATH}/deepgram/{vid_id}_deepgram.txt"
			
# 			run_assemblyai_model(full_video_link, video_id, assemblyai_file)
# 			# run_deepgram_model(full_video_link, video_id, deepgram_file)


def main():
	texts = get_groundtruth_text()
	breakpoint()
	for vid_id, truth in texts.items():

		groundtruth_file = f"{DIRPATH}/groundtruth/{vid_id}_truth.txt"
		# assemblyai_file = f"{DIRPATH}/assembly/{vid_id}_assembly.txt"
		# deepgram_file = f"{DIRPATH}/deepgram/{vid_id}_deepgram.txt"

		# add space after any punctuation
		re.sub(r'\.(?!\s|\d|$)', '. ', truth)
		# save the groundtruth data to open later
		if not os.path.exists(groundtruth_file):
			with open(groundtruth_file, "w") as f:
				f.write(truth)
				f.close()
		
		# with open(assemblyai_file, "r") as f:
		# 	assemblyai = f.read().replace('\n', '')

		# with open(deepgram_file, "r") as f:
		# 	deepgram = f.read().replace('\n', '')

		# metrics(vid_id, truth, assemblyai, deepgram)
		break


def metrics(video_id: str, truth: str, assemblyai: str, deepgram: str):

	print("\nGroundtruth:")
	print(truth)
	print("\nAssemblyAI:")
	print(assemblyai)
	print("\nDeepgram:")
	print(deepgram)

	print("\n")

	# sentence to sentence comparison
	# diarization

	cosine_similarity_assembly = cosine_similarity(video_id, truth, assemblyai, "assemblyai")
	cosine_similarity_deepgram = cosine_similarity(video_id, truth, deepgram, "deepgram")


def cosine_similarity(video_id: str, truth: str, test: str, model: str):

	truth_text = nlp(truth)
	test_text = nlp(test)

	# truth_text_no_puncuation = truth.translate(str.maketrans('', '', string.punctuation))
	# assemblyai_no_punctuation = assemblyai.translate(str.maketrans('', '', string.punctuation))

	truth_text_no_stop_words = nlp(' '.join([str(t) for t in truth_text if not t.is_stop]))
	test_text_no_stop_words = nlp(' '.join([str(t) for t in test_text if not t.is_stop]))

	similarity = truth_text.similarity(test_text)
	similarity_no_stop_words = truth_text_no_stop_words.similarity(test_text_no_stop_words)

	print(f"Video ID: {video_id}: ")
	print(f"   Cosine similarity {model}: {similarity}")
	print(f"   Cosine similarity {model} no stop words: {similarity_no_stop_words}")


if __name__ == "__main__":
    # read_csv()
    # main()
    get_texts()
    # Steps
    # count how many bills are in 2018
    # Run assembly ai model on 2018 




