import csv
import os
import pandas as pd
import spacy
import re
import string


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


