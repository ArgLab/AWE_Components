#!/usr/bin/env python3
# Copyright 2022, Educational Testing Service

# TODO: using numpy>1.26.4 seems to cause errors when installing from awe_workbench.
# This is related to the desired model (subwordbert) that we use for AWE.
# Version-locking seems to keep this from failing.
from spacy.cli.download import download
import os


def download_models():
    print('Downloading Spacy and Coreferee Lexicons')
    download('en_core_web_sm')
    download('en_core_web_lg')
    download('en_core_web_trf')
    os.system("python3 -m coreferee install en")


if __name__ == '__main__':
    download_models()
