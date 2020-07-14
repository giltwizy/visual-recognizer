###############################################
# author:   Gilton Bosco
# date:     14 July 2020
# twitter:  @giltwizy
###############################################

import os
from dotenv import load_dotenv
import json
from watson_developer_cloud import VisualRecognitionV3

#loading the .env file from the root directory
load_dotenv()

#getting API key from the .env file
my_api_key = os.getenv('api_key')

visual_recognition = VisualRecognitionV3(
    '2018-03-19',
    my_api_key)

with open('./testimage.jpg', 'rb') as images_file:
    classes = visual_recognition.classify(
        images_file,
        threshold='0.6',
        classifier_ids='default').get_result()
print(json.dumps(classes, indent=2))
