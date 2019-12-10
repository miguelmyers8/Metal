from dotenv import load_dotenv
import os
from os.path import join, dirname

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

AWSAccessKeyId = os.getenv("AWSAccessKeyId")
AWSSecretKey = os.getenv("AWSSecretKey")
region = os.getenv("region")
