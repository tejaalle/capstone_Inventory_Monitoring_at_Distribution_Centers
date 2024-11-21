import logging
import os
import sys

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
import json
import torch.nn.functional as F
import subprocess
import sys
import copy

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

try:
    import smdebug
except ImportError:
    install('smdebug')


JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

logging.basicConfig(level=logging.INFO,
                    format="'levelName:'%(levelname)s, 'ts':%(asctime)s, pathname: %(pathname)s, message: %(message)s")
logger = logging.getLogger("ModelLogger")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def Net():
    logger.info("Model creation  started.")
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    logger.info("Model creation completed.")
    return model





def model_fn(model_dir):
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running on device= {device}")
        model = Net().to(device)
        logger.info(f"model_dir is {model_dir}")
        with open(os.path.join(model_dir, "model.pth"), "rb") as f:
            logger.info("Loading the model")
            checkpoint = torch.load(f , map_location =device)
            logger.info("created checkpoint")
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                logger.error("Checkpoint is not a valid state dictionary")
                model = checkpoint
                return model
            logger.info("Loaded Model sucessfully")
        model.eval()
        return model
    except Exception as ex:
        logger.error(f"modelfn failed with exception {ex}")
        return model




def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    #if content_type == JPEG_CONTENT_TYPE: return io.BytesIO(request_body)
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    if content_type == JPEG_CONTENT_TYPE: return Image.open(io.BytesIO(request_body))
    logger.debug('SO loded JPEG content')
    # process a URL submitted to the endpoint
    
    if content_type == JSON_CONTENT_TYPE:
        #img_request = requests.get(url)
        logger.debug(f'Request body is: {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# inference
def predict_fn(input_object, model):
    logger.info('defining transform')
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
    logger.info("transforming input")
    input_object=test_transform(input_object)
    
    with torch.no_grad():
        logger.info("Making a call to model")
        prediction = model(input_object.unsqueeze(0))
    return prediction