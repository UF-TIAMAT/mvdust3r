import pickle
from mvdust3r_server import MVDust3RModelClient
import json
import numpy as np
from PIL import Image

with open("/blue/prabhat/duminduaelamurem/wd/repo_tests/aaai/mvdust3r/data_out/data_test/metadata/abcdefg.json", "r") as f:
    dataset = json.load(f)

dataset = dataset["frames"]

images = []
w2cs = []

for frame in dataset:
    # Load image and convert to numpy array
    img = Image.open(frame["image_path"])
    img_np = np.array(img)
    images.append(img_np)
    
    # Convert w2c to numpy array and append
    w2c_np = np.array(frame["w2c"])
    w2cs.append(w2c_np)


with open("/blue/prabhat/duminduaelamurem/wd/repo_tests/aaai/vlfm/results/06-27/pickles/rgb_cache.pkl", "rb") as f:
    rgb_cache = pickle.load(f)

with open("/blue/prabhat/duminduaelamurem/wd/repo_tests/aaai/vlfm/results/06-27/pickles/transformation_matrix.pkl", "rb") as f:
    transformation_matrix = pickle.load(f)  


payload = {"rgb_cache": rgb_cache, "transformation_matrix": rgb_cache["transforms"]}
payload = pickle.dumps(payload)

# Uncomment this when doing server testing. 
client = MVDust3RModelClient()

for i in range(4, 6):

    rgb_cache = {}
    rgb_cache["images"] = images[:i]  # Taking one image at a time
    rgb_cache["transforms"] = w2cs[:i]  # Taking corresponding w2c matrices

    payload = {"rgb_cache": rgb_cache, "transformation_matrix": rgb_cache["transforms"]}
    payload = pickle.dumps(payload)
    response = client.generate_novel_views(payload=payload)
    print(f"Response for {i+1} images:")
    print(response)  # Assuming the response is a dict with a "response" key
