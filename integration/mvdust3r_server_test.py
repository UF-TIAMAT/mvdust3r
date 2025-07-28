import pickle
from mvdust3r_server import MVDust3RModelClient


with open("/blue/prabhat/duminduaelamurem/wd/repo_tests/aaai/vlfm/results/06-27/pickles/rgb_cache.pkl", "rb") as f:
    rgb_cache = pickle.load(f)

with open("/blue/prabhat/duminduaelamurem/wd/repo_tests/aaai/vlfm/results/06-27/pickles/transformation_matrix.pkl", "rb") as f:
    transformation_matrix = pickle.load(f)  


payload = {"rgb_cache": rgb_cache, "transformation_matrix": transformation_matrix}
payload = pickle.dumps(payload)


client = MVDust3RModelClient()
response = client.generate_novel_views(payload=payload)
print(response)
print(type(response))
print(response["response"])  # Assuming the response is a dict with a "response" key
