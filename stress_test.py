from threading import Thread
import requests
import time
from time import sleep, perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
 
AZURE_VM_IP = "XXX"

KERAS_REST_API_URL = "http://" + AZURE_VM_IP + "/predict"
IMAGE_PATH = "fish.png"
NUM_REQUESTS = 10
SLEEP_COUNT = 0.05
image = open(IMAGE_PATH, "rb").read()
def call_predict_endpoint(n):
	payload = {"image": image}
	r = requests.post(KERAS_REST_API_URL, files=payload).json()
	if r["success"]:
		return "[INFO] thread {} OK".format(n)
	else:
		return "[INFO] thread {} FAILED".format(n)
	

thread_results = []
start_time = perf_counter()

with ThreadPoolExecutor(max_workers=5) as executor:
	for i in range(0, NUM_REQUESTS):
		thread_results.append(executor.submit(call_predict_endpoint, i))
	for res in as_completed(thread_results):
		print(res.result())

end_time = perf_counter()

print("It took total time ", end_time-start_time) 