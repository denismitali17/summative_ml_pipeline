from locust import HttpUser, task, between
import os

# Configure a local image to post; the locust worker will read this file repeatedly.
IMAGE_PATH = os.environ.get('LOCUST_TEST_IMAGE', 'pneumonia_dataset/Normal/Normal-10.png')

class PredictUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def predict_image(self):
        if not os.path.exists(IMAGE_PATH):
            return
        with open(IMAGE_PATH, 'rb') as img:
            files = {'file': (os.path.basename(IMAGE_PATH), img, 'image/png')}
            # Sends multipart/form-data POST to /api/predict
            self.client.post('/api/predict', files=files, name='/api/predict')
