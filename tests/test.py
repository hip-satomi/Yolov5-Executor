import unittest
import os
import requests
from io import BytesIO
from PIL import Image
import json
import mlflow


class TestSegmentation(unittest.TestCase):

    def setUp(self):
        # download the image
        import requests

        url = 'https://fz-juelich.sciebo.de/s/wAXbC0MoN1G3ST7/download'
        r = requests.get(url, allow_redirects=True)

        print(len(r.content))

        with open('test.png', 'wb') as file:
            file.write(r.content)

    def test_standard(self):
        # test entrypoints: main (Yolov5s)
        self.predict('main')

    def test_custom(self):
        """Test execution with custom model file
        """
        self.predict('custom', params=dict(model='https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt'))

    def predict(self, entrypoint, params=None):
        if params is None:
            params = {}
        
        contours = []

        image = Image.open('test.png')

        # convert image into a binary png stream
        byte_io = BytesIO()
        image.save(byte_io, "png")
        byte_io.seek(0)

        # pack this into form data
        multipart_form_data = [
            ("files", ("data.png", byte_io, "image/png"))
        ]

        # get job specific environment variables
        CI_COMMIT_SHA = os.environ['CI_COMMIT_SHA']
        CI_REPOSITORY_URL = os.environ['CI_REPOSITORY_URL']

        additional_parameters = params

        # exactly request segmentation with the current repo version
        params = dict(
            repo=CI_REPOSITORY_URL,
            entry_point=entrypoint,
            version=CI_COMMIT_SHA,
            parameters=json.dumps(additional_parameters),
        )

        # send a request to the server
        response = requests.post(
            'http://localhost:8000/batch-image-prediction/', params=params, files=multipart_form_data, timeout=600
        )

        # output response
        print(response.content)

        # the request should be successful
        self.assertTrue(response.status_code == 200)

if __name__ == '__main__':
    unittest.main()
