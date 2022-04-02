import unittest
import mlflow
import json

class TestSegmentation(unittest.TestCase):   
    def test_info(self):
        from info import main
        main()

    def test_mlproject_info(self):
        run = mlflow.projects.run(
            './',
            entry_point="info",
            backend='local',
        )

        # download the output artifact
        client = mlflow.tracking.MlflowClient()
        client.download_artifacts(run.run_id, 'output.json', './')

        with open('output.json', 'r') as input_file:
            info_result = json.load(input_file)
            self.assertTrue(info_result['name'] == 'yolov5-executor')

        
if __name__ == '__main__':
    unittest.main()