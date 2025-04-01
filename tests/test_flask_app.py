import unittest
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        
    def test_home_page(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis</title>', res.data)
        
    def test_predict_page(self):
        res = self.client.post('/prredict', data=dict(text="I love this!"))
        self.assertEqual(res.status_code, 200)
        self.assertTrue(
            b'Positive' in res.data or b'Negative' in res.data,
            "Response should contain either 'Positive' or 'Negative'"
            )
        
if __name__=="__main__":
    unittest.main()
        
    