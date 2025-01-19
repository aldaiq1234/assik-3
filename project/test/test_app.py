import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from AP import get_ollama_response, add_to_chromadb, search_in_chromadb, generate_random_embedding

class TestChatbot(unittest.TestCase):

    def test_get_ollama_response_success(self):
        mock_response = {"choices": [{"message": {"content": "Test response"}}]}
        with patch('requests.post') as mocked_post:
            mocked_post.return_value.status_code = 200
            mocked_post.return_value.json.return_value = mock_response

            result = get_ollama_response("Hello")
            self.assertEqual(result, "Test response")

    def test_get_ollama_response_error(self):
        with patch('requests.post') as mocked_post:
            mocked_post.return_value.status_code = 500
            mocked_post.return_value.text = "Internal Server Error"

            result = get_ollama_response("Hello")
            self.assertIn("Error Ollama API", result)

    def test_generate_random_embedding(self):
        text = "test"
        embedding = generate_random_embedding(text, embedding_size=384)
        self.assertEqual(len(embedding), 384)
        self.assertTrue(all(isinstance(val, float) for val in embedding))

    def test_add_to_chromadb(self):
        with patch('chromadb.Client') as mocked_client:
            mocked_collection = MagicMock()
            mocked_client.return_value.get_or_create_collection.return_value = mocked_collection

            add_to_chromadb("Test text")
            mocked_collection.add.assert_called_once()

    def test_search_in_chromadb(self):
        with patch('chromadb.Client') as mocked_client:
            mocked_collection = MagicMock()
            mocked_collection.query.return_value = {
                "documents": ["Test document"]
            }
            mocked_client.return_value.get_or_create_collection.return_value = mocked_collection

            results = search_in_chromadb("Test query")
            self.assertIn("Test document", results)

if __name__ == '__main__':
    unittest.main()
