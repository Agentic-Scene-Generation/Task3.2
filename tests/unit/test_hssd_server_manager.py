import unittest

from unittest.mock import MagicMock, patch

from scenesmith.agent_utils.hssd_retrieval_server.server_manager import (
    HssdRetrievalServer,
)


class TestHssdRetrievalServerShutdown(unittest.TestCase):
    @patch(
        "scenesmith.agent_utils.hssd_retrieval_server.server_manager.make_server"
    )
    @patch(
        "scenesmith.agent_utils.hssd_retrieval_server.server_manager.HssdRetrievalApp"
    )
    def test_stop_uses_wsgi_server_shutdown(
        self, app_class: MagicMock, make_server_mock: MagicMock
    ):
        app = app_class.return_value
        http_server = make_server_mock.return_value
        http_server.serve_forever.side_effect = lambda: None

        server = HssdRetrievalServer(port=0, preload_retriever=False)
        with patch.object(server, "_wait_until_ready"):
            server.start()

        server.stop()

        http_server.shutdown.assert_called_once_with()
        http_server.server_close.assert_called_once_with()
        app.stop_processing.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
