from unittest import TestCase
from unittest.mock import patch, MagicMock


class TestSlack(TestCase):

    def test_slack(self):

        slack_config = {'bot_name': 'bot', 'token': '123', 'names': ['dummyuser']}

        mock_slacker_module = MagicMock(name='slacker_module')
        slacker_class_mock = MagicMock(name='SlackerMock')
        slacker_class_mock.users.get_user_id = MagicMock(return_value='bot_id')
        response = MagicMock()
        response.body = {'members': [{'name': 'dummyuser', 'id': '123'}]}
        slacker_class_mock.users.list = MagicMock(name='mock_users_list',
                                                  return_value=response)
        mock_slacker_module.Slacker = slacker_class_mock
        mock_slacker_module.Slacker = MagicMock(return_value=slacker_class_mock)

        with patch.dict('sys.modules', slacker=mock_slacker_module):
            from qcodes.utils.slack import Slack

            slack = Slack(config=slack_config, auto_start=False)
            self.assertIn('dummyuser', slack.users)




