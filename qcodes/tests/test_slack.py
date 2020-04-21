from unittest import TestCase
from unittest.mock import patch, MagicMock


class TestSlack(TestCase):

    def test_Slack(self):

        slack_config = {'bot_name': 'test_bot', 'token': '123123', 'names': ['dummyuser']}

        mock_slacker_module = MagicMock(name='slacker_module')
        slacker_class_mock = MagicMock(name='SlackerMock')
        slacker_class_mock.users.get_user_id = MagicMock(return_value='bot_id')
        response = MagicMock()
        response.body = {'members': [{'name': 'dummyuser', 'id': '123'}]}
        slacker_class_mock.users.list = MagicMock(name='mock_users_list', return_value=response)
        mock_slacker_module.Slacker = slacker_class_mock
        mock_slacker_module.Slacker = MagicMock(return_value=slacker_class_mock)

        with patch.dict('sys.modules', slacker=mock_slacker_module):

            from qcodes.utils.slack import Slack

            slack = Slack(config=slack_config, auto_start=False)


config = {'bot_name': 'test_bot', 'token': '123123', 'names': ['dummyuser']}


# %%
mock_slacker_module = MagicMock(name='slacker_module')
slacker_class_mock = MagicMock(name='SlackerMock')
slacker_class_mock.users.get_user_id = MagicMock(return_value='bot_id')
response = MagicMock()
response.body = {'members': [{'name': 'dummyuser', 'id': '123'}]}
slacker_class_mock.users.list = MagicMock(name='mock_users_list', return_value=response)
mock_slacker_module.Slacker = slacker_class_mock

r = slacker_class_mock.users.get_user_id()
print(r)
r = slacker_class_mock.users.list()
print(r)

mock_slacker_module.Slacker = MagicMock(return_value=slacker_class_mock)

# %%
with patch.dict('sys.modules', slacker=mock_slacker_module):

    # with patch.object(slacker, 'Slacker', return_value=slacker_class_mock ) as slacker_mock:
    #from slacker import Slacker

    from slacker import Slacker
    from qcodes.utils.slack import Slack

    slack = Slacker(config['token'])
    slack.users.get_user_id('s')
    slack.users.list()
    #slack.get_users = get_users

    print(slack)

    slack = Slack(config=config, auto_start=False)
    users = {'dummyuser': {'id': '123'}}
    r = slack.get_im_ids(users)
