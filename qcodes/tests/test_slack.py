import pytest

import qcodes.utils.slack


@pytest.fixture(name='mock_webclient')
def setup_webclient(mocker):
    response = {'members': [{'name': 'dummyuser', 'id': 'DU123'}]}
    mock_webclient = mocker.patch('qcodes.utils.slack.WebClient')()
    mock_webclient.users_list.return_value = response

    return mock_webclient


@pytest.fixture(name='slack')
def setup_slack(mock_webclient):
    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }
    slack = qcodes.utils.slack.Slack(config=slack_config, auto_start=False)

    return slack


def test_slack_instance_should_contain_supplied_usernames(slack):
    assert 'dummyuser' in slack.users.keys()


def test_slack_instance_should_get_config_from_qc_config(mocker, mock_webclient):
    from qcodes import config as qc_config
    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }
    qc_config.add(key='slack', value=slack_config)
    slack = qcodes.utils.slack.Slack(config=None, auto_start=False)


def test_slack_instance_should_start(mocker, mock_webclient):
    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }
    mocker.patch('threading.Thread.start')

    slack = qcodes.utils.slack.Slack(config=slack_config)


def test_slack_instance_should_not_start_when_already_started(mocker, mock_webclient):
    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }
    mock_thread_start = mocker.patch('threading.Thread.start')
    mock_thread_start.side_effect = RuntimeError

    slack = qcodes.utils.slack.Slack(config=slack_config)


def test_slack_instance_should_start_and_stop(mocker):
    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }
    response = {'members': [{'name': 'dummyuser', 'id': 'DU123'}]}
    mock_webclient = mocker.patch('qcodes.utils.slack.WebClient')()
    mock_webclient.users_list.return_value = response
    mocker.patch('threading.Thread.start')

    slack = qcodes.utils.slack.Slack(config=slack_config, interval=0)
    slack.stop()


def test_slack_instance_should_return_username_from_id(mock_webclient, slack):
    def mock_users_info(user):
        if user == 'DU123':
            return {'user': {'name': 'dummyuser', 'id': 'DU123'}}

    mock_webclient.users_info.side_effect = mock_users_info

    assert {'name': 'dummyuser', 'id': 'DU123'} == slack.user_from_id('DU123')


def test_slack_instance_should_get_im_ids(mock_webclient, slack):
    def mock_conversations_list(types):
        if 'im' in types.split(','):
            return {'channels': [{'user': 'DU123', 'id': 'CH234'}]}

    mock_webclient.conversations_list.side_effect = mock_conversations_list

    users = {'dummyuser': {'name': 'dummyuser', 'id': 'DU123'}}
    slack.get_im_ids(users)

    assert users['dummyuser']['im_id'] == 'CH234'
