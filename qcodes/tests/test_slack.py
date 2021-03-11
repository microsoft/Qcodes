from unittest.mock import call

import pytest
from requests.exceptions import ReadTimeout, HTTPError, ConnectTimeout
from urllib3.exceptions import ReadTimeoutError

from qcodes import Parameter


class AnyStringWith(str):
    def __eq__(self, other):
        return self in other


@pytest.fixture(name='mock_webclient', autouse=True)
def setup_webclient(mocker):
    mock_slack_sdk_module = mocker.MagicMock(name='slack_sdk_module')
    mock_webclient = mocker.MagicMock(name='WebclientMock')
    mock_slack_sdk_module.WebClient = mocker.MagicMock(return_value=mock_webclient)
    mocker.patch.dict('sys.modules', slack_sdk=mock_slack_sdk_module)

    response = {'members': [{'name': 'dummyuser', 'id': 'DU123'}]}
    mock_webclient.users_list.return_value = response

    def mock_conversations_list(types):
        if 'im' in types.split(','):
            return {'channels': [{'user': 'DU123', 'id': 'CH234'}]}

    mock_webclient.conversations_list.side_effect = mock_conversations_list

    return mock_webclient


@pytest.fixture(name='slack')
def slack_fixture():
    return setup_slack()


def setup_slack():
    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }
    import qcodes.utils.slack
    slack = qcodes.utils.slack.Slack(config=slack_config, auto_start=False)

    return slack


def test_convert_command_should_convert_floats():
    import qcodes.utils.slack
    command, arg, kwarg = qcodes.utils.slack.convert_command('comm 0.234 key=0.1')
    assert command == 'comm'
    assert arg == [pytest.approx(0.234)]
    assert kwarg == {'key': pytest.approx(0.1)}


def test_slack_instance_should_contain_supplied_usernames(slack):
    assert 'dummyuser' in slack.users.keys()


def test_slack_instance_should_get_config_from_qc_config():
    from qcodes import config as qc_config
    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }
    qc_config.add(key='slack', value=slack_config)
    import qcodes.utils.slack
    slack = qcodes.utils.slack.Slack(config=None, auto_start=False)
    assert 'dummyuser' in slack.users.keys()


def test_slack_instance_should_start(mocker):
    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }
    mocker.patch('threading.Thread.start')
    import qcodes.utils.slack
    slack = qcodes.utils.slack.Slack(config=slack_config)


def test_slack_instance_should_not_start_when_already_started(mocker):
    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }
    mock_thread_start = mocker.patch('threading.Thread.start')
    mock_thread_start.side_effect = RuntimeError

    import qcodes.utils.slack
    slack = qcodes.utils.slack.Slack(config=slack_config)


def test_slack_instance_should_start_and_stop(mocker):
    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }
    mocker.patch('threading.Thread.start')

    import qcodes.utils.slack
    slack = qcodes.utils.slack.Slack(config=slack_config, interval=0)
    slack.stop()


def test_slack_instance_should_return_username_from_id(mock_webclient, slack):
    def mock_users_info(user):
        if user == 'DU123':
            return {'user': {'name': 'dummyuser', 'id': 'DU123'}}

    mock_webclient.users_info.side_effect = mock_users_info

    assert {'name': 'dummyuser', 'id': 'DU123'} == slack.user_from_id('DU123')


def test_slack_instance_should_get_im_ids(mock_webclient):
    def mock_conversations_history(channel, limit=None):
        if channel == 'CH234':
            response = {'messages': [{'user': 'DU123', 'text': f'm{i}', 'ts': f'{45.5 + i}'} for i in range(limit)]}
            return response

    mock_webclient.conversations_history.side_effect = mock_conversations_history

    slack = setup_slack()

    assert slack.users['dummyuser']['im_id'] == 'CH234'
    assert slack.users['dummyuser']['last_ts'] == pytest.approx(45.5)


def test_slack_instance_should_get_im_ids_with_zero_messages(mock_webclient):
    def mock_conversations_history(channel, limit=None):
        if channel == 'CH234':
            response = {'messages': []}
            return response

    mock_webclient.conversations_history.side_effect = mock_conversations_history
    slack = setup_slack()

    assert slack.users['dummyuser']['last_ts'] is None


def test_slack_instance_should_get_im_messages_with_count_iso_limit_specified(slack, mock_webclient):
    def mock_conversations_history(channel, limit=None):
        if channel == 'CH234':
            response = {'messages': [{'user': 'DU123', 'text': f'message{i}'} for i in range(limit)]}
            return response

    mock_webclient.conversations_history.side_effect = mock_conversations_history

    messages = slack.get_im_messages('dummyuser', count=3)
    assert len(messages) == 3


def test_slack_instance_should_get_im_messages_without_channel(mock_webclient):
    def mock_conversations_history(channel, limit=None):
        if channel == 'CH234':
            response = {'messages': [{'user': 'DU123', 'text': f'm{i}', 'ts': f'{45.5 + i}'} for i in range(limit)]}
            return response

    mock_webclient.conversations_history.side_effect = mock_conversations_history

    def mock_conversations_list(types):
        if 'im' in types.split(','):
            return {'channels': []}

    mock_webclient.conversations_list.side_effect = mock_conversations_list

    slack = setup_slack()

    messages = slack.get_im_messages('dummyuser')
    assert len(messages) == 0


def test_slack_instance_should_get_new_im_messages(mock_webclient):
    def generator_function():
        total = 8
        new = 3
        while True:
            response = {'messages': [{'user': 'DU123', 'text': f'm{i}', 'ts': f'{45.5 + i}'} for i
                                     in range(total)][-new:]}
            yield response
            total += new

    generator = generator_function()

    def mock_conversations_history(channel, limit=None, oldest=None):
        if channel == 'CH234':
            return next(generator)

    mock_webclient.conversations_history.side_effect = mock_conversations_history

    slack = setup_slack()

    new_messages = slack.get_new_im_messages()
    assert len(new_messages['dummyuser']) == 3


def test_slack_instance_should_update(slack):
    slack.update()
    assert slack.tasks == []


def test_slack_instance_should_update_with_task_returning_false(slack):
    slack.add_task('finished', channel='CH234')
    slack.update()
    assert slack.tasks == []


def test_slack_instance_should_update_with_task_returning_true(slack, mocker):
    mocker.patch('qcodes.utils.slack.active_loop', return_value=not None)

    slack.add_task('finished', channel='CH234')
    slack.update()
    task_added = slack.tasks[-1]

    assert 'Slack.check_msmt_finished' in str(task_added.func)


def test_slack_instance_should_update_with_new_im_messages_exception(slack, mocker):
    mock_get_new_im_messages = mocker.patch('qcodes.utils.slack.Slack.get_new_im_messages')
    mocker.patch('warnings.warn')
    mocker.patch('logging.info')

    for exception in [ReadTimeout, HTTPError, ConnectTimeout,
                      ReadTimeoutError('pool', 'url', 'message')]:
        mock_get_new_im_messages.side_effect = exception
        slack.update()
        assert slack.tasks == []


def test_slack_instance_should_give_help_message(slack):
    message = slack.help_message()
    expected_message = '\nAvailable commands: `plot`, `msmt`, `measurement`, ' \
                       '`notify`, `help`, `task`'
    assert message == expected_message


def test_slack_instance_should_handle_messages(mock_webclient, slack):
    messages = {'dummyuser': [{'user': 'DU123', 'text': 'help'}]}
    slack.handle_messages(messages)
    expected_output = {'channel': 'CH234',
                       'text': 'Results: \nAvailable commands: `plot`, '
                               '`msmt`, `measurement`, `notify`, `help`, `task`'}

    mock_webclient.chat_postMessage.assert_called_with(**expected_output)


def test_slack_instance_should_handle_messages_with_args_and_kwargs(mock_webclient, slack):
    messages = {'dummyuser': [{'user': 'DU123', 'text': 'task finished key=1'}]}
    slack.handle_messages(messages)
    expected_output = {'channel': 'CH234', 'text': 'Added task "finished"'}
    mock_webclient.chat_postMessage.assert_called_with(**expected_output)


def test_slack_instance_should_handle_messages_with_base_parameter(mock_webclient, slack):
    slack.commands.update({'comm': Parameter(name='param')})
    messages = {'dummyuser': [{'user': 'DU123', 'text': 'comm'}]}
    slack.handle_messages(messages)
    expected_output = {'channel': 'CH234', 'text': 'Executing comm'}
    mock_webclient.chat_postMessage.assert_called_with(**expected_output)


def test_slack_instance_should_handle_messages_with_exception(mock_webclient, slack, mocker):
    messages = {'dummyuser': [{'user': 'DU123', 'text': 'help toomany'}]}
    slack.handle_messages(messages)
    partial_expected_text = 'TypeError: help_message() takes 1 positional argument but 2 were given\n'
    expected_output = {'channel': 'CH234', 'text': AnyStringWith(partial_expected_text)}
    mock_webclient.chat_postMessage.assert_called_with(**expected_output)


def test_slack_instance_should_handle_messages_with_unknown_command(mock_webclient, slack):
    messages = {'dummyuser': [{'user': 'DU123', 'text': 'comm'}]}
    slack.handle_messages(messages)
    expected_output = {'channel': 'CH234', 'text': 'Command comm not understood. Try `help`'}
    mock_webclient.chat_postMessage.assert_called_with(**expected_output)


def test_slack_instance_should_add_unknown_task_command(mock_webclient, slack):
    slack.add_task('tcomm', channel='CH234')
    expected_output = {'channel': 'CH234', 'text': 'Task command tcomm not understood'}
    mock_webclient.chat_postMessage.assert_called_with(**expected_output)


def test_slack_instance_should_upload_latest_plot(mock_webclient, slack, mocker):
    mocker.patch('qcodes.utils.slack.BasePlot.latest_plot', return_value=not None)
    mocker.patch('os.remove')
    slack.upload_latest_plot(channel='CH234')
    expected_output = {'channels': 'CH234', 'file': AnyStringWith('.jpg')}
    mock_webclient.files_upload.assert_called_with(**expected_output)


def test_slack_instance_should_not_fail_when_uploading_latest_plot_without_plot(mock_webclient, slack):
    slack.upload_latest_plot(channel='CH234')
    expected_output = {'channel': 'CH234', 'text': 'No latest plot'}
    mock_webclient.chat_postMessage.assert_called_with(**expected_output)


def test_slack_instance_should_print_measurement_information(mock_webclient, slack, mocker):
    dataset = mocker.MagicMock()
    dataset.fraction_complete.return_value = 0.123
    mocker.patch('qcodes.utils.slack.active_data_set', return_value=dataset)

    slack.print_measurement_information(channel='CH234')

    print(mock_webclient.chat_postMessage.calls)

    expected_output = {'channel': 'CH234', 'text': 'Measurement is {:.0f}% complete'.format(0.123 * 100)}
    expected_output2 = {'channel': 'CH234', 'text': AnyStringWith('MagicMock')}
    assert mock_webclient.chat_postMessage.call_args_list == [call(**expected_output), call(**expected_output2)]


def test_slack_instance_should_print_measurement_information_without_latest_dataset(mock_webclient, slack):
    slack.print_measurement_information(channel='CH234')
    expected_output = {'channel': 'CH234', 'text': 'No latest dataset found'}
    mock_webclient.chat_postMessage.assert_called_with(**expected_output)
