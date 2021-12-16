from unittest.mock import call

import pytest
from requests.exceptions import ConnectTimeout, HTTPError, ReadTimeout
from urllib3.exceptions import ReadTimeoutError

from qcodes import Parameter


class AnyStringWith(str):
    def __eq__(self, other):
        return self in other


@pytest.fixture(name='mock_webclient', autouse=True)
def setup_webclient(mocker):
    mock_slack_sdk_module = mocker.MagicMock(name='slack_sdk_module')
    mock_webclient = mocker.MagicMock(name='WebclientMock')
    mock_slack_sdk_module.WebClient = mocker.MagicMock()
    mock_slack_sdk_module.WebClient.return_value = mock_webclient
    mocker.patch.dict('sys.modules', slack_sdk=mock_slack_sdk_module)

    response = {'members': [{'name': 'dummyuser', 'id': 'DU123'}]}
    mock_webclient.users_list.return_value = response

    def mock_conversations_list(types):
        if 'im' in types.split(','):
            return {'channels': [{'user': 'DU123', 'id': 'CH234'}]}
        else:
            return None

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
    import qcodes.utils.slack  # pylint: disable=import-outside-toplevel
    slack = qcodes.utils.slack.Slack(config=slack_config, auto_start=False)

    return slack


def test_convert_command_should_convert_floats():
    import qcodes.utils.slack  # pylint: disable=import-outside-toplevel
    cmd, arg, kwarg = qcodes.utils.slack.convert_command('comm 0.234 key=0.1')
    assert cmd == 'comm'
    assert arg == [pytest.approx(0.234)]
    assert kwarg == {'key': pytest.approx(0.1)}


def test_slack_instance_should_contain_supplied_usernames(slack):
    assert 'dummyuser' in slack.users.keys()


def test_slack_instance_should_get_config_from_qc_config():
    from qcodes import config as cf  # pylint: disable=import-outside-toplevel
    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }
    cf.add(key='slack', value=slack_config)
    import qcodes.utils.slack  # pylint: disable=import-outside-toplevel
    slack = qcodes.utils.slack.Slack(config=None, auto_start=False)
    assert 'dummyuser' in slack.users.keys()


def test_slack_instance_should_start(mocker):
    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }
    mock_thread_start = mocker.patch('threading.Thread.start')
    import qcodes.utils.slack  # pylint: disable=import-outside-toplevel
    _ = qcodes.utils.slack.Slack(config=slack_config)

    mock_thread_start.assert_called()


def test_slack_instance_should_not_start_when_already_started(mocker):
    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }
    mock_thread_start = mocker.patch('threading.Thread.start')
    mock_thread_start.side_effect = RuntimeError

    import qcodes.utils.slack  # pylint: disable=import-outside-toplevel
    _ = qcodes.utils.slack.Slack(config=slack_config)

    mock_thread_start.assert_called()


def test_slack_instance_should_start_and_stop(mocker):
    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }
    mocker.patch('threading.Thread.start')

    import qcodes.utils.slack  # pylint: disable=import-outside-toplevel
    slack = qcodes.utils.slack.Slack(config=slack_config, interval=0)
    slack.stop()

    assert not slack._is_active


def test_slack_instance_should_return_username_from_id(mock_webclient, slack):
    def mock_users_info(user):
        if user == 'DU123':
            return {'user': {'name': 'dummyuser', 'id': 'DU123'}}
        else:
            return None

    mock_webclient.users_info.side_effect = mock_users_info

    assert {'name': 'dummyuser', 'id': 'DU123'} == slack.user_from_id('DU123')


def test_slack_instance_should_get_im_ids(mock_webclient):
    def conversations_history(channel, limit=None):
        if channel == 'CH234':
            messages = [{'user': 'DU123', 'text': f'm{i}',
                         'ts': f'{45.5 + i}'} for i in range(limit)]
            response = {'messages': messages}
            return response
        else:
            return None

    mock_webclient.conversations_history.side_effect = conversations_history

    slack = setup_slack()

    assert slack.users['dummyuser']['im_id'] == 'CH234'
    assert slack.users['dummyuser']['last_ts'] == pytest.approx(45.5)


def test_slack_instance_should_get_im_ids_with_zero_messages(mock_webclient):
    def conversations_history(channel, limit=None):
        if channel == 'CH234':
            response = {'messages': []}
            return response
        else:
            return None

    mock_webclient.conversations_history.side_effect = conversations_history
    slack = setup_slack()

    assert slack.users['dummyuser']['last_ts'] is None


def test_slack_instance_should_get_im_messages_w_count(slack, mock_webclient):
    def conversations_history(channel, limit=None):
        if channel == 'CH234':
            messages = [{'user': 'DU123', 'text': f'message{i}'}
                        for i in range(limit)]
            response = {'messages': messages}
            return response
        else:
            return None

    mock_webclient.conversations_history.side_effect = conversations_history

    messages = slack.get_im_messages('dummyuser', count=3)
    assert len(messages) == 3


def test_slack_instance_should_get_im_messages_without_channel(mock_webclient):
    def conversations_history(channel, limit=None):
        if channel == 'CH234':
            messages = [{'user': 'DU123', 'text': f'm{i}',
                         'ts': f'{45.5 + i}'} for i in range(limit)]
            response = {'messages': messages}
            return response
        else:
            return None

    mock_webclient.conversations_history.side_effect = conversations_history

    def mock_conversations_list(types):
        if 'im' in types.split(','):
            return {'channels': []}
        else:
            return None

    mock_webclient.conversations_list.side_effect = mock_conversations_list

    slack = setup_slack()

    messages = slack.get_im_messages('dummyuser')
    assert len(messages) == 0


def test_slack_instance_should_get_new_im_messages(mock_webclient):
    def generator_function():
        total = 8
        new = 3
        while True:
            new_messages = [{'user': 'DU123', 'text': f'm{i}',
                             'ts': f'{45.5 + i}'} for i in range(total)][-new:]
            response = {'messages': new_messages}
            yield response
            total += new

    generator = generator_function()

    def conversations_history(channel, limit=None, oldest=None):
        if channel == 'CH234':
            return next(generator)
        else:
            return None

    mock_webclient.conversations_history.side_effect = conversations_history

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


def test_slack_instance_should_update_with_exception(slack, mocker):
    method_name = 'qcodes.utils.slack.Slack.get_new_im_messages'
    mock_get_new_im_messages = mocker.patch(method_name)
    mocker.patch('warnings.warn')
    mocker.patch('logging.info')

    for exception in [ReadTimeout, HTTPError, ConnectTimeout,
                      ReadTimeoutError('pool', 'url', 'message')]:
        mock_get_new_im_messages.side_effect = exception
        slack.update()
        assert slack.tasks == []


def test_slack_instance_should_give_help_message(slack):
    message = slack.help_message()
    expected_message = '\nAvailable commands: `plot`, `msmt`, ' \
                       '`measurement`, `notify`, `help`, `task`'
    assert message == expected_message


def test_slack_instance_should_handle_messages(mock_webclient, slack):
    messages = {'dummyuser': [{'user': 'DU123', 'text': 'help'}]}
    slack.handle_messages(messages)
    expected_text = 'Results: \nAvailable commands: `plot`, ' \
                    '`msmt`, `measurement`, `notify`, `help`, `task`'
    expected_output = {'channel': 'CH234',
                       'text': expected_text}

    mock_webclient.chat_postMessage.assert_called_with(**expected_output)


def test_slack_inst_should_handle_messages_w_args_kw(mock_webclient, slack):
    text = 'task finished key=1'
    messages = {'dummyuser': [{'user': 'DU123', 'text': text}]}
    slack.handle_messages(messages)
    expected_output = {'channel': 'CH234', 'text': 'Added task "finished"'}
    mock_webclient.chat_postMessage.assert_called_with(**expected_output)


def test_slack_inst_should_handle_messages_w_parameter(mock_webclient, slack):
    slack.commands.update({'comm': Parameter(name='param')})
    messages = {'dummyuser': [{'user': 'DU123', 'text': 'comm'}]}
    slack.handle_messages(messages)
    expected_output = {'channel': 'CH234', 'text': 'Executing comm'}
    mock_webclient.chat_postMessage.assert_called_with(**expected_output)


def test_slack_inst_should_handle_messages_w_exception(mock_webclient, slack):
    messages = {'dummyuser': [{'user': 'DU123', 'text': 'help toomany'}]}
    slack.handle_messages(messages)
    text = "help_message() takes 1 positional argument but 2 were given\n"
    expected_output = {"channel": "CH234", "text": AnyStringWith(text)}
    mock_webclient.chat_postMessage.assert_called_with(**expected_output)


def test_slack_inst_should_handle_messages_w_unkn_cmd(mock_webclient, slack):
    messages = {'dummyuser': [{'user': 'DU123', 'text': 'comm'}]}
    slack.handle_messages(messages)
    text = 'Command comm not understood. Try `help`'
    expected_output = {'channel': 'CH234', 'text': text}
    mock_webclient.chat_postMessage.assert_called_with(**expected_output)


def test_slack_inst_should_add_unknown_task_command(mock_webclient, slack):
    slack.add_task('tcomm', channel='CH234')
    text = 'Task command tcomm not understood'
    expected_output = {'channel': 'CH234', 'text': text}
    mock_webclient.chat_postMessage.assert_called_with(**expected_output)


def test_slack_inst_should_upload_latest_plot(mock_webclient, slack, mocker):
    method_name = 'qcodes.utils.slack.BasePlot.latest_plot'
    mocker.patch(method_name, return_value=not None)
    mocker.patch('os.remove')
    slack.upload_latest_plot(channel='CH234')
    expected_output = {'channels': 'CH234', 'file': AnyStringWith('.jpg')}
    mock_webclient.files_upload.assert_called_with(**expected_output)


def test_slack_inst_should_not_fail_upl_latest_wo_plot(mock_webclient, slack):
    slack.upload_latest_plot(channel='CH234')
    expected_output = {'channel': 'CH234', 'text': 'No latest plot'}
    mock_webclient.chat_postMessage.assert_called_with(**expected_output)


def test_slack_inst_should_print_measurement(mock_webclient, slack, mocker):
    dataset = mocker.MagicMock()
    dataset.fraction_complete.return_value = 0.123
    mocker.patch('qcodes.utils.slack.active_data_set', return_value=dataset)

    slack.print_measurement_information(channel='CH234')

    print(mock_webclient.chat_postMessage.calls)

    text1 = f"Measurement is {0.123 * 100:.0f}% complete"
    expected_out1 = {"channel": "CH234", "text": text1}
    expected_out2 = {"channel": "CH234", "text": AnyStringWith("MagicMock")}
    actual = mock_webclient.chat_postMessage.call_args_list
    expected = [call(**expected_out1), call(**expected_out2)]
    assert actual == expected


def test_slack_inst_should_print_measurement_wo_latest(mock_webclient, slack):
    slack.print_measurement_information(channel='CH234')
    expected_output = {'channel': 'CH234', 'text': 'No latest dataset found'}
    mock_webclient.chat_postMessage.assert_called_with(**expected_output)
