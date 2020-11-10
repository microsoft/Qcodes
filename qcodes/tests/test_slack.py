def test_slack(mocker):

    slack_config = {
        'bot_name': 'bot',
        'token': '123',
        'names': ['dummyuser']
    }

    mock_slacker_module = mocker.MagicMock(name='slacker_module')
    slacker_class_mock = mocker.MagicMock(name='SlackerMock')
    slacker_class_mock.users.get_user_id = mocker.MagicMock(
        return_value='bot_id'
    )
    response = mocker.MagicMock()
    response.body = {'members': [{'name': 'dummyuser', 'id': '123'}]}
    slacker_class_mock.users.list = mocker.MagicMock(
        name='mock_users_list',
        return_value=response
    )
    mock_slacker_module.Slacker = slacker_class_mock
    mock_slacker_module.Slacker = mocker.MagicMock(
        return_value=slacker_class_mock
    )

    mocker.patch.dict('sys.modules', slacker=mock_slacker_module)
    from qcodes.utils.slack import Slack
    slack = Slack(config=slack_config, auto_start=False)
    assert 'dummyuser' in slack.users
