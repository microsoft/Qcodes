import os
import tempfile
from IPython.display import display

from slacker import Slacker

from qcodes.plots.base import BasePlot
from qcodes import config


class Slack:
    def __init__(self, interval=5, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = config['user']['slack']

        self.slack = Slacker(self.config['token'])
        self.bot_id = self.slack.users.get_user_id(self.config['bot_name'])
        self.users = self.get_users(self.config['names'])
        self.get_im_ids(self.users)

        self.commands = {'plot': self.upload_latest_plot,
                         'msmt': self.print_measurement_information,
                         'notify': self.add_notification}
        self.notification_commands = {'finished': self.check_msmt_finished}

        self.interval = interval
        if config['gui']['notebook'] and config['core']['legacy_mp']:
            from qcodes.widgets.widgets import HiddenUpdateWidget
            self.update_widget = HiddenUpdateWidget(self.update, interval)
            display(self.update_widget)

        self.tasks = []

    def user_from_id(self, user_id):
        users = [user for user in self.users if
                 self.users[user]['id'] == user_id]
        assert len(users) == 1, "Could not find unique user with id {}".format(
            user_id)
        return users[0]

    def get_users(self, names):
        users = {}
        response = self.slack.users.list()
        for member in response.body['members']:
            if member['name'] in names:
                users[member['name']] = member
        if len(users) != len(names):
            remaining_names = [name for name in names if name not in users]
            raise RuntimeError(
                'Could not find names {}'.format(remaining_names))
        return users

    def get_im_ids(self, users):
        response = self.slack.im.list()
        user_ids = {user: users[user]['id'] for user in users}
        im_ids = {im['user']: im['id'] for im in response.body['ims']}
        for user, user_id in user_ids.items():
            if user_id in im_ids:
                users[user]['im_id'] = im_ids[user_id]
                # update last ts
                users[user]['last_ts'] = float(
                    self.get_im_messages(user, count=1)[0]['ts'])

    def get_im_messages(self, name, **kwargs):
        response = self.slack.im.history(channel=self.users[name]['im_id'],
                                         **kwargs)
        return response.body['messages']

    def get_new_im_messages(self):
        im_messages = {}
        for username, user in self.users.items():
            last_ts = user.get('last_ts', None)
            new_messages = self.get_im_messages(name=username, oldest=last_ts)
            # Sometimes kwarg 'oldest' does not work and also return message with ts==last_ts
            new_messages = [m for m in new_messages if
                            float(m['ts']) != last_ts]
            im_messages[username] = new_messages
            if new_messages:
                self.users[username]['last_ts'] = float(new_messages[0]['ts'])
        return im_messages

    def update(self):
        new_tasks = []
        for task in self.tasks:
            keep_task = task()
            if keep_task:
                new_tasks.append(task)
        self.tasks = new_tasks

        self.handle_new_messages()

    def handle_new_messages(self):
        new_messages = self.get_new_im_messages()
        for user, user_messages in new_messages.items():
            for message in user_messages:
                if message.get('subtype', None) == 'bot_message':
                    continue
                # Extract command (first word) and possible args
                text = message['text'].lower()
                text = text.rstrip(' ')
                command, *args = text.split(' ')
                if command in self.commands:
                    channel = self.users[user]['im_id']
                    func = self.commands[command]
                    func(*args, channel=channel, slack=self)

    def add_notification(self, command, channel, **kwargs):
        if command in self.notification_commands:
            self.slack.chat.post_message(
                text='Added notification "{}"'.format(command),
                channel=channel)
            func = self.notification_commands[command]
            self.tasks.append(partial(func, channel=channel))
        else:
            self.slack.chat.post_message(
                text='Notification command {} not understood'.format(command),
                channel=channel)

    def upload_latest_plot(self, channel, **kwargs):
        # Create temporary filename
        temp_filename = tempfile.mktemp(suffix='.jpg')
        # Retrieve latest plot
        latest_plot = BasePlot.latest_plot
        if latest_plot is not None:
            # Saves latest plot to filename
            latest_plot.save(filename=temp_filename)
            # Upload plot to slack
            self.slack.files.upload(temp_filename, channels=channel)
            os.remove(temp_filename)
        else:
            self.slack.chat.post_message(text='No latest plot',
                                         channel=channel)

    def print_measurement_information(self, channel, **kwargs):
        dataset = DataSet.latest_dataset
        if dataset is not None:
            dataset.sync()
            self.slack.chat.post_message(
                text='Measurement is {:.0f}% complete'.format(
                    100 * dataset.fraction_complete()),
                channel=channel)
            self.slack.chat.post_message(
                text=repr(dataset), channel=channel)
        else:
            self.slack.chat.post_message(
                text='No latest dataset found',
                channel=channel)

    def check_msmt_finished(self, channel, **kwargs):
        dataset = DataSet.latest_dataset
        if dataset is None:
            self.slack.chat.post_message(
                text='No latest dataset found',
                channel=channel)
            return False

        if dataset.sync():
            # Measurement is still running
            return True

        self.slack.chat.post_message(
            text='Measurement complete\n' + repr(dataset),
            channel=channel)
        return False
