"""
Slack bot is used to send information about qcodes via Slack IMs.
Some default commands are provided, and custom commands/tasks can be
attached (see below).

To setup the Slack bot, a bot first has to be registered via Slack
by clicking 'creating a new bot user' on https://api.slack.com/bot-users.
Once registered, the bot will have a name and unique token.
These and other settings have to be saved in a config dict (see init( or
Parameters) in :class:`Slack`).

Communication with the Slack bot is performed via instant messaging.
When an IM is sent to the Slack bot, it will be processed during the next
`update()` call (provided the username is registered in the config).
Standard commands provided to the Slack bot are:

- plot: Upload latest qcodes plot.
- msmt/measurement: Print information about latest measurement.
- notify finished: Send message once measurement is finished.

Custom commands can be added as (cmd, func) key-value pairs to
`self.commands`. When `cmd` is sent to the bot, `func` is evaluated.

Custom tasks can be added as well. These are functions that are performed
every time an update is called. The function must return a boolean that
indicates if the task should be removed from the list of tasks.
A custom task can be added as a (cmd, func) key-value pair  to
`self.task_commands`.
They can then be called through Slack IM via:

``notify/task {cmd} *args:`` register task with name `cmd` that is
performed every time `update()` is called.
"""

import os
import tempfile
from functools import partial
from time import sleep
import inspect
from slacker import Slacker
import warnings
import logging
import threading
import traceback
from requests.exceptions import ReadTimeout, HTTPError, ConnectTimeout
from requests.packages.urllib3.exceptions import ReadTimeoutError


from qcodes.plots.base import BasePlot
from qcodes import config as qc_config
from qcodes.instrument.parameter import _BaseParameter
from qcodes.loops import active_loop, active_data_set


class SlackTimeoutWarning(UserWarning):
    pass


def convert_command(text):
    def try_convert_str(string):
        try:
            val = int(string)
            return val
        except ValueError:
            pass
        try:
            val = float(string)
            return val
        except ValueError:
            pass

        return string

    # Format text to lowercase, and remove trailing whitespaces
    text = text.lower().rstrip(' ')
    command, *args_str = text.split(' ')

    # Convert string args to floats/kwargs
    args = []
    kwargs = {}
    for arg in args_str:
        if '=' in arg:
            # arg is a kwarg
            key, val = arg.split('=')
            # Try to convert into a float
            val = try_convert_str(val)
            kwargs[key] = val
        else:
            # arg is not a kwarg
            # Try to convert into a float
            val = try_convert_str(arg)
            args.append(val)
    return command, args, kwargs


class Slack(threading.Thread):

    def __init__(self, interval=3, config=None, auto_start=True, **commands):
        """
        Initializes Slack bot, including auto-updating widget if in notebook
        and using multiprocessing.

        Args:
            interval (int): Update interval for widget (must be over 1s).
            config (Optional[dict]): Config dict
                If not given, uses qc.config['user']['slack']
                The config dict must contain the following keys:

                - 'bot_name': Name of the bot
                - 'bot_token': Token from bot (obtained from slack website)
                - 'names': Usernames to periodically check for IM messages

            auto_start (bool): Defaults to True.

        """
        if config is not None:
            self.config = config
        else:
            self.config = qc_config.user.slack

        self.slack = Slacker(self.config['token'])
        self.bot_id = self.slack.users.get_user_id(self.config['bot_name'])
        self.users = self.get_users(self.config['names'])
        self.get_im_ids(self.users)

        self.commands = {'plot': self.upload_latest_plot,
                         'msmt': self.print_measurement_information,
                         'measurement': self.print_measurement_information,
                         'notify': self.add_task,
                         'help': self.help_message,
                         'task': self.add_task,
                         **commands}
        self.task_commands = {'finished': self.check_msmt_finished}

        self.interval = interval
        self.tasks = []

        # Flag that exits loop when set to True (called via self.exit())
        self._exit = False

        # Flag that enables actions to be performed in the event loop
        # Enabled via self.start(), disabled via self.stop()
        self._is_active = False

        # Call Thread init
        super().__init__()

        if auto_start:
            self.start()

    def start(self):
        self._is_active = True
        try:
            # Start thread, can only be called once
            super().start()
        except RuntimeError:
            # Thread already started, ignoring
            pass

    def run(self):
        """
        Thread event loop that periodically checks for updates.
        Can be stopped via  :meth:`stop` , after which the Thread is stopped.
        Returns:
            None.
        """
        while not self._exit:
            # Continue event loop
            if self._is_active:
                # check for updates
                self.update()
            sleep(self.interval)

    def stop(self):
        """
        Stop checking for updates. Can be started again via :meth:`start`.
        Returns:
            None.
        """
        self._is_active = False

    def exit(self):
        """
        Exit event loop, stop Thread.
        Returns:
            None
        """
        self._stop = True

    def user_from_id(self, user_id):
        """
        Retrieve user from user id.
        Args:
            user_id: Id from which to retrieve user information.

        Returns:
            dict: User information.
        """
        users = [user for user in self.users if
                 self.users[user]['id'] == user_id]
        assert len(users) == 1, "Could not find unique user with id {}".format(
            user_id)
        return users[0]

    def get_users(self, usernames):
        """
        Extracts user information for users.
        Args:
            usernames: Slack usernames of users.

        Returns:
            dict: {username: user}
        """
        users = {}
        response = self.slack.users.list()
        for member in response.body['members']:
            if member['name'] in usernames:
                users[member['name']] = member
        if len(users) != len(usernames):
            remaining_names = [name for name in usernames if name not in users]
            raise RuntimeError(
                f'Could not find names {remaining_names}')
        return users

    def get_im_ids(self, users):
        """
        Adds IM ids of users to users dict.
        Also adds `last_ts` to the latest IM message
        Args:
            users (dict): {username: user}

        Returns:
            None.
        """
        response = self.slack.im.list()
        user_ids = {user: users[user]['id'] for user in users}
        im_ids = {im['user']: im['id'] for im in response.body['ims']}
        for username, user_id in user_ids.items():
            if user_id in im_ids:
                users[username]['im_id'] = im_ids[user_id]
                # update last ts
                messages = self.get_im_messages(username=username, count=1)
                if messages:
                    users[username]['last_ts'] = float(messages[0]['ts'])
                else:
                    users[username]['last_ts'] = None

    def get_im_messages(self, username, **kwargs):
        """
        Retrieves IM messages from username.
        Args:
            username: Name of user.
            **kwargs: Additional kwargs for retrieving IM messages.

        Returns:
            List of IM messages.
        """
        channel = self.users[username].get('im_id', None)
        if channel is None:
            return []
        else:
            response = self.slack.im.history(channel=channel,
                                             **kwargs)
            return response.body['messages']

    def get_new_im_messages(self):
        """
        Retrieves new IM messages for each user in self.users.
        Updates user['last_ts'] to ts of newest message.
        Returns:
            im_messages (Dict): {username: [messages list]} newer than last_ts.
        """
        im_messages = {}
        for username, user in self.users.items():
            last_ts = user.get('last_ts', None)
            new_messages = self.get_im_messages(username=username,
                                                oldest=last_ts)
            # Kwarg 'oldest' sometimes also returns message with ts==last_ts
            new_messages = [m for m in new_messages if
                            float(m['ts']) != last_ts]
            im_messages[username] = new_messages
            if new_messages:
                self.users[username]['last_ts'] = float(new_messages[0]['ts'])
        return im_messages

    def update(self):
        """
        Performs tasks, and checks for new messages.
        Periodically called from widget update.
        Returns:
            None.
        """
        new_tasks = []
        for task in self.tasks:
            task_finished = task()
            if not task_finished:
                new_tasks.append(task)
        self.tasks = new_tasks

        new_messages = {}
        try:
            new_messages = self.get_new_im_messages()
        except (ReadTimeout, HTTPError, ConnectTimeout, ReadTimeoutError) as ex:
            # catch any timeouts caused by network delays
            warnings.warn('error retrieving slack messages',
                          SlackTimeoutWarning)
            logging.info(ex)
        self.handle_messages(new_messages)

    def help_message(self):
        """ Return simple help message """
        cc = ', '.join(['`' + str(k) + '`' for k in self.commands.keys()])
        return '\nAvailable commands: %s' % cc

    def handle_messages(self, messages):
        """
        Performs commands depending on messages.
        This includes adding tasks to be performed during each update.
        """
        for user, user_messages in messages.items():
            for message in user_messages:
                if message.get('user', None) != self.users[user]['id']:
                    # Filter out bot messages
                    continue
                channel = self.users[user]['im_id']
                # Extract command (first word) and possible args
                command, args, kwargs = convert_command(message['text'])
                if command in self.commands:
                    msg = f'Executing {command}'
                    if args:
                        msg += f' {args}'
                    if kwargs:
                        msg += f' {kwargs}'
                    self.slack.chat.post_message(text=msg, channel=channel)

                    func = self.commands[command]
                    try:
                        if isinstance(func, _BaseParameter):
                            results = func(*args, **kwargs)
                        else:
                            # Only add channel and Slack if they are explicit
                            # kwargs
                            func_sig = inspect.signature(func)
                            if 'channel' in func_sig.parameters:
                                kwargs['channel'] = channel
                            if 'slack' in func_sig.parameters:
                                kwargs['slack'] = self
                            results = func(*args, **kwargs)

                        if results is not None:
                            self.slack.chat.post_message(
                                text=f'Results: {results}',
                                channel=channel)

                    except:
                        self.slack.chat.post_message(
                            text=f'Error: {traceback.format_exc()}',
                            channel=channel)
                else:
                    self.slack.chat.post_message(
                        text='Command {} not understood. Try `help`'.format(
                            command),
                        channel=channel)

    def add_task(self, command, *args, channel, **kwargs):
        """
        Add a task to self.tasks, which will be executed during each update
        Args:
            command: Task command.
            *args: Additional args for command.
            channel: Slack channel (can also be IM channel).
            **kwargs: Additional kwargs for particular.

        Returns:
            None.
        """
        if command in self.task_commands:
            self.slack.chat.post_message(
                text=f'Added task "{command}"',
                channel=channel)
            func = self.task_commands[command]
            self.tasks.append(partial(func, *args, channel=channel, **kwargs))
        else:
            self.slack.chat.post_message(
                text=f'Task command {command} not understood',
                channel=channel)

    def upload_latest_plot(self, channel, **kwargs):
        """
        Uploads latest plot (if any) to slack channel.
        The latest plot is retrieved from
        :class:`qcodes.plots.base.BasePlot`, which is updated
        every time a new qcodes plot is instantiated.
        Args:
            channel: Slack channel (can also be IM channel).
            **kwargs: Not used.

        Returns:
            None.
        """
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
        """
        Prints information about the current measurement.
        Information printed is percentage complete, and dataset representation.
        Dataset is retrieved from DataSet.latest_dataset, which updates itself
        every time a new dataset is created
        Args:
            channel: Slack channel (can also be IM channel).
            **kwargs: Not used.

        Returns:
            None.
        """
        dataset = active_data_set()
        if dataset is not None:
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
        """
        Checks if the latest measurement is completed.
        Args:
            channel: Slack channel (can also be IM channel).
            **kwargs: Not used.

        Returns:
            bool: True if measurement is finished, False otherwise.
        """
        if active_loop() is None:
            self.slack.chat.post_message(
                text='Measurement complete',
                channel=channel)
            return True
        else:
            return False
