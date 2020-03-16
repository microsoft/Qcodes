import time
import ipywidgets as widgets
import logging
from IPython.display import display


class LoggingWidget(logging.StreamHandler):
    """Widget that shows the last filtered logging messages

    Usage:
        To create the logging widget and have it show only speicfic messages
        from the layout, execute the following code

        ```
        logging_widget = LoggingWidget(
            select_messages=[
                {'sender': 'layout', 'message': 'targeting pulse sequence'},
                {'sender': 'layout', 'message': 'layout setup'},
                {'sender': 'layout', 'message': 'layout started'},
                {'sender': 'layout', 'message': 'layout stopped'},
                {'sender': 'layout', 'message': 'performing acquisition'}
            ],
            max_rows=10
        )
        logging_widget.display()
        ```
    """
    def __init__(self, widget=None, select_messages=None, max_rows=10):
        if widget is None:
            widget = widgets.SelectMultiple(rows=1, layout=widgets.Layout(width='99%', max_width='500px'))

        self.widget = widget

        self.select_messages = select_messages or []
        self.max_rows = max_rows

        self._last_records = []

        super().__init__()

    def display(self):
        display(self.widget)

    def _valid_message(self, record):
        msg = self.format(record)
        sender = record.name

        for select_message in self.select_messages:
            if isinstance(select_message, str):
                if select_message in msg:
                    return True
            elif isinstance(select_message, dict):
                if 'message' in select_message and select_message['message'].lower() not in msg.lower():
                    continue
                if 'sender' in select_message and select_message['sender'].lower() not in sender.lower():
                    continue
                return True
        else:
            return False

    def emit(self, record):
        self._last_records.append(record)
        self._last_records = self._last_records[:100]

        msg = self.format(record)
        sender = record.name.split('.')[-1]

        if not self._valid_message(record):
            return

        # Add message to stack
        timestamp = time.strftime('%H:%m:%S')
        message = f'{timestamp} - {sender}: {msg}'
        options = list(self.widget.options)
        options = [message] + options
        options = options[:self.max_rows]
        self.widget.options = options
        self.widget.rows = len(options)+2