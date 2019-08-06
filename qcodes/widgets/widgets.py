"""Qcodes-specific widgets for jupyter notebook."""
import threading
import os
from IPython.display import display
from ipywidgets.widgets import *
from traitlets import Unicode, Bool

import qcodes as qc
from qcodes.utils.threading import UpdaterThread


class LoopManagerWidget(DOMWidget):
    def __init__(self, interval=1):
        super().__init__()
        self.active_loop_label = Label(value='No active loop')

        self.pause_button = Button(icon='pause', tooltip='Pause measurement')
        self.stop_button = Button(icon='stop', tooltip='Stop measurement')
        self.force_stop_button = Button(icon='stop', button_style='danger',
                                        tooltip='Force stop measurement (not safe)')
        self.buttons_hbox = HBox([self.pause_button, self.stop_button, self.force_stop_button])

        self.progress_bar = FloatProgress(
            value=0,
            min=0,
            max=100,
            step=0.1,
            #             description='Loading:',
            bar_style='info',
            orientation='horizontal',
            layout=Layout(width='95%')
        )

        self.vbox = VBox([self.active_loop_label,
                          self.buttons_hbox,
                          self.progress_bar])

        self.pause_button.on_click(self.pause_loop)
        self.stop_button.on_click(self.stop_loop)
        self.force_stop_button.on_click(self.force_stop_loop)

        self.dot_counter = 0

        self.updater = UpdaterThread(self.update_widget,
                                     interval=interval)

    def display(self):
        display(self.vbox)

    def stop_loop(self, *args, **kwargs):
        qc.stop()
        # Loop won't stop while paused
        qc.active_loop().paused = False
        self.stop_button.disabled = True

    def pause_loop(self, *args, **kwargs):
        qc.active_loop().paused = ~qc.active_loop().paused
        # Toggle play/pause icon
        self.pause_button.icon = 'play' if self.pause_button.icon == 'pause' else 'pause'

    def force_stop_loop(self, *args, **kwargs):
        for thread in threading.enumerate():
            if thread.name == 'qcodes_loop':
                thread.terminate()

    def update_widget(self):
        try:
            import sys
            sys.stdout.flush()
            if not qc.active_loop():
                self.active_loop_label.value = 'No active loop'
                self.progress_bar.value = 0
                self.progress_bar.description = ''
                self.pause_button.icon = 'pause'
            else:
                dataset_location = f'Active loop: {qc.active_data_set().location}'
                dataset_name = os.path.split(dataset_location)[-1]
                self.active_loop_label.value = dataset_name

            if not qc.active_loop()._is_stopped and self.stop_button.disabled:
                self.stop_button.disabled = False
            if not qc.active_data_set():
                self.progress_bar.value = 0
            else:
                self.progress_bar.value = qc.active_data_set().fraction_complete() * 100
                if qc.active_loop().paused:
                    if qc.active_loop()._is_paused:
                        self.progress_bar.description = u'\u231b ' + f'{self.progress_bar.value:.0f}%'
                        self.progress_bar.bar_style = 'warning'
                    else:
                        dots = ['    ',
                                u'\u00b7   ',
                                u'\u00b7\u00b7  ',
                                u'\u00b7\u00b7\u00b7 ']
                        self.progress_bar.description = dots[self.dot_counter] + f'{self.progress_bar.value:.0f}%'
                        self.dot_counter = (self.dot_counter  + 1) % 4
                else:
                    self.progress_bar.description = f'{self.progress_bar.value:.0f}%'
                    self.progress_bar.bar_style = 'info'
        except:
            pass