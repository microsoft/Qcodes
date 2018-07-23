from typing import List

import time
from statistics import mean, stdev
import math
import logging
import sys
import os
import pickle
import glob

import matplotlib.pyplot as plt
import numpy as np

import qcodes as qc

from qcodes.instrument_drivers.tektronix.AWG5208 import AWG5208



def generate_waveform_and_marker_data():
    waveform_name = 'test_sync_waveform.wfmx'
    N = 5000
    n_periods = 10

    wf = np.linspace(0, n_periods * 2 * np.pi, N)
    wf = np.sin(wf)
    wf = np.asarray(wf > 0, dtype=float)
    m1 = np.copy(wf).astype(int)
    m2 = np.copy(wf).astype(int)
    m3 = np.copy(wf).astype(int)
    m4 = np.copy(wf).astype(int)
    wf = np.array([wf * 0.5, m1, m2, m3, m4])
    return wf, waveform_name


def plot_waveform_and_markers():
    wf, wf_name = generate_waveform_and_marker_data()
    plt.plot(wf[0].T, label='main_wf')
    plt.plot(wf[1].T, label='m1')
    plt.plot(wf[2].T, label='m2')
    plt.legend()
    plt.show()


def upload_waveform_and_marker_data(awg: AWG5208, wf: np.ndarray,
                                    filename: str, channels: List[str]):
    short_filename = filename.replace('.wfmx', '')
    wfmx_file = awg.makeWFMXFile(wf, 0.350)
    awg.sendWFMXFile(wfmx_file, filename)
    awg.loadWFMXFile(filename)

    for chan in channels:
        exec('awg.' + chan + '.setWaveform(short_filename)')

    for chan in channels:
        exec('awg.' + chan + '.state(1)')

    awg.play()


def stop_and_disable(awg: AWG5208, channels: List[str]):
    pass


if __name__ == '__main__':
    awg_address = 'TCPIP0::192.168.15.118::inst0::INSTR'
    awg_channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']

    wf, wf_name = generate_waveform_and_marker_data()
    awg = AWG5208('awg5208', awg_address)
    upload_waveform_and_marker_data(awg, wf, wf_name, awg_channels)
