from threading import Thread
import socket
import time

PORT=5556


class Server(Thread):
    running = None
    server = None
    dostop = False

    def __init__(self):
        # let the thread start
        time.sleep(0.01)
        super().__init__()
        self.start()

    def run(self):
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        connection.settimeout(None)
        connection.bind(('0.0.0.0', PORT))
        connection.listen(0)
        while True:
            try:
                current_connection, address = connection.accept()
                data = current_connection.recv(2048)
                current_connection.send(data)

                current_connection.shutdown(socket.SHUT_RDWR)
                current_connection.close()
                self.run_command(data)
            except socket.timeout:
                if self.dostop:
                    return


    def stop(self):
        """
        Shutdown the server, close the event loop and join the thread
        """
        # this is not really threadsave
        self.dostop = True
        time.sleep(1)
        self.join()

    def run_command(self, input:str) -> None:
        # if input == 'init_state':
        print('started command {}'.format(input))
        time.sleep(0.5)
        print('finished command {}'.format(input))



from qcodes.instrument_drivers.Abaco.AbacoDac import AbacoDAC
s = Server()
try:
    abaco = AbacoDAC('abaco', 'localhost', port=PORT)
    abaco.close()
finally:
    s.stop()

del s


        # # this contains the server
        # # or any exception
        # server = self.future_restult.result()
        # # server.close()
        # self.loop.call_soon_threadsafe(server.close)
        # self.loop.call_soon_threadsafe(self.loop.stop)
        # self.join()
        # Monitor.running = None
# from qcodes.instrument_drivers.Abaco.AbacoDac import AbacoDAC

# import broadbean as bb
# ramp = bb.PulseAtoms.ramp  # args: start, stop
# sine = bb.PulseAtoms.sine  # args: freq, ampl, off


# # seq1 = bb.Sequence()
# # # Create the blueprints
# # bp_fill = bb.BluePrint()
# # bp_fill.setSR(1e9)
# # bp_fill.insertSegment(0, ramp, (0, 0), dur=6*28e-9)
# # bp_square = bb.BluePrint()
# # bp_square.setSR(1e9)
# # bp_square.insertSegment(0, ramp, (0, 0), dur=100e-9)
# # bp_square.insertSegment(1, ramp, (1e-3, 1e-3), name='top', dur=100e-9)
# # bp_square.insertSegment(2, ramp, (0, 0), dur=100e-9)
# # bp_boxes = bp_square + bp_square + bp_fill
# # #
# # bp_sine = bb.BluePrint()
# # bp_sine.setSR(1e9)
# # bp_sine.insertSegment(0, sine, (3.333e6, 1.5e-3, 0), dur=300e-9)
# # bp_sineandboxes = bp_sine + bp_square + bp_fill

# # # create elements
# # elem1 = bb.Element()
# # elem1.addBluePrint(1, bp_boxes)
# # elem1.addBluePrint(3, bp_sineandboxes)
# # #
# # elem2 = bb.Element()
# # elem2.addBluePrint(3, bp_boxes)
# # elem2.addBluePrint(1, bp_sineandboxes)

# # # Fill bp1 the sequence
# # seq1.addElement(1, elem1)  # Call signature: seq. pos., element
# # seq1.addElement(2, elem2)
# # seq1.addElement(3, elem1)

# # # set its sample rate
# # seq1.setSR(elem1.SR)
# # seq1.setChannelAmplitude(1, 1e-3)  # Call signature: channel, amplitude (peak-to-peak)
# # seq1.setChannelAmplitude(3, 1e-3)

# # seq1.plotSequence()


# bp1= bb.BluePrint()
# bp1.setSR(1e9)
# for i in range(3):
#     bp1.insertSegment(0, ramp, (0, 1.5), dur=3e-6)
#     bp1.insertSegment(0, ramp, (1.5, 0.0), dur=3e-6)



# elem1 = bb.Element()
# for i in range(8):
#     elem1.addBluePrint(i, bp1)

# seq = bb.Sequence()
# seq.addElement(1, elem1)
# # seq.addElement(2, elem1)

# seq.setSR(elem1.SR)
# for i in range(8):
#     seq.setChannelAmplitude(i, 1e-3)
# # seq.setChannelAmplitude(2, 1e-3)
# seq.plotSequence()

# package = seq.outputForAbacoDacFile()


# import qcodes as qc
# qc.Instrument.close_all()
# abaco = AbacoDAC('abaco', '172.20.3.94', port=27015)
# abaco.create_txt_file(package)
# abaco.create_dat_file(package)


# # 782/11: self.ask("init_state")
# # 833/4: abaco.ask("glWaveFileMask=test_")
# # 833/5: abaco.ask("glWaveFileMask=test_")
# # 833/6: abaco.ask("init_state")
# # 833/7: abaco.ask("config_state")
# # 835/2: abaco.ask("glWaveFileMask=test_")
# # 835/3: abaco.ask("init_state")
# # 835/4: abaco.ask("config_state")
# # 835/5: abaco.ask("load_waveform_state")
# # 835/6: abaco.ask("enable_upload_state")
# # 835/7: abaco.ask("enable_offload_state")
# # 835/9: abaco.ask("disable_offload_state")
# # 835/10: abaco.ask("glWaveFileMask=test2_")
# # 835/11: abaco.ask("load_waveform_state")
# # 835/12: abaco.ask("enable_offload_state")
