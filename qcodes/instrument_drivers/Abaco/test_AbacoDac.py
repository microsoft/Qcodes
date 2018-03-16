from qcodes.instrument_drivers.Abaco.AbacoDac import AbacoDAC

import broadbean as bb
ramp = bb.PulseAtoms.ramp  # args: start, stop
sine = bb.PulseAtoms.sine  # args: freq, ampl, off


# seq1 = bb.Sequence()
# # Create the blueprints
# bp_fill = bb.BluePrint()
# bp_fill.setSR(1e9)
# bp_fill.insertSegment(0, ramp, (0, 0), dur=6*28e-9)
# bp_square = bb.BluePrint()
# bp_square.setSR(1e9)
# bp_square.insertSegment(0, ramp, (0, 0), dur=100e-9)
# bp_square.insertSegment(1, ramp, (1e-3, 1e-3), name='top', dur=100e-9)
# bp_square.insertSegment(2, ramp, (0, 0), dur=100e-9)
# bp_boxes = bp_square + bp_square + bp_fill
# #
# bp_sine = bb.BluePrint()
# bp_sine.setSR(1e9)
# bp_sine.insertSegment(0, sine, (3.333e6, 1.5e-3, 0), dur=300e-9)
# bp_sineandboxes = bp_sine + bp_square + bp_fill

# # create elements
# elem1 = bb.Element()
# elem1.addBluePrint(1, bp_boxes)
# elem1.addBluePrint(3, bp_sineandboxes)
# #
# elem2 = bb.Element()
# elem2.addBluePrint(3, bp_boxes)
# elem2.addBluePrint(1, bp_sineandboxes)

# # Fill bp1 the sequence
# seq1.addElement(1, elem1)  # Call signature: seq. pos., element
# seq1.addElement(2, elem2)
# seq1.addElement(3, elem1)

# # set its sample rate
# seq1.setSR(elem1.SR)
# seq1.setChannelAmplitude(1, 1e-3)  # Call signature: channel, amplitude (peak-to-peak)
# seq1.setChannelAmplitude(3, 1e-3)

# seq1.plotSequence()


bp1= bb.BluePrint()
bp1.setSR(1e9)
for i in range(3):
    bp1.insertSegment(0, ramp, (0, 1.5), dur=3e-6)
    bp1.insertSegment(0, ramp, (1.5, 0.0), dur=3e-6)



elem1 = bb.Element()
for i in range(8):
    elem1.addBluePrint(i, bp1)

seq = bb.Sequence()
seq.addElement(1, elem1)
# seq.addElement(2, elem1)

seq.setSR(elem1.SR)
for i in range(8):
    seq.setChannelAmplitude(i, 1e-3)
# seq.setChannelAmplitude(2, 1e-3)
seq.plotSequence()

package = seq.outputForAbacoDacFile()


import qcodes as qc
qc.Instrument.close_all()
abaco = AbacoDAC('abaco', '172.20.3.94', port=27015)
abaco.create_txt_file(package)
abaco.create_dat_file(package)


# 782/11: self.ask("init_state")
# 833/4: abaco.ask("glWaveFileMask=test_")
# 833/5: abaco.ask("glWaveFileMask=test_")
# 833/6: abaco.ask("init_state")
# 833/7: abaco.ask("config_state")
# 835/2: abaco.ask("glWaveFileMask=test_")
# 835/3: abaco.ask("init_state")
# 835/4: abaco.ask("config_state")
# 835/5: abaco.ask("load_waveform_state")
# 835/6: abaco.ask("enable_upload_state")
# 835/7: abaco.ask("enable_offload_state")
# 835/9: abaco.ask("disable_offload_state")
# 835/10: abaco.ask("glWaveFileMask=test2_")
# 835/11: abaco.ask("load_waveform_state")
# 835/12: abaco.ask("enable_offload_state")
