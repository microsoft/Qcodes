import qcodes as qc
# inst_meas to be a list
# if more than one meas intrumt
# make subplots // windows(not possilbe atm) of all of them
# label with
# label measured intrumet setintrument (all if two dimenensional) then ID
# plot foreground

def init(mainfolder):
    loc_provider = qc.FormatLocation(
        fmt=mainfolder + '/data/{counter}_{name}')
    qc.data.data_set.DataSet.location_provider=loc_provider

def do1d(inst_set, start, stop, division, delay, *inst_meas):
    if not "{name}" in  qc.data.data_set.DataSet.location_provider.fmt:
        raise ValueError("missing  in {}".format( qc.data.data_set.DataSet.location_provider.fmt))

    name = inst_set.label
    name = name + "".join([i.label for i in inst_meas])
    name = name.replace(" ", "")
    loop = qc.Loop(inst_set[start:stop:division], delay).each(*inst_meas)
    data = loop.get_data_set(name=name)
    title = "#{}".format(data.location_provider.counter)
    plot = qc.QtPlot()
    for j, i in enumerate(inst_meas):
        title = title+"{}_{}".format(i._instrument.name, i.name)
    for j, i in enumerate(inst_meas):
        inst_meas_name = "{}_{}".format(i._instrument.name, i.name)
        plot.add(getattr(data, inst_meas_name),subplot=j+1,name=name)
        plot.subplots[j].showGrid(True,True)
        if j == 0 :
                plot.subplots[0].setTitle(title)
        else:
                plot.subplots[j].setTitle("")
    try:
        _ = loop.with_bg_task(plot.update, plot.save).run()
    except KeyboardInterrupt:
        print("Measurement Interrupted")
    return data

def do2d(inst_set, start, stop, division, delay, inst_set2, start2, stop2, division2, delay2, *inst_meas):
    if not "{name}" in  qc.data.data_set.DataSet.location_provider.fmt:
        raise ValueError("missing  in {}".format( qc.data.data_set.DataSet.location_provider.fmt))

    name =  inst_set.label+ inst_set2.label
    name = name + "".join([i.label for i in inst_meas])
    name = name.replace(" ", "")

    loop = qc.Loop(inst_set[start:stop:division], delay).loop(inst_set2[start2:stop2:division2], delay2).each(
       *inst_meas)
    data = loop.get_data_set(name=name)
    title = "#{}".format(data.location_provider.counter)
    plot = qc.QtPlot()
    name = "{}{}".format(data.location_provider.counter, name)
    for j, i in enumerate(inst_meas):
        inst_meas_name = "{}_{}".format(i._instrument.name, i.name)
        plot.add(getattr(data, inst_meas_name),subplot=j+1,name=name)
        plot.subplots[j].showGrid(True,True)
        if j == 0 :
                plot.subplots[0].setTitle(title)
        else:
                plot.subplots[j].setTitle("")
    try:
        _ = loop.with_bg_task(plot.update, plot.save).run()
    except KeyboardInterrupt:
        print("Measurement Interrupted")
    return data


def show_num(id):
    str_id = str(id)
    if len(str_id) <3:
        str_id = "0" + str_id
    t = qc.DataSet.location_provider.fmt.format(counter=str_id, name="*")
    file = [i for i  in qc.DiskIO(".").list(t) if ".dat" in i ][0]
    data = qc.load_data(file)
    plots = []
    for value in data.arrays.keys():
        if "set" not in value:
            plot = qc.QtPlot(getattr(data, value))
            title = "#{}".format(str_id)
            plot.subplots[0].setTitle(title)
            plots.append(plot)
    return data, plots
