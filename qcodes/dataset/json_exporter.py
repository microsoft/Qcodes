json_template_linear={"type": 'linear',
                      'x': {'data': [], 'name': "", 'full_name': '', 'is_setpoint':True,  'unit':''},
                      'y': {'data': [], 'name': "", 'full_name': '', 'is_setpoint':False, 'unit':''}}

json_template_heatmap = {"type": 'heatmap',
                         'x': {'data': [], 'name': "", 'full_name': '', 'is_setpoint':True,  'unit':''},
                         'y': {'data': [], 'name': "", 'full_name': '', 'is_setpoint':True,  'unit':''},
                         'z': {'data': [], 'name': "", 'full_name': '', 'is_setpoint':False,  'unit':''}}



def export_data_as_json_linear(data, length, state, location):
    import numpy as np
    import json
    if len(data) > 0:
        npdata = np.array(data)
        xdata = npdata[:,0]
        ydata = npdata[:,1]
        state['json']['x']['data'] += xdata.tolist()
        state['json']['y']['data'] += ydata.tolist()

        with open(location, mode='w') as f:
            json.dump(state['json'], f)


def export_data_as_json_heatmap(data, length, state, location):
    import numpy as np
    import json
    if len(data) > 0:
        npdata = np.array(data)
        array_start = state['data']['location']
        array_end = length
        state['data']['x'][array_start:array_end] = npdata[:, 0]
        state['data']['y'][array_start:array_end] = npdata[:, 1]
        state['data']['z'][array_start:array_end] = npdata[:, 2]

        state['data']['location'] = array_end

        state['json']['x']['data'] = state['data']['x'][
                                     0:-1:state['data']['ylen']].tolist()
        state['json']['y']['data'] = state['data']['y'][
                                     0:state['data']['ylen']].tolist()
        state['json']['z']['data'] = state['data']['z'].reshape(
            state['data']['xlen'], state['data']['ylen']).tolist()
        with open(location, mode='w') as f:
            json.dump(state['json'], f)

