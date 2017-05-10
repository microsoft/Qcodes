import asyncio
from aiohttp import web
from aiohttp.hdrs import METH_POST
import socket
import time
from typing import Dict, Union
import websockets
import json

from qcodes.instrument_drivers.oxford.triton import Triton
from qcodes.instrument_drivers.oxford.mock_triton import MockTriton

class FridgeHttpServer:

    def __init__(self, name='triton', tritonaddress='http://localhost', use_mock_triton=True, tritonport=33576, websocket_wait_time=10):
        self._websocket_wait_time = websocket_wait_time
        if use_mock_triton:
            self.triton = MockTriton()
        else:
            self.triton = Triton(name=name, address=tritonaddress, port=tritonport)

    async def handle_parameter(self, request):
        """
        Handler for qcodes parameters, Can expose the value, unit, name and label of a parameter via get
        or allow the setting of a parameter via a POST request.
        """
        parametername = request.match_info.get('parametername', None)
        query = request.query
        valid_attributes = ('value', 'unit', 'name', 'label')
        if parametername in self.triton.parameters:
            parameter = getattr(self.triton, parametername)
            if request.method == METH_POST:
                data = await request.json()
                try:
                    parameter.set(data['setpoint'])
                except ValueError:
                    return web.Response(status=405)
                return web.Response(text='OK')
            attribute = query.get('attribute', None)
            if attribute in valid_attributes:
                if attribute == 'value':
                    data = parameter()
                else:
                    data = getattr(parameter, attribute)

                return web.Response(text=str(data))
            elif attribute == None:
                return web.Response(status=404, text="Usage ip/parameter?attribute=attributes i.e. "
                                                     "ip/T1/attribute=value. "
                                                     "Valid attributes are {}".format(valid_attributes))
            else:
                return web.Response(status=404, text="Parameter {} does not have attribute {}".format(parameter, attribute))
        else:
            return web.Response(status=404, text="Parameter {} not found".format(parametername))

    async def index(self, request):
        """
        Handler for a very basic index page that just tells you how to use the html api.
        """
        return web.Response(text="Usage ip/parameter?attribute=value i.e. ip/T1/attribute=value")

    async def handle_hostname(self, request):
        host =  socket.gethostname()
        return web.Response(text=host)

    @staticmethod
    def prepare_monitor_data(triton) -> Dict[str, Union[list, float]]:
        """
        Return a dict that contains the parameter from the Triton to be monitored.
        """

        def get_data(triton, parametername):
            parameter = getattr(triton, parametername)
            parameter.get()
            temp = parameter._latest()
            temp['unit'] = parameter.unit
            # replace with reading directly from reg database once running on fridge computer
            namedict = triton.chan_temp_names.get(parametername, None)
            if namedict:
                fullname = namedict['name']
            else:
                fullname = None
            temp['name'] = fullname or parameter.label or parameter.name
            temp['value'] = str(temp['value'])
            if temp["ts"] is not None:
                temp["ts"] = time.mktime(temp["ts"].timetuple())

            return temp

        triton_parameters = []
        for parametername in triton.parameters:
            temp = get_data(triton, parametername)
            triton_parameters.append(temp)

        triton_parameters.append(get_data(triton, 'action'))
        triton_parameters.append(get_data(triton, 'status'))

        parameters_dict = {"instrument": triton.name, "parameters": triton_parameters}
        state = {"ts": time.time(), "parameters": [parameters_dict]}

        return state

    @staticmethod
    def prepare_ws_data(triton) -> Dict[str, Union[list, float]]:
        """
        Return a dict that contains the parameter from the Triton to be exported via WS. This includes a read only copy
        of all parametes exposed by the driver.
        """

        triton_parameters = []
        for parametername in triton.parameters:
            parameter = getattr(triton, parametername)
            temp = {}
            temp['value'] = parameter.get()
            temp['name'] = parameter.name
            temp['label'] = parameter.label
            temp['unit'] = parameter.unit
            triton_parameters.append(temp)
        hostname = socket.gethostname()
        state = {"ts": time.time(), "parameters": triton_parameters, 'hostname': hostname}

        return state

    def run_app(self, loop):
        app = web.Application()
        app.router.add_get('/', self.index)

        # construct a regex matching all parameters that the
        # triton driver exposes.
        parameter_regex = ""
        settable_parameter_regex = ""
        for parameter in self.triton.parameters:
            parameter_regex += parameter
            parameter_regex += "|"
            if getattr(self.triton, parameter).has_set:
                settable_parameter_regex += parameter
                settable_parameter_regex += '|'
        parameter_regex = parameter_regex[0:-1]
        settable_parameter_regex = parameter_regex[0:-1]

        # route all parameters to handle_parameter.
        # The slightly cryptic syntax below means {{parametername:paramregex}}.
        # means route all matching the paramregex and make the parameter known
        # to the handler as parametername. {{ is needed for literal { in format strings
        app.router.add_get('/{{parametername:{}}}'.format(parameter_regex), self.handle_parameter)
        app.router.add_post('/{{parametername:{}}}'.format(settable_parameter_regex), self.handle_parameter)
        app.router.add_get('/', self.index)
        app.router.add_get('/hostname', self.handle_hostname)
        return app

    async def websocket_client(self):
        while True:
            print("connecting")
            try:
                async with websockets.connect('ws://localhost:8765') as websocket:
                    while True:
                        data = self.prepare_ws_data(self.triton)
                        jsondata = json.dumps(data)
                        print("sending")
                        await websocket.send(jsondata)
                        await asyncio.sleep(1)
            except OSError:
                print("Server is offline will retry later")
                await asyncio.sleep(10)
            except websockets.exceptions.ConnectionClosed:
                print("connection lost")
                await asyncio.sleep(10)


def create_app(loop):
    fridgehttpserver = FridgeHttpServer()
    app = fridgehttpserver.run_app(loop)
    return app


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    fridgehttpserver = FridgeHttpServer()
    #fridgehttpserver = FridgeHttpServer(name='Triton t10', use_mock_triton=False, tritonaddress='172.20.2.203', tritonport=33576)
    loop.create_task(fridgehttpserver.websocket_client())
    app = fridgehttpserver.run_app(loop)
    web.run_app(app, port=5678)