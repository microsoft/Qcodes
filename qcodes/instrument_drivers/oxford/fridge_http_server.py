import asyncio
import socket
from typing import Dict, Union
import time
import json
import logging

from aiohttp import web
from aiohttp.hdrs import METH_POST
import websockets

import qcodes
from qcodes.instrument_drivers.oxford.triton import Triton
from qcodes.instrument_drivers.oxford.mock_triton import MockTriton

log = logging.getLogger(__name__)

class FridgeHttpServer:

    def __init__(self, name='triton', triton_address='http://localhost', use_mock_triton=True, triton_port=33576):

        self._send_websockets = False
        if use_mock_triton:
            self.triton = MockTriton()
        else:
            self.triton = Triton(name=name, address=triton_address, port=triton_port)

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
        ws_addr = qcodes.config['fridgeserver']['wsserver']
        ws_port = qcodes.config['fridgeserver']['wsport']
        ws_server = '{}:{}'.format(ws_addr, ws_port)
        ws_sleep = qcodes.config['fridgeserver']['wssendinterval']
        ws_connect_sleep = qcodes.config['fridgeserver']['wsconnectinterval']
        while True:
            try:
                log.debug("Connecting via websockets to {}".format(ws_server))
                async with websockets.connect(ws_server) as websocket:
                    log.debug("connected from sender to {}".format(ws_server))
                    while True:
                        if self._send_websockets:
                            data = self.prepare_ws_data(self.triton)
                            jsondata = json.dumps(data)
                            log.debug("Sending data on websocket {}".format(ws_server))
                            await websocket.send(jsondata)
                        await asyncio.sleep(ws_sleep)
            except OSError:
                log.info("Websocket server is offline will retry later")
                await asyncio.sleep(ws_connect_sleep)
            except websockets.exceptions.ConnectionClosed:
                log.info("Websocket closed. Will try to reconnect later")
                await asyncio.sleep(ws_connect_sleep)

    async def websocket_enable(self):
        ws_addr = qcodes.config['fridgeserver']['wsserver']
        ws_port = qcodes.config['fridgeserver']['wsport'] + 1
        ws_server = '{}:{}'.format(ws_addr, ws_port)
        ws_sleep = qcodes.config['fridgeserver']['wssendinterval']
        ws_connect_sleep = qcodes.config['fridgeserver']['wsconnectinterval']
        while True:
            try:
                log.debug("Connecting via websockets to {}".format(ws_server))
                async with websockets.connect(ws_server) as websocket:
                    log.debug("connected from receiver to {}".format(ws_server))
                    while True:
                        log.debug("waiting for websocket data")
                        data = await websocket.recv()
                        log.debug("got websocket data")
                        data = json.loads(data)
                        log.debug("data {}".format(data))
                        self._send_websockets = data['enable']
                        await asyncio.sleep(ws_sleep)
            except OSError:
                log.info("Websocket server is offline will retry later")
                await asyncio.sleep(ws_connect_sleep)
            except websockets.exceptions.ConnectionClosed:
                log.info("Websocket closed. Will try to reconnect later")
                await asyncio.sleep(ws_connect_sleep)


def create_app(loop):
    fridgehttpserver = FridgeHttpServer()
    app = fridgehttpserver.run_app(loop)
    return app


if __name__ == '__main__':
    log.debug("Starting server")
    loop = asyncio.get_event_loop()
    use_mock_triton = qcodes.config['fridgeserver']['usemocktriton']
    triton_address = qcodes.config['fridgeserver']['tritonaddress']
    triton_name = qcodes.config['fridgeserver']['tritonname']
    fridgehttpserver = FridgeHttpServer(name=triton_name,
                                        use_mock_triton=use_mock_triton,
                                        triton_address=triton_address)
    loop.create_task(fridgehttpserver.websocket_client())
    loop.create_task(fridgehttpserver.websocket_enable())
    app = fridgehttpserver.run_app(loop)
    http_port = qcodes.config['fridgeserver']['httpport']
    web.run_app(app, port=http_port)