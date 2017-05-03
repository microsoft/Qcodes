import asyncio
from aiohttp import web
from aiohttp.hdrs import METH_POST
import aiohttp
import time
import json
from typing import Dict, Union

from qcodes.instrument_drivers.oxford.triton import Triton
from qcodes.instrument_drivers.oxford.mock_triton import MockTriton

class FridgeHttpServer:

    def __init__(self, name='triton', tritonaddress='http://localhost', use_mock_triton=True):
        if use_mock_triton:
            self.triton = MockTriton()
        else:
            self.triton = Triton(name=name, address=tritonaddress)

    async def handle_parameter(self, request):
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
        return web.Response(text="Usage ip/parameter?attribute=value i.e. ip/T1/attribute=value")

    async def handle_hostname(self, request):
        import socket
        host =  socket.gethostname()
        return web.Response(text=host)

    async def handle_monitor(self, request):
        """
        Handle websocket requests to serve the qcodes monitor
        """
        ws = web.WebSocketResponse(autoping=False)
        await ws.prepare(request)
        i = 0
        while True:
            # Idealy we would only send a message if the last one has been received. However,
            # with the aiohttp api there is no good way of detecting
            # that the client goes offline. await send_json only awaits coping to buffer,
            # so we can have hundreds of buffered messages. Drain only ensures that the buffer
            # is below some (non configurable) high level mark.
            # See blog post below for more details.
            # https://vorpus.org/blog/some-thoughts-on-asynchronous-api-design-in-a-post-asyncawait-world/#websocket-servers
            # We could try hacking in use of the pypi websockets packages which seems to be much more sane.
            # and what is used in the monitor. Such as https://gist.github.com/amirouche/a5da3cf6f0f11eaeb976

            # trying to solve this by sending a ping pong from the client
            # which we can await. However due to the 'design' of aiohttp we cant await a
            # ping with receive without either disabling autoping (as above) and sent the pong
            # manually or send a different message afterwards
            meta = self.prepare_monitor_data(self.triton)
            i += 1
            await ws.send_json(meta)
            await asyncio.sleep(1)
            print("awaing ping")
            msg = await ws.receive()
            ws.pong(msg.data)
            print("sent pong")
        return ws

    def prepare_monitor_data(self, triton) -> Dict[str, Union[list, float]]:
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
        app.router.add_route('*', '/monitor', self.handle_monitor)
        return app

def create_app(loop):
    fridgehttpserver = FridgeHttpServer()
    app = fridgehttpserver.run_app(loop)
    return app


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    fridgehttpserver = FridgeHttpServer()
    app = fridgehttpserver.run_app(loop)
    web.run_app(app, port=5678)