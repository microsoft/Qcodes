import asyncio
import aiohttp
from aiohttp import web
from qcodes.instrument_drivers.oxford.triton import Triton
from aiohttp.hdrs import METH_POST
from qcodes.instrument_drivers.oxford.mock_triton import MockTriton
import time

class FridgeHttpServer:

    def __init__(self, name='triton', tritonaddress='http://localhost', use_mock_triton=True):
        if use_mock_triton:
            self.triton = MockTriton()
        else:
            self.triton = Triton(name=name, address=tritonaddress)

    async def handle(self, request):
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


    def run_app(self, loop):
        app = web.Application()
        app.router.add_get('/', self.index)
        parameter_regex = ""
        for parameter in self.triton.parameters:
            parameter_regex += parameter
            parameter_regex += "|"

        parameter_regex = parameter_regex[0:-1]
        app.router.add_get('/{{parametername:{}}}'.format(parameter_regex), self.handle)
        app.router.add_post('/{{parametername:{}}}'.format(parameter_regex), self.handle)
        return app


def create_app(loop):
    fridgehttpserver = FridgeHttpServer()
    app = fridgehttpserver.run_app(loop)
    return app


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    fridgehttpserver = FridgeHttpServer()
    app = fridgehttpserver.run_app(loop)
    web.run_app(app, port=8000)