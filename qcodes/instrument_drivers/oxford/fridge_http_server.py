import asyncio
from aiohttp import web
from qcodes.instrument_drivers.oxford.triton import Triton
from aiohttp.hdrs import METH_POST
from mock_triton import MockTriton

async def handle(request):
    parametername = request.match_info.get('parametername', None)
    query = request.query
    valid_attributes = ('value', 'unit', 'name', 'label')
    if parametername in triton.parameters:
        parameter = getattr(triton, parametername)
        if request.method == METH_POST:
            data = await request.json()
            parameter.set(data['setpoint'])
            return web.Response(text='OK')
        attribute = query.get('attribute', None)
        if attribute in valid_attributes:
            if attribute == 'value':
                data = parameter()
            else:
                data = getattr(parameter, attribute)

            return web.Response(text=str(data))
        else:
            # TODO proper 404
            return web.Response(status=404, text="Parameter {} does not have attribute {}".format(parameter, attribute))
    else:
        return web.Response(status=404, text="Parameter {} not found".format(parametername))

async def index(request):
    return web.Response(text="Usage ip/parameter?attribute=value i.e. ip/T1/attribute=value")


def create_app(loop):
    global triton
    app = web.Application()
    app.router.add_get('/', index)
    #app.router.add_get('/{parameter}', handle_get)
    #app.router.add_post('/{parameter}', handle_post)

    #triton = Triton(name = 'Triton 10', address='172.20.2.203', port=33576, tmpfile='t10-thermometry.reg')
    triton = MockTriton()
    parameter_regex = ""
    for parameter in triton.parameters:
        parameter_regex += parameter
        parameter_regex += "|"

    parameter_regex = parameter_regex[0:-1]
    app.router.add_get('/{{parametername:{}}}'.format(parameter_regex), handle)
    app.router.add_post('/{{parametername:{}}}'.format(parameter_regex), handle)
    return app

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    app = create_app(loop)
    web.run_app(app, port=8000)