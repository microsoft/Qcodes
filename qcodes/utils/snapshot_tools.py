def inspect_snapshot(snapshot, key_word=''):
    '''
    Allows to inspect the contents of the snapshot and print them in a nice way
    '''
    _inspect_snapshot(snapshot, [], key_word=key_word)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # General flow and stuff
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def _inspect_snapshot(snapshot, path, key_word=''):
    if isinstance(snapshot, dict):
        if _is_parameter_like(snapshot):
            dprint(f"path = {path}, inspect as parameter")
            _inspect_parameter(snapshot, path, key_word=key_word)
        elif _is_instrument_like(snapshot):
            dprint(f"path = {path}, inspect as instrument")
            _inspect_instrument(snapshot, path, key_word=key_word)
        else:
            for key in snapshot.keys():
                dprint(f"path = {path}, key = {key}, inspect further..")
                _inspect_snapshot(snapshot[key], path + [key],
                                  key_word=key_word)
    else:
        dprint(f"printing the value..")
        if key_word in path[-1]:
            _inspect_value(snapshot, path)


def _print_with_path(value, path):
    print(f"{'.'.join(path)} --- {value}")


def dprint(str_):
    return
    print(f"++++++++++ {str_}")


# should be replicating meta_attrs (do not refer to them, as
# you do not know about subclasses and `snapshot_base` implementation)
_qcodes_object_keys = [
    'name',
    '__class__',
]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # Parameters
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


_parameter_keys = \
    _qcodes_object_keys + [
        'unit',
        'label',
        #     'full_name',  # do we care?
        #     'instrument',  # we should not be strict about this, but rather highlight it when we see it
        #     'instrument_name',  # do we care?
    ]


def _is_parameter_like(snapshot_part):
    return True if \
        all(key in snapshot_part.keys() for key in _parameter_keys) \
        else False


def _inspect_parameter(snapshot_part, path, key_word=''):
    parameter_dict = snapshot_part
    p = parameter_dict

    if key_word not in p['name']:
        return

    repr_ = f"{p['name']}"

    if hasattr(p, 'instrument_name'):
        repr_ += f" ({p['instrument_name']})"

    repr_ += f": {p['label']} = {p['value']} {p['unit']}"

    if 'division_value' in snapshot_part.keys():
        repr_ += f"; Divider = {p['division_value']}"

    _print_with_path(repr_, path)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # Instruments
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


_instrument_keys = \
    _qcodes_object_keys + [
        'parameters',
        'functions',  # to be deprecated
        'submodules',
    ]


def _is_instrument_like(snapshot_part):
    return True if \
        all(key in snapshot_part.keys() for key in _instrument_keys) \
        else False


def _inspect_instrument(snapshot_part, path, key_word=''):
    if key_word in snapshot_part['name']:
        repr_ = f"{snapshot_part['name']} ({snapshot_part['__class__']})"
        _print_with_path(repr_, path)

    for name in snapshot_part.keys():
        if name == 'parameters':
            parameters = snapshot_part['parameters']
            for p_name in parameters.keys():
                _inspect_parameter(parameters[p_name],
                                   path + ['parameters', p_name],
                                   key_word=key_word)
        if name == 'submodules':
            submodules = snapshot_part['submodules']
            for s_name in submodules.keys():
                s = submodules[s_name]
                _inspect_snapshot(s, path + ['submodules', s_name],
                                  key_word=key_word)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # Values
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def _inspect_value(snapshot, path):
    _print_with_path(snapshot, path)
