# Module containing the config file reader class


class Config:
    """
    Object to be used for interacting with the config file.

    The ConfigFile is constantly synced with the config file on disk
    (provided that only this object was used to change the file).

    Args:
        filename (str): The path to the configuration file on disk
        isdefault (Optional[bool]): Whether this is the default Config object.
            Default: True.

    Attributes:
        default (Union[None, Config]): A reference to the default Config
            object, if it exists. Else None.
    """

    default = None

    def __init__(self, filename, isdefault=True):

        # If this is the default config object, we make it available as a
        # class method.
        if isdefault:
            default = self

        self._filename = filename
        self._cfg = ConfigParser()
        self._load()

    def _load(self):
        self._cfg.read(self._filename)

    def reload(self):
        """
        Reload the file from disk.
        """
        self._load()

    def get(self, section, field=None):
        """
        Gets the value of the specified section/field.
        If no field is specified, the entire section is returned
        as a dict.

        Example: Config.get('QDac Channel Labels', '2')

        Args:
            section (str): Name of the section
            field (Optional[str]): The field to return. If omitted, the entire
                section is returned as a dict
        """
        # Try to be really clever about the input
        if not isinstance(field, str) and (field is not None):
            field = '{}'.format(field)

        if field is None:
            output = dict(zip(self._cfg[section].keys(),
                              self._cfg[section].values()))
        else:
            output = self._cfg[section][field]

        return output

    def set(self, section, field, value):
        """
        Set a value in the config file.
        Immediately writes to disk.

        Args:
            section (str): The name of the section to write to
            field (str): The name of the field to write
            value (Union[str, float, int]): The value to write
        """
        if not isinstance(value, str):
            value = '{}'.format(value)

        self._cfg[section][field] = value

        with open(self._filename, 'w') as configfile:
            self._cfg.write(configfile)
