import configparser
import os
import six


class HorseConfig(configparser.ConfigParser):

    def show(self):
        """ Show all the entries in the config object """
        for each_section in self.sections():
            for (each_key, each_val) in self.items(each_section):
                print('%s.%s: %s' % (each_section, each_key, each_val))

    def load_defaults(self, cfile=None):
        """ Load settings from file

        Arguments:
            cfile (string or None): file to load from. If None, then use config_filename()
        """
        if cfile is None:
            cfile = self.config_filename()
        self.read(cfile)

    def add_entry(self, combined_key: str, value):
        """ Add an entry to the config system 
        
        Arguments:
            combined_key (string): a key of the form bar.foo
            value: value to be set for the key
        """
        kk = combined_key.split('.')

        p = self
        for k in kk[:-1]:
            try:
                p[k]
            except:
                p[k] = dict()
            p = self[k]
        k = kk[-1]
        p[k] = str(value)

    @staticmethod
    def config_filename(horserc='horse.config'):
        """
        Get the location of the config file.

        The file location is determined in the following order

        - `$PWD/qcodesrc`

        - `$QCODESRC/qcodesrc`

        If no file is found, None is returned
        """

        if 'HORSERC' in os.environ:
            path = os.environ['HORSERC']
            if os.path.exists(path):
                fname = os.path.join(path, horserc)
                if os.path.exists(fname):
                    return fname

        if six.PY2:
            cwd = os.getcwdu()
        else:
            cwd = os.getcwd()
        fname = os.path.join(cwd, horserc)
        return fname
