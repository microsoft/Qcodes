"""Helper for adding content stored in a file to a jupyter notebook."""
import os
from pkg_resources import resource_string
from IPython.display import display, Javascript, HTML


# Originally I implemented this using regular open() and read(), so it
# could use relative paths from the importing file.
#
# But for distributable packages, pkg_resources.resource_string is the
# best way to load data files, because it works even if the package is
# in an egg or zip file. See:
# http://pythonhosted.org/setuptools/setuptools.html#accessing-data-files-at-runtime

def display_auto(qcodes_path, file_type=None):
    """
    Display some javascript, css, or html content in a jupyter notebook.

    Content comes from a package-relative file path. Will use the file
    extension to determine file type unless overridden by file_type

    Args:
        qcodes_path (str): the path to the target file within the qcodes
            package, like 'widgets/widgets.js'

        file_type (Optional[str]): Override the file extension to determine
            what type of file this is. Case insensitive, supported values
            are 'js', 'css', and 'html'
    """
    contents = resource_string('qcodes', qcodes_path).decode('utf-8')

    if file_type is None:
        ext = os.path.splitext(qcodes_path)[1].lower()
    elif 'js' in file_type.lower():
        ext = '.js'
    elif 'css' in file_type.lower():
        ext = '.css'
    else:
        ext = '.html'

    if ext == '.js':
        display(Javascript(contents))
    elif ext == '.css':
        display(HTML('<style>' + contents + '</style>'))
    else:
        # default to html. Anything else?
        display(HTML(contents))
