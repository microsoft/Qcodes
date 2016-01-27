import os
from IPython.display import display, Javascript, HTML


def display_relative(base_file, rel_path, file_type=None):
    '''
    display some javascript, css, or html content in the notebook
    from a relative file path. Will use the file extension

    base_file: a directory or file (will use its parent directory) from which
        to find the target file

    rel_path: the path to the target file from the base file
    '''

    # some people use pkg_resources.resource_string for this, but that seems to
    # require an absolute path within the package.
    # this way lets us use a relative path
    if os.path.isdir(base_file):
        base_path = base_file
    else:
        base_path = os.path.split(base_file)[0]

    with open(os.path.join(base_path, rel_path)) as f:
        contents = f.read()
        if file_type is None:
            ext = os.path.splitext(rel_path)[1].lower()
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
