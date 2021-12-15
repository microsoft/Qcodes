"""
IO managers for QCodes.

IO managers wrap whatever physical storage layer the user wants to use
in an interface mimicking the built-in <open> context manager, with
some restrictions to minimize the overhead in creating new IO managers.

The main thing these managers need to implement is the open context manager:

- Only the context manager needs to be implemented, not separate
  open function and close methods.

- open takes the standard parameters:

    - filename: (string)
    - mode: (string) only 'r' (read), 'w' (write), and 'a' (append) are
      expected to be implemented. As with normal file objects, the only
      difference between write and append is that write empties the file
      before adding new data, and append leaves the existing contents in
      place but starts writing at the end.
    - encoding: If a special output encoding is desired. i.e. 'utf8

- the file-like object returned should implement a minimal set of operations.

  In read mode:
    - read([size]): read to the end or at most size bytes into a string
    - readline([size]): read until a newline or up to size bytes, into a string
    - iter(): usually return self, but can be any iterator over lines
    - next(): assuming iter() returns self, this yields the next line.

  In write or append mode:
    - write(s): add string s to the end of the file.
    - writelines(seq): add a sequence of strings

IO managers should also implement:

- a join method, ala ``os.path.join(*args)``.
- a list method, that returns all objects matching location
- a remove method, ala os.remove(path) except that it will remove directories
  as well as files, since we're allowing "locations" to be directories
  or files.
"""

import os
import re
import shutil
from contextlib import contextmanager
from fnmatch import fnmatch

ALLOWED_OPEN_MODES = ('r', 'w', 'a')


class DiskIO:

    """
    Simple IO object to wrap disk operations with a custom base location.

    Also accepts both forward and backward slashes at any point, and
    normalizes both to the OS we are currently on.

    Args:
        base_location (str): a path to the root data folder.
            Converted to an absolute path immediately, so even if you supply a
            relative path, later changes to the OS working directory will not
            affect data paths.
    """

    def __init__(self, base_location):
        if base_location is None:
            self.base_location = None
        else:
            base_location = self._normalize_slashes(base_location)
            self.base_location = os.path.abspath(base_location)

    @contextmanager
    def open(self, filename, mode, encoding=None):
        """
        Mimic the interface of the built in open context manager.

        Args:
            filename (str): path relative to base_location.

            mode (str): 'r' (read), 'w' (write), or 'a' (append).
                Other open modes are not supported because we don't want
                to force all IO managers to support others.

        Returns:
            context manager yielding the open file
        """
        if mode not in ALLOWED_OPEN_MODES:
            raise ValueError(f'mode {mode} not allowed in IO managers')

        filepath = self.to_path(filename)

        # make directories if needed
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        # normally we'd construct this context manager with try/finally, but
        # here we already have a context manager for open so we just wrap it
        with open(filepath, mode, encoding=encoding) as f:
            yield f

    def _normalize_slashes(self, location):
        # note that this is NOT os.path.join - the difference is os.path.join
        # discards empty strings, so if you use it on a re.split absolute
        # path you will get a relative path!
        return os.sep.join(re.split('[\\\\/]', location))

    def to_path(self, location):
        """
        Convert a location string into a path on the local file system.

        For DiskIO this just fixes slashes and prepends the base location,
        doing nothing active with the file. But for other io managers that
        refer to remote storage, this method may actually fetch the file and
        put it at a temporary local path.

        Args:
            location (str): A location string for a complete dataset or
                a file within it.

        Returns:
            str: The path on disk to which this location maps.
        """
        location = self._normalize_slashes(location)
        if self.base_location:
            return os.path.join(self.base_location, location)
        else:
            return location

    def to_location(self, path):
        """
        Convert a local filesystem path into a location string.

        Args:
            path (str): a path on the local file system.

        Returns:
            str: the location string corresponding to this path.
        """
        if self.base_location:
            return os.path.join(self.base_location, path)
        else:
            return path

    def __repr__(self):
        """Show the base location in the repr."""
        return f"<DiskIO, base_location={repr(self.base_location)}>"

    def join(self, *args):
        """Context-dependent os.path.join for this io manager."""
        return os.path.join(*list(map(self._normalize_slashes, args)))

    def isfile(self, location):
        """Check whether this location matches a file."""
        path = self.to_path(location)
        return os.path.isfile(path)

    def list(self, location, maxdepth=1, include_dirs=False):
        """
        Return all files that match location.

        This is either files whose names match up to an arbitrary extension,
        or any files within an exactly matching directory name.

        Args:
            location (str): the location to match.
                May contain the usual path wildcards * and ?

            maxdepth (Optional[int]): maximum levels of directory nesting to
                recurse into looking for files. Default 1.

            include_dirs (Optional[bool]): whether to allow directories in
                the results or just files. Default False.

        Returns:
            A list of matching files and/or directories, as locations
            relative to our base_location.
        """
        location = self._normalize_slashes(location)
        search_dir, pattern = os.path.split(location)
        path = self.to_path(search_dir)

        if not os.path.isdir(path):
            return []

        matches = [fn for fn in os.listdir(path) if fnmatch(fn, pattern + '*')]
        out = []

        for match in matches:
            matchpath = self.join(path, match)
            if os.path.isdir(matchpath) and fnmatch(match, pattern):
                if maxdepth > 0:
                    # exact directory match - walk down to maxdepth
                    for root, dirs, files in os.walk(matchpath, topdown=True):
                        depth = root[len(path):].count(os.path.sep)
                        if depth == maxdepth:
                            dirs[:] = []  # don't recurse any further

                        for fn in files + (dirs if include_dirs else []):
                            out.append(self.to_location(self.join(root, fn)))

                elif include_dirs:
                    out.append(self.join(search_dir, match))

            elif (os.path.isfile(matchpath) and
                  (fnmatch(match, pattern) or
                   fnmatch(os.path.splitext(match)[0], pattern))):
                # exact filename match, or match up to an extension
                # note that we need fnmatch(match, pattern) in addition to the
                # splitext test to cover the case of the base filename itself
                # containing a dot.
                out.append(self.join(search_dir, match))

        return out

    def remove(self, filename):
        """Delete a file or folder and prune the directory tree."""
        path = self.to_path(filename)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

        filepath = os.path.split(path)[0]
        try:
            os.removedirs(filepath)
        except OSError:
            # directory was not empty - good that we're not removing it!
            pass

    def remove_all(self, location):
        """
        Delete all files/directories in the dataset at this location.

        Afterward prunes the directory tree.
        """
        for fn in self.list(location):
            self.remove(fn)
