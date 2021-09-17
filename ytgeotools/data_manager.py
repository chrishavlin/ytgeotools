import os
from typing import Union

from yt.config import ytcfg

_envvar = "YTGEOTOOLSDIR"


def join_then_check_path(filename: str, dirname: str) -> Union[str, None]:
    newname = os.path.join(dirname, filename)
    if os.path.isfile(newname):
        return newname
    return None


class DataManager:
    def __init__(self, priority=["fullpath", "envvar", "ytconfig"]):
        """
        A file manager class

        Parameters
        ----------
        priority: list
            contains the priority order for filename location, used to determine
            what filename to return if it exists in multiple locations. Priority
            labels are defined as:

            "fullpath"  : the file exists in the immediate relative or absolute path
            "envvar"    : the file exists relative to the directory set by
                          the YTGEOTOOLSDIR environment variable
            "ytconfig"  : the file exists relative to the directory set by
                          the test_data_dir parameter in the yt configuration file

            default order is ["fullpath", "envvar", "ytconfig"]
        """
        self.envvar_dir = os.environ.get(_envvar, None)
        tdd = ytcfg.get("yt", "test_data_dir")
        if tdd == "/does/not/exist":
            self.yt_test_data_dir = None
        else:
            self.yt_test_data_dir = tdd

        self.priority = priority

    def fullpath(self, filename: str) -> Union[str, None]:
        if os.path.isfile(os.path.abspath(filename)):
            return os.path.abspath(filename)
        return None

    def check_location(self, filename: str, location: str):

        if location == "fullpath":
            if os.path.isfile(os.path.abspath(filename)):
                return os.path.abspath(filename)
        elif location == "ytconfig":
            return join_then_check_path(filename, self.yt_test_data_dir)
        elif location == "envvar":
            return join_then_check_path(filename, self.envvar_dir)
        return None

    def validate_file(self, filename: str) -> str:
        """
        checks for existence of a file, returns an absolute path.
        Parameters
        ----------
        filename: str
            the filename string to check for

        Returns
        -------
        str, None
            returns the validated filename or None if it does not exist.

        Note:
        this function uses the ytgeotools.data_manager.data_manager object to check
        for filename in relative and absolute paths but also as paths relative to the
        directory set by the YTGEOTOOLSDIR environment variable and in the test_data_dir
        directory set by the yt configuration file. If the filename exists in multiple
        locations, the return priority is set by data_manager.priority

        """
        file_location = [self.check_location(filename, p) for p in self.priority]

        for fname in file_location:
            if fname:
                return fname

        raise FileNotFoundError(
            f"Could not find {filename}. Checked relative and absolute"
            f" paths as well as relative paths from the {_envvar} environment"
            f" variable and `test_data_directory` from the yt config file."
        )


data_manager = DataManager()
