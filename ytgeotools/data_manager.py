import os


class filesystemDB(object):
    def __init__(self, dataDir=None):
        if dataDir is None:
            dataDir = os.environ.get("geoSamplerDataDir", None)
            if dataDir is None:
                raise ValueError(
                    (
                        "either dataDir parameter or ",
                        "geoSamplerDataDir environment variable must be set",
                    )
                )

        dataDir = os.path.abspath(dataDir)
        if os.path.isdir(dataDir) is False:
            raise ValueError("dataDir, " + dataDir + ", does not exist")
        else:
            self.dataDir = dataDir

        self.buildFileHash()

        return

    def buildFileHash(self):
        self.fileHash = {}
        for dir, subdirs, files in os.walk(self.dataDir):
            for file in files:
                fp = os.path.join(dir, file)
                self.fileHash[file] = fp

    def fullpath(self, file):
        try:
            fullfi = self.fileHash[file]
        except:
            raise KeyError(
                (
                    "file " + file + " not found in " + self.dataDir,
                    ". If you know it is there, try re-initializing the filesystemDB object.",
                )
            )

        return fullfi
