class DependencyChecker:
    _has_cartopy = None

    @property
    def has_cartopy(self):
        if self._has_cartopy is None:
            try:
                import cartopy  # noqa: F401

                self._has_cartopy = True
            except ImportError:
                self._has_cartopy = False
        return self._has_cartopy

    def requires_cartopy(self, func):
        def wrapper(*args, **kwargs):
            if self.has_cartopy:
                return func(*args, **kwargs)
            else:
                raise ImportError("This method requires cartopy.")

        return wrapper


dependency_checker = DependencyChecker()
