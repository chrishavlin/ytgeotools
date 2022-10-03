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

    _has_yt = None

    @property
    def has_yt(self):
        if self._has_yt is None:
            try:
                import yt  # noqa: F401

                self._has_yt = True
            except ImportError:
                self._has_yt = False
        return self._has_yt

    def requires(self, module_name, func):
        def wrapper(*args, **kwargs):
            att_name = f"has_{module_name}"
            if getattr(self, att_name):
                return func(*args, **kwargs)
            else:
                raise ImportError(f"This method requires {module_name}.")

        return wrapper


dependency_checker = DependencyChecker()
