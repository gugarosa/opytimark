"""Base class for benchmark functions."""

import opytimark.utils.exception as e


class _Missing:
    def __repr__(self):
        return "<default>"


_MISSING = _Missing()


def _validated_property(name, expected_type, type_message, value_check=None):
    private_name = f"_{name}"

    def getter(instance):
        return getattr(instance, private_name)

    def setter(instance, value):
        if not isinstance(value, expected_type):
            raise e.TypeError(type_message)
        if value_check and not value_check(value):
            raise e.ValueError(f"`{name}` should be >= -1 and different than 0")
        setattr(instance, private_name, value)

    return property(getter, setter)


class Benchmark:
    """Base class for callable benchmark functions."""

    _defaults = ("Benchmark", 1, False, False, False, False, False)

    name = _validated_property("name", str, "`name` should be a string")
    dims = _validated_property(
        "dims",
        int,
        "`dims` should be a integer",
        lambda value: value >= -1 and value != 0,
    )
    continuous = _validated_property(
        "continuous", bool, "`continuous` should be a boolean"
    )
    convex = _validated_property("convex", bool, "`convex` should be a boolean")
    differentiable = _validated_property(
        "differentiable", bool, "`differentiable` should be a boolean"
    )
    multimodal = _validated_property(
        "multimodal", bool, "`multimodal` should be a boolean"
    )
    separable = _validated_property(
        "separable", bool, "`separable` should be a boolean"
    )

    def __init__(
        self,
        name=_MISSING,
        dims=_MISSING,
        continuous=_MISSING,
        convex=_MISSING,
        differentiable=_MISSING,
        multimodal=_MISSING,
        separable=_MISSING,
    ):
        supplied = (
            name,
            dims,
            continuous,
            convex,
            differentiable,
            multimodal,
            separable,
        )
        for field, value, default in zip(
            (
                "name",
                "dims",
                "continuous",
                "convex",
                "differentiable",
                "multimodal",
                "separable",
            ),
            supplied,
            self._defaults,
        ):
            setattr(self, field, default if value is _MISSING else value)

    def __call__(self, x):
        raise NotImplementedError
