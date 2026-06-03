"""Custom exceptions used by the image recognition project."""


class ModelLoadException(Exception):
    """Base exception for model loading failures."""


class ModelNotFoundError(ModelLoadException):
    """Raised when a local model file cannot be found."""


class ModelDownloadError(ModelLoadException):
    """Raised when pretrained weights cannot be loaded or downloaded."""


class ImageLoadException(Exception):
    """Raised when an input image cannot be loaded."""


class DataDirectoryNotFoundError(Exception):
    """Raised when the expected data/train or data/val directory is missing."""


class ClassCountMismatchError(Exception):
    """Raised when the discovered class count does not match configuration."""
