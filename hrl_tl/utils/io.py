import importlib
import os


def format_large_number(timesteps: int) -> str:
    if timesteps >= 1_000_000:
        return f"{timesteps / 1_000_000}M"
    elif timesteps >= 1_000:
        return f"{timesteps / 1_000}K"
    else:
        return str(timesteps)


def get_class(cls_name: str | None) -> type | None:
    """Get a class by its name from the module."""
    if cls_name and "." in cls_name:
        module_name, class_name = cls_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    elif cls_name is None:
        return None
    else:
        raise ValueError(
            f"Class name '{cls_name}' must include the module prefix."
        )


def add_number_to_existing_filepath(file_path: str) -> str:
    """
    Add a number to the end of the file name if it already exists.

    Parameters
    ----------
    file_path: str
        The original file path to modify.

    Returns
    -------
    file_path: str
        The modified file path with a number added if it already existed.
    """
    base, ext = os.path.splitext(file_path)
    i = 1
    while os.path.exists(file_path):
        file_path = f"{base}_{i}{ext}"
        i += 1
    return file_path


def get_file_with_largest_number(file_path: str) -> str:
    """
    Get the file path with the largest number suffix.

    Parameters
    ----------
    file_path: str
        The original file path to modify.

    Returns
    -------
    file_path: str
        The modified file path with the largest number suffix.
    """
    base, ext = os.path.splitext(file_path)
    i = 1
    while True:
        new_file_path = f"{base}_{i}{ext}"
        if not os.path.exists(new_file_path):
            return file_path
        else:
            file_path = new_file_path
            i += 1
