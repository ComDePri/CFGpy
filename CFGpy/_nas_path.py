import os


def get_nas_path():
    default_nas_path = r"\\132.64.186.144\HartLabNAS"
    default_nas_path = r"/Volumes/HartLabNAS"
    environment_variable_name = "NAS_PATH"

    if os.path.isdir(default_nas_path):
        return default_nas_path
    try:
        env_var = os.environ[environment_variable_name]  # may raise KeyError
        if os.path.isdir(env_var):
            return env_var

        error_msg = f"Path indicated by the NAS_PATH environment variable does not exist: {env_var}"

    except KeyError:  # environment variable doesn't exist
        error_msg = f"Lab NAS has to be mapped to {default_nas_path} or to a path specified by the NAS_PATH " \
                    f"environment variable.\nFor details, see https://comdepri.slab.com/posts/connecting-to-nas-ir4z367g"

    raise NotADirectoryError(error_msg)


NAS_PATH = get_nas_path()
