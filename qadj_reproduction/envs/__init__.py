from functools import partial
import sys
import os
import socket

from envs.multiagentenv import MultiAgentEnv
from envs.mpe_wrapper import MPEEnvWrapper


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


def _disable_local_proxy_for_sc2():
    # PySC2 talks to the local SC2 process over websocket on 127.0.0.1.
    # If shell-level proxy vars leak into this process, websocket-client may
    # try to route local traffic through the proxy and reconnects can fail.
    for proxy_var in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        os.environ.pop(proxy_var, None)

    localhost_no_proxy = "127.0.0.1,localhost"
    os.environ["NO_PROXY"] = localhost_no_proxy
    os.environ["no_proxy"] = localhost_no_proxy


def sc2_env_fn(**kwargs) -> MultiAgentEnv:
    import portpicker
    from smac.env import StarCraft2Env

    def _fallback_pick_unused_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return sock.getsockname()[1]

    portpicker.pick_unused_port = _fallback_pick_unused_port
    if hasattr(portpicker, "_pick_unused_port_without_server"):
        portpicker._pick_unused_port_without_server = _fallback_pick_unused_port

    if "difficulty" in kwargs and kwargs["difficulty"] is not None:
        kwargs["difficulty"] = str(kwargs["difficulty"])
    _disable_local_proxy_for_sc2()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    os.environ["SC2PATH"] = os.path.join(repo_root, "3rdparty", "StarCraftII")
    return StarCraft2Env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = sc2_env_fn
REGISTRY["mpe"] = partial(env_fn, env=MPEEnvWrapper)

if sys.platform == "linux":
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    _disable_local_proxy_for_sc2()
    os.environ.setdefault("SC2PATH",
                          os.path.join(repo_root, "3rdparty", "StarCraftII"))
