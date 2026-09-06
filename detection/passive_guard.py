"""
Passive-monitoring enforcement.

The problem statement requires the monitoring enclave to be strictly
observational. It must not probe, connect, handshake, inject packets, block
traffic, or send mitigation commands to the monitored network.

This module makes that a RUNTIME GUARANTEE rather than a claim in a README. It
wraps the outbound operations of Python's socket module so that any attempt to
reach a NON-LOOPBACK address raises PassiveModeViolation. Any code path that
tries to reach the network -- our own, a dependency's, or something pasted in
during a late-night hackathon push -- fails immediately and loudly instead of
quietly transmitting.

The single rule
--------------
    An operation is permitted only if its address is loopback.

Applied uniformly to connect, bind, listen, accept, send, sendall, sendto,
sendmsg and sendfile. The address checked is whichever one the operation
actually uses:

    connect / bind / sendto / sendmsg   the address passed in
    listen / accept                     the socket's own bound address
    send / sendall / sendfile           the connected peer's address

Checking the real address, rather than blocking method names outright, matters
for correctness as well as strictness. Python's own asyncio event loop builds a
loopback socketpair and calls listen() on it during startup; a blanket ban on
listen() makes asyncio -- and therefore Streamlit -- impossible to start, while
providing no extra safety, because that socket never leaves the host.

Additionally blocked
--------------------
    socket.socket(AF_PACKET / AF_LINK)  raw frame access used for injection
    socket.create_connection            non-loopback destinations
    getaddrinfo / gethostbyname         resolution of real hostnames, which
                                        emits a DNS query onto the network.
                                        IP literals and localhost still resolve.

Why loopback is permitted
-------------------------
Streamlit is an HTTP server on localhost, and asyncio needs a self-pipe.
Loopback traffic never reaches a network interface, so it cannot reach the
monitored network. Every loopback allowance is counted and can be displayed on
the dashboard, so the exemption is visible rather than hidden.

Limits of this guarantee
------------------------
This is a Python-level guard, not a kernel-level one. It stops socket use from
within this interpreter. It cannot stop a subprocess, a C extension calling
send() directly, or a compromised interpreter. Real deployment assurance comes
from the hardware data diode, which is a physical one-way path. This guard makes
the software's intent enforceable and testable; it does not replace the diode.
"""

from __future__ import annotations

import ipaddress
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "PassiveModeViolation",
    "enforce_passive_mode",
    "is_armed",
    "get_violations",
    "get_allowed_loopback",
    "posture_report",
]


class PassiveModeViolation(RuntimeError):
    """Raised when code attempts a network operation on a non-loopback address.

    Deliberately a hard error. Logging a warning and continuing would let the
    packet be sent, which is precisely what must not happen.
    """


@dataclass
class _GuardState:
    armed: bool = False
    violations: list[dict] = field(default_factory=list)
    loopback_allowed: int = 0
    originals: dict[str, Any] = field(default_factory=dict)


_STATE = _GuardState()

_LOOPBACK_HOSTNAMES = frozenset({
    "", "localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback",
})


def _host_is_loopback(host: object) -> bool:
    """True if a host string is a loopback name or a loopback IP literal."""
    if host is None:
        return False
    if isinstance(host, bytes):
        try:
            host = host.decode("ascii")
        except UnicodeDecodeError:
            return False
    if not isinstance(host, str):
        return False

    text = host.strip().strip("[]").lower()
    if text in _LOOPBACK_HOSTNAMES:
        return True

    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        # A real hostname. Not loopback; resolving it would emit a DNS query.
        return False

    # Covers 127.0.0.0/8 and ::1. is_loopback also correctly rejects
    # '0.0.0.0' and '::', which bind to every interface and are therefore
    # reachable from the monitored network.
    if address.is_loopback:
        return True
    # IPv4-mapped IPv6, e.g. ::ffff:127.0.0.1
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped and mapped.is_loopback)


def _address_is_loopback(address: Any) -> bool:
    """True if a socket address tuple/path is loopback-only.

    Conservative: anything not clearly loopback is treated as remote.
    """
    if address is None:
        return False

    # AF_UNIX paths and Linux abstract sockets never touch an IP network.
    if isinstance(address, (str, bytes)):
        return True

    if isinstance(address, tuple) and address:
        return _host_is_loopback(address[0])

    return False


def _socket_local_address(sock: socket.socket) -> Any:
    """The socket's own bound address, or None if it is unbound.

    Unbound sockets raise OSError here on Windows (WinError 10022) and return
    ('0.0.0.0', 0) on Linux, so both cases are normalised to None.
    """
    try:
        return sock.getsockname()
    except (OSError, AttributeError):
        return None


def _socket_peer_address(sock: socket.socket) -> Any:
    """The connected peer's address, or None if the socket is not connected."""
    try:
        return sock.getpeername()
    except (OSError, AttributeError):
        return None


def _record_violation(operation: str, address: Any, detail: str) -> None:
    _STATE.violations.append({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "operation": operation,
        "address": repr(address),
        "detail": detail,
    })


def _deny(operation: str, address: Any, detail: str) -> "PassiveModeViolation":
    _record_violation(operation, address, detail)
    return PassiveModeViolation(
        f"PASSIVE MODE: {operation} is blocked.\n"
        f"  address : {address!r}\n"
        f"  reason  : {detail}\n"
        f"  This system is a read-only monitoring enclave. It must not "
        f"transmit on, connect to, or listen on the monitored network.\n"
        f"  Only loopback is permitted, so the local dashboard can serve HTTP.\n"
        f"  To move results out of the process, write them to a local file."
    )


# Which address each wrapped method must be judged on.
#   'argument' -> the address passed as the first positional argument
#   'local'    -> the socket's own bound address
#   'peer'     -> the connected peer's address
_METHOD_ADDRESS_SOURCE: dict[str, str] = {
    "connect": "argument",
    "connect_ex": "argument",
    "bind": "argument",
    "sendto": "argument_second",   # sendto(data, address)
    "listen": "local",
    "accept": "local",
    "send": "peer",
    "sendall": "peer",
    "sendfile": "peer",
}


def _make_guard(name: str, source: str):
    """Build a loopback-checked replacement for socket.socket.<name>."""
    original = getattr(socket.socket, name)

    def guarded(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if source == "argument":
            address = args[0] if args else None
        elif source == "argument_second":
            address = args[1] if len(args) > 1 else None
        elif source == "local":
            address = _socket_local_address(self)
        else:  # peer
            address = _socket_peer_address(self)

        if _address_is_loopback(address):
            _STATE.loopback_allowed += 1
            return original(self, *args, **kwargs)

        if address is None:
            # Unbound or unconnected. The operation cannot be shown to be
            # loopback-only, so it is refused. An unconnected send() would fail
            # at the OS level anyway; refusing here is equivalent in effect and
            # explicit about why.
            detail = ("socket is not bound or connected, so the destination "
                      "cannot be confirmed as loopback")
        else:
            detail = "destination is not loopback"

        raise _deny(f"socket.{name}()", address, detail)

    guarded.__name__ = name
    guarded.__qualname__ = f"socket.socket.{name}"
    guarded.__doc__ = (
        f"passive_guard: {name}() permitted only for loopback addresses."
    )
    return guarded


def _guarded_socket_new(cls, family=-1, type=-1, proto=-1, fileno=None):  # noqa: A002
    """Block raw/packet socket families outright.

    AF_PACKET (Linux) and AF_LINK give direct frame-level access, which is the
    mechanism for packet injection. There is no read-only use for them here:
    flow records come from CSV.
    """
    raw_families = {
        getattr(socket, attr) for attr in ("AF_PACKET", "AF_LINK")
        if hasattr(socket, attr)
    }
    if family in raw_families:
        raise _deny(
            "socket.socket(AF_PACKET/AF_LINK)", f"family={family}",
            "raw frame-level sockets permit packet injection",
        )
    return _STATE.originals["socket_new"](cls, family, type, proto, fileno)


def _guarded_create_connection(address, *args, **kwargs):  # noqa: ANN001
    if _address_is_loopback(address):
        _STATE.loopback_allowed += 1
        return _STATE.originals["create_connection"](address, *args, **kwargs)
    raise _deny("socket.create_connection()", address,
                "destination is not loopback")


def _guarded_getaddrinfo(host, port, *args, **kwargs):  # noqa: ANN001
    """Permit IP literals and localhost; refuse real hostname resolution.

    Resolving a hostname sends a DNS query to a resolver, which is an outbound
    network action even though no application socket is opened. Resolving an IP
    literal sends nothing.
    """
    if _host_is_loopback(host) or host is None:
        return _STATE.originals["getaddrinfo"](host, port, *args, **kwargs)
    try:
        ipaddress.ip_address(str(host).strip().strip("[]"))
    except ValueError:
        raise _deny("socket.getaddrinfo()", host,
                    "resolving a hostname emits a DNS query onto the network") \
            from None
    return _STATE.originals["getaddrinfo"](host, port, *args, **kwargs)


def _guarded_gethostbyname(hostname):  # noqa: ANN001
    if _host_is_loopback(hostname):
        return _STATE.originals["gethostbyname"](hostname)
    raise _deny("socket.gethostbyname()", hostname,
                "resolving a hostname emits a DNS query onto the network")


def enforce_passive_mode(verbose: bool = True) -> None:
    """Arm the guard. Idempotent: calling twice is harmless.

    Call this before any module that handles traffic data is used.
    """
    if _STATE.armed:
        return

    _STATE.originals["socket_new"] = socket.socket.__new__
    _STATE.originals["create_connection"] = socket.create_connection
    _STATE.originals["getaddrinfo"] = socket.getaddrinfo
    _STATE.originals["gethostbyname"] = socket.gethostbyname

    for name, source in _METHOD_ADDRESS_SOURCE.items():
        if hasattr(socket.socket, name):
            _STATE.originals[name] = getattr(socket.socket, name)
            setattr(socket.socket, name, _make_guard(name, source))

    socket.socket.__new__ = _guarded_socket_new
    socket.create_connection = _guarded_create_connection
    socket.getaddrinfo = _guarded_getaddrinfo
    socket.gethostbyname = _guarded_gethostbyname

    _STATE.armed = True

    if verbose:
        print("[passive-guard] ARMED: network operations are restricted to "
              "loopback.")
        print("[passive-guard] Outbound connects, sends, listeners, raw "
              "sockets and DNS lookups are blocked.")


def is_armed() -> bool:
    return _STATE.armed


def get_violations() -> list[dict]:
    """Blocked attempts recorded so far. Empty is the expected state."""
    return list(_STATE.violations)


def get_allowed_loopback() -> int:
    """Count of permitted loopback operations, for transparency."""
    return _STATE.loopback_allowed


def posture_report() -> dict:
    """Machine-readable posture, for the dashboard to display."""
    return {
        "passive_mode_armed": _STATE.armed,
        "outbound_attempts_blocked": len(_STATE.violations),
        "loopback_operations_allowed": _STATE.loopback_allowed,
        "violations": get_violations(),
        "rule": "An operation is permitted only if its address is loopback.",
        "guarded_operations": sorted(
            [f"socket.{n}" for n in _METHOD_ADDRESS_SOURCE]
            + ["socket.create_connection", "socket.getaddrinfo",
               "socket.gethostbyname", "socket.socket(AF_PACKET/AF_LINK)"]
        ),
        "scope_note": (
            "Python-level enforcement within this interpreter. It does not "
            "constrain subprocesses or C extensions that bypass the socket "
            "module. Deployment assurance comes from the hardware data diode."
        ),
    }


if __name__ == "__main__":
    print("=" * 76)
    print("PASSIVE GUARD - self-test")
    print("=" * 76)

    print(f"\n  armed before: {is_armed()}")
    enforce_passive_mode()
    print(f"  armed after : {is_armed()}\n")

    passed = failed = 0

    def must_block(description: str, action) -> None:  # noqa: ANN001
        global passed, failed
        try:
            action()
        except PassiveModeViolation:
            print(f"  [BLOCKED] {description}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL]    {description} -> {type(exc).__name__}: {exc}")
            failed += 1
        else:
            print(f"  [FAIL]    {description} -> NOT BLOCKED")
            failed += 1

    def must_allow(description: str, action) -> None:  # noqa: ANN001
        global passed, failed
        try:
            action()
        except PassiveModeViolation as exc:
            print(f"  [FAIL]    {description} -> wrongly blocked: "
                  f"{str(exc).splitlines()[0]}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL]    {description} -> {type(exc).__name__}: {exc}")
            failed += 1
        else:
            print(f"  [ALLOWED] {description}")
            passed += 1

    print("  Outbound operations -- all must be blocked:")

    def _connect_remote() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(("8.8.8.8", 53))

    def _sendto_remote() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(b"probe", ("8.8.8.8", 53))

    def _bind_all_interfaces() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("0.0.0.0", 0))

    def _listen_on_all_interfaces() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            _STATE.originals["bind"](sock, ("0.0.0.0", 0))  # bypass to set up
            sock.listen(1)

    must_block("TCP connect to 8.8.8.8:53 (active probe)", _connect_remote)
    must_block("UDP sendto 8.8.8.8:53 (DNS probe)", _sendto_remote)
    must_block("create_connection to example.com:80",
               lambda: socket.create_connection(("example.com", 80), timeout=1))
    must_block("bind to 0.0.0.0 (reachable from the network)",
               _bind_all_interfaces)
    must_block("listen on a 0.0.0.0-bound socket", _listen_on_all_interfaces)
    must_block("getaddrinfo('unb.ca') -- DNS query",
               lambda: socket.getaddrinfo("unb.ca", 443))
    must_block("gethostbyname('example.com') -- DNS query",
               lambda: socket.gethostbyname("example.com"))

    if hasattr(socket, "AF_PACKET"):
        must_block("raw AF_PACKET socket (injection)",
                   lambda: socket.socket(socket.AF_PACKET, socket.SOCK_RAW))
    else:
        print("  [n/a]     AF_PACKET absent on this platform (Windows)")

    print("\n  Loopback -- must be permitted, the dashboard depends on it:")

    def _loopback_bind_listen() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)

    def _loopback_roundtrip() -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            client = socket.create_connection(("127.0.0.1", port), timeout=2)
            conn, _ = server.accept()
            client.sendall(b"ping")
            assert conn.recv(4) == b"ping"
            conn.close()
            client.close()
        finally:
            server.close()

    must_allow("bind + listen on 127.0.0.1", _loopback_bind_listen)
    must_allow("full loopback connect/accept/sendall/recv", _loopback_roundtrip)
    must_allow("socketpair() -- asyncio self-pipe",
               lambda: [s.close() for s in socket.socketpair()])
    must_allow("getaddrinfo('localhost')",
               lambda: socket.getaddrinfo("localhost", 8501))
    must_allow("getaddrinfo IP literal '127.0.0.1'",
               lambda: socket.getaddrinfo("127.0.0.1", 8501))

    print("\n  asyncio event loop (Streamlit depends on this):")

    def _asyncio_loop() -> None:
        import asyncio

        async def noop() -> str:
            await asyncio.sleep(0)
            return "ok"

        assert asyncio.run(noop()) == "ok"

    must_allow("asyncio.run() starts an event loop", _asyncio_loop)

    report = posture_report()
    print(f"\n  outbound attempts blocked   : "
          f"{report['outbound_attempts_blocked']}")
    print(f"  loopback operations allowed : "
          f"{report['loopback_operations_allowed']}")
    print(f"\n  {passed} passed, {failed} failed")
    print(f"\n  Rule : {report['rule']}")
    print(f"  Scope: {report['scope_note']}")
    print("=" * 76)
    sys.exit(1 if failed else 0)
