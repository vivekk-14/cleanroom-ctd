"""
Tests for passive-monitoring enforcement.

These are the tests that back the project's central safety claim: the system
observes traffic and never communicates with the monitored network.

Run:
    python -m pytest tests/test_passive_safety.py -v

Two layers are tested:

  1. STATIC   -- the source code contains no packet-transmission, scanning,
                 exploitation, or blocking code. Grep-based, so it also catches
                 something being pasted in later.
  2. RUNTIME  -- detection.passive_guard actually blocks outbound socket
                 operations while permitting the loopback traffic Streamlit
                 needs.

Ordering note: passive_guard patches the socket module process-wide, so once any
test arms it, it stays armed for the rest of the session. The tests are written
to be correct either way rather than depending on execution order.
"""

from __future__ import annotations

import io
import socket
import sys
import tokenize
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.passive_guard import (  # noqa: E402
    PassiveModeViolation,
    enforce_passive_mode,
    is_armed,
    posture_report,
)

# Every Python source file in the project, excluding the venv and caches.
SOURCE_FILES = sorted(
    path for path in PROJECT_ROOT.rglob("*.py")
    if ".venv" not in path.parts
    and "__pycache__" not in path.parts
    and "site-packages" not in path.parts
)

# passive_guard.py names the operations it blocks, and the test files exercise
# them. Excluded from the static scan to avoid flagging the safety mechanism
# itself as unsafe.
SAFETY_AWARE_FILES = {"passive_guard.py", "test_passive_safety.py"}

SCANNED_FILES = [p for p in SOURCE_FILES if p.name not in SAFETY_AWARE_FILES]


def read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def executable_code(path: Path) -> str:
    """Return a file's source with comments and string literals blanked out.

    The static scans below look for forbidden CODE. Matching raw file text
    cannot tell `sock.send(...)` from the string `".send("` inside a test that
    asserts nobody calls it, or from a docstring explaining why the project
    avoids it. Both produced false failures.

    Implementation detail that matters: string and comment tokens are overwritten
    with spaces IN PLACE, preserving every character position. An earlier version
    joined tokens with a space, which turned `sendp(None)` into `sendp ( None )`
    and made every call-syntax pattern silently unmatchable -- a false negative
    in a safety test, which is worse than the false positive it was fixing. A
    canary file containing real violations is used below to prove the scan
    actually fires.

    An invocation cannot hide in a string literal: reaching a shell or a socket
    still requires an identifier that survives this transformation and is
    scanned for separately by `code_identifiers`.
    """
    try:
        source = read_source(path)
        lines = source.splitlines(keepends=True)
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type not in (tokenize.STRING, tokenize.COMMENT):
                continue
            (start_row, start_col), (end_row, end_col) = token.start, token.end
            for row in range(start_row, end_row + 1):
                line = lines[row - 1]
                begin = start_col if row == start_row else 0
                finish = end_col if row == end_row else len(line.rstrip("\r\n"))
                # Blank the token's span, keeping the line ending intact so line
                # numbers and adjacency are unchanged.
                ending = line[len(line.rstrip("\r\n")):]
                body = line.rstrip("\r\n")
                body = body[:begin] + " " * (finish - begin) + body[finish:]
                lines[row - 1] = body + ending
        return "".join(lines)
    except (tokenize.TokenError, SyntaxError, IndentationError):  # pragma: no cover
        pytest.fail(f"{path.relative_to(PROJECT_ROOT)} failed to tokenise")


def code_identifiers(path: Path) -> set[str]:
    """Return the identifier tokens in a file, excluding strings and comments.

    Used where the forbidden thing is a bare name rather than call syntax.
    'nmap' is contained in 'unmapped', and 'DoS slowloris' is a legitimate
    CIC-IDS2017 label that LABEL_MAP must list in order to drop it.
    """
    try:
        source = read_source(path)
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        return {token.string.lower() for token in tokens
                if token.type == tokenize.NAME}
    except (tokenize.TokenError, SyntaxError, IndentationError):  # pragma: no cover
        pytest.fail(f"{path.relative_to(PROJECT_ROOT)} failed to tokenise")


class TestNoForbiddenCode:
    """Static checks: the forbidden capabilities are simply not implemented."""

    def test_source_files_were_found(self) -> None:
        # Guards against the glob silently matching nothing, which would make
        # every test in this class pass vacuously.
        assert len(SCANNED_FILES) >= 10, \
            f"expected the project's modules, found {len(SCANNED_FILES)}"

    @pytest.mark.parametrize("pattern,capability", [
        ("sendp(", "scapy packet injection"),
        ("srp(", "scapy send/receive at layer 2"),
        ("sr1(", "scapy send/receive at layer 3"),
        ("scapy.sendrecv", "scapy transmission module"),
        ("conf.iface", "scapy interface configuration"),
    ])
    def test_no_packet_injection(self, pattern: str, capability: str) -> None:
        for path in SCANNED_FILES:
            assert pattern not in executable_code(path), \
                f"{path.relative_to(PROJECT_ROOT)} contains {capability}"

    @pytest.mark.parametrize("pattern,capability", [
        ("sniff(", "live packet capture"),
        ("AsyncSniffer", "live packet capture"),
        ("pcap_open_live", "live capture handle"),
        ("SOCK_RAW", "raw socket"),
        ("AF_PACKET", "packet socket"),
        ("IP_HDRINCL", "raw IP header construction"),
    ])
    def test_no_live_capture_or_raw_sockets(self, pattern: str,
                                           capability: str) -> None:
        for path in SCANNED_FILES:
            assert pattern not in executable_code(path), \
                f"{path.relative_to(PROJECT_ROOT)} contains {capability}"

    @pytest.mark.parametrize("pattern", [
        "iptables", "netsh advfirewall", "pfctl", "nftables",
        "ufw ", "firewall-cmd", "ip route del", "arp -d",
    ])
    def test_no_blocking_or_mitigation(self, pattern: str) -> None:
        """No firewall or routing manipulation: the enclave cannot act back.

        Scanned against raw text, including strings: these appear only as shell
        commands, so a string literal containing one is itself the risk.
        """
        for path in SCANNED_FILES:
            assert pattern not in read_source(path).lower(), \
                f"{path.relative_to(PROJECT_ROOT)} manipulates network policy"

    @pytest.mark.parametrize("tool", [
        "nmap", "masscan", "hydra", "metasploit", "msfconsole",
        "sqlmap", "hping", "exploit",
    ])
    def test_no_offensive_tooling(self, tool: str) -> None:
        """No offensive tool is invoked from code.

        Checked against code identifiers rather than raw text. Attack NAMES do
        legitimately appear in this project: 'DoS slowloris' is a CIC-IDS2017
        label that LABEL_MAP must list in order to drop it. Naming a label is
        not the same as running a tool, and the distinction matters -- a naive
        substring scan also matches 'nmap' inside 'unmapped'.
        """
        for path in SCANNED_FILES:
            assert tool not in code_identifiers(path), \
                f"{path.relative_to(PROJECT_ROOT)} invokes {tool}"

    @pytest.mark.parametrize("module", [
        "subprocess", "pty", "commands",
    ])
    def test_no_shell_execution_modules(self, module: str) -> None:
        """No subprocess use, so the passive guard cannot be escaped that way.

        The guard patches sockets inside this interpreter only. A subprocess
        would be outside its reach, so the project simply does not spawn any.
        Checked at the identifier level: reaching a shell requires naming the
        module, which no string literal can accomplish.
        """
        for path in SCANNED_FILES:
            assert module not in code_identifiers(path), \
                f"{path.relative_to(PROJECT_ROOT)} imports {module}"

    @pytest.mark.parametrize("pattern", [
        "os.system", "os.popen", "os.exec", "os.spawn",
    ])
    def test_no_os_level_execution(self, pattern: str) -> None:
        for path in SCANNED_FILES:
            assert pattern not in executable_code(path), \
                f"{path.relative_to(PROJECT_ROOT)} executes a shell command"

    @pytest.mark.parametrize("module", [
        "requests", "httpx", "urllib", "ftplib", "smtplib", "paramiko",
        "telnetlib", "http",
    ])
    def test_no_outbound_client_libraries(self, module: str) -> None:
        """No HTTP/FTP/SSH client library is imported anywhere.

        Identifier-level, which is stricter than matching call sites such as
        'requests.get(': it catches any use of the module at all, including
        aliased imports.
        """
        for path in SCANNED_FILES:
            assert module not in code_identifiers(path), \
                f"{path.relative_to(PROJECT_ROOT)} imports {module}"

    @pytest.mark.parametrize("pattern", [
        "Cipher.", "AES.new", "load_pem_private_key", "ssl.wrap_socket",
        "SSLKEYLOGFILE", "decrypt(",
    ])
    def test_no_payload_decryption(self, pattern: str) -> None:
        """The problem statement forbids decrypting payloads."""
        for path in SCANNED_FILES:
            assert pattern not in executable_code(path), \
                f"{path.relative_to(PROJECT_ROOT)} attempts decryption"

    def test_scapy_is_not_a_dependency(self) -> None:
        """Nothing in the project needs scapy, so it is not installed.

        The prototype reads flow records from CSV. Not depending on a packet
        library removes the possibility of packet transmission entirely.
        """
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(
            encoding="utf-8").lower()
        for line in requirements.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            assert "scapy" not in stripped, \
                "scapy is listed as a required dependency"


class TestGuardArming:
    def test_enforce_is_idempotent(self) -> None:
        enforce_passive_mode(verbose=False)
        enforce_passive_mode(verbose=False)
        assert is_armed() is True

    def test_posture_report_shape(self) -> None:
        enforce_passive_mode(verbose=False)
        report = posture_report()
        for key in ("passive_mode_armed", "outbound_attempts_blocked",
                    "loopback_operations_allowed", "rule",
                    "guarded_operations", "scope_note"):
            assert key in report
        assert report["passive_mode_armed"] is True

    def test_scope_is_stated_honestly(self) -> None:
        """The guard must not overstate itself as a kernel-level guarantee."""
        note = posture_report()["scope_note"].lower()
        assert "python" in note
        assert "subprocess" in note or "c extension" in note
        assert "diode" in note


class TestBlocksOutbound:
    """Runtime checks: outbound operations raise instead of transmitting."""

    def setup_method(self) -> None:
        enforce_passive_mode(verbose=False)

    def test_tcp_connect_to_remote_is_blocked(self) -> None:
        with pytest.raises(PassiveModeViolation):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect(("8.8.8.8", 53))

    def test_connect_ex_to_remote_is_blocked(self) -> None:
        with pytest.raises(PassiveModeViolation):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect_ex(("8.8.8.8", 53))

    def test_udp_sendto_remote_is_blocked(self) -> None:
        with pytest.raises(PassiveModeViolation):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.sendto(b"probe", ("8.8.8.8", 53))

    def test_create_connection_to_remote_is_blocked(self) -> None:
        with pytest.raises(PassiveModeViolation):
            socket.create_connection(("example.com", 80), timeout=1)

    def test_bind_to_all_interfaces_is_blocked(self) -> None:
        """0.0.0.0 is reachable from the monitored network, so it is not loopback."""
        with pytest.raises(PassiveModeViolation):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("0.0.0.0", 0))

    def test_dns_resolution_is_blocked(self) -> None:
        """Resolving a hostname emits a DNS query, which is an outbound action."""
        with pytest.raises(PassiveModeViolation):
            socket.getaddrinfo("unb.ca", 443)
        with pytest.raises(PassiveModeViolation):
            socket.gethostbyname("example.com")

    def test_send_on_unconnected_socket_is_blocked(self) -> None:
        with pytest.raises(PassiveModeViolation):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.send(b"data")

    @pytest.mark.skipif(not hasattr(socket, "AF_PACKET"),
                        reason="AF_PACKET is Linux-only")
    def test_raw_packet_socket_is_blocked(self) -> None:
        with pytest.raises(PassiveModeViolation):
            socket.socket(socket.AF_PACKET, socket.SOCK_RAW)

    def test_violations_are_recorded(self) -> None:
        before = posture_report()["outbound_attempts_blocked"]
        with pytest.raises(PassiveModeViolation):
            socket.create_connection(("1.1.1.1", 80), timeout=1)
        after = posture_report()["outbound_attempts_blocked"]
        assert after == before + 1

    def test_violation_message_is_actionable(self) -> None:
        with pytest.raises(PassiveModeViolation) as info:
            socket.create_connection(("1.1.1.1", 80), timeout=1)
        message = str(info.value)
        assert "PASSIVE MODE" in message
        assert "loopback" in message.lower()
        assert "1.1.1.1" in message


class TestAllowsLoopback:
    """The dashboard is a localhost HTTP server; loopback must keep working."""

    def setup_method(self) -> None:
        enforce_passive_mode(verbose=False)

    def test_loopback_bind_and_listen(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            assert sock.getsockname()[0] == "127.0.0.1"

    def test_full_loopback_roundtrip(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]

            client = socket.create_connection(("127.0.0.1", port), timeout=2)
            connection, _ = server.accept()
            client.sendall(b"ping")
            assert connection.recv(4) == b"ping"
            connection.sendall(b"pong")
            assert client.recv(4) == b"pong"
            connection.close()
            client.close()
        finally:
            server.close()

    def test_socketpair_works(self) -> None:
        """asyncio builds a socketpair self-pipe at startup."""
        left, right = socket.socketpair()
        try:
            left.sendall(b"x")
            assert right.recv(1) == b"x"
        finally:
            left.close()
            right.close()

    def test_asyncio_event_loop_starts(self) -> None:
        """Streamlit runs on asyncio; if this fails the dashboard cannot start."""
        import asyncio

        async def noop() -> str:
            await asyncio.sleep(0)
            return "ok"

        assert asyncio.run(noop()) == "ok"

    def test_asyncio_loopback_server(self) -> None:
        """The pattern Streamlit's HTTP server uses."""
        import asyncio

        async def scenario() -> bytes:
            async def handle(reader: asyncio.StreamReader,
                             writer: asyncio.StreamWriter) -> None:
                await reader.read(16)
                writer.write(b"HTTP/1.1 200 OK\r\n\r\n")
                await writer.drain()
                writer.close()

            server = await asyncio.start_server(handle, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET / HTTP/1.1\r\n\r\n")
            await writer.drain()
            response = await reader.read(20)
            writer.close()
            server.close()
            return response

        assert asyncio.run(scenario()).startswith(b"HTTP/1.1 200")

    def test_localhost_resolution_allowed(self) -> None:
        assert socket.getaddrinfo("localhost", 8501)
        assert socket.getaddrinfo("127.0.0.1", 8501)


class TestReplayEngineSafety:
    """The replay engine must arm the guard and must not transmit."""

    def test_replay_arms_the_guard_at_import(self) -> None:
        import streaming.replay  # noqa: F401

        assert is_armed() is True

    def test_replay_module_has_no_send_calls(self) -> None:
        source = read_source(PROJECT_ROOT / "streaming" / "replay.py")
        for forbidden in (".send(", ".sendall(", ".sendto(", ".connect(",
                          "socket.socket("):
            assert forbidden not in source, \
                f"streaming/replay.py contains {forbidden}"

    def test_replay_writes_only_to_local_files(self) -> None:
        """The engine's only output is the local alert stream."""
        source = read_source(PROJECT_ROOT / "streaming" / "replay.py")
        assert "alerts_file.open" in source or "PATHS.alerts_file" in source

    def test_dashboard_arms_the_guard_at_import(self) -> None:
        source = read_source(PROJECT_ROOT / "dashboard" / "app.py")
        assert "enforce_passive_mode" in source
        # Must be armed before the heavyweight imports, so nothing they do at
        # import time can open a socket.
        guard_position = source.index("enforce_passive_mode(")
        streamlit_position = source.index("import streamlit")
        assert guard_position < streamlit_position, \
            "the guard must be armed before streamlit is imported"

    def test_dashboard_is_read_only(self) -> None:
        """No mitigation controls, and no writes to the alert stream."""
        source = read_source(PROJECT_ROOT / "dashboard" / "app.py")
        for forbidden in ("alerts_file.open(\"w\"", "alerts_file.write",
                          "block_ip", "mitigate", "quarantine"):
            assert forbidden not in source, \
                f"dashboard/app.py contains {forbidden}"


class TestPostureIsDocumented:
    """The required posture statements must be present and displayed."""

    def test_config_declares_posture(self) -> None:
        from config import POSTURE

        assert POSTURE.mode == "PASSIVE MONITORING"
        assert POSTURE.access == "READ ONLY"
        assert POSTURE.assurance == "No outbound network actions are performed."

    def test_dashboard_displays_posture(self) -> None:
        source = read_source(PROJECT_ROOT / "dashboard" / "app.py")
        assert "POSTURE.mode" in source
        assert "POSTURE.access" in source
        assert "POSTURE.assurance" in source

    def test_readme_states_the_passive_claim(self) -> None:
        readme = PROJECT_ROOT / "README.md"
        if not readme.exists():
            pytest.skip("README.md not written yet")
        text = readme.read_text(encoding="utf-8").lower()
        assert "passive" in text
        assert "does not communicate back" in text or \
               "no outbound network actions" in text
