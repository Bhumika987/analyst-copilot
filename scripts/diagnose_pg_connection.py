"""
Standalone Postgres connectivity diagnostic -- no app code involved, just
asyncpg directly, to isolate exactly where a connection attempt is failing
when the app-level error is a generic timeout.

Usage:
  DATABASE_URL=postgresql://... python scripts/diagnose_pg_connection.py
"""

import asyncio
import os
import ssl
import sys


async def main():
    dsn = os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        print("Set DATABASE_URL first.")
        return

    import asyncpg

    print(f"asyncpg version: {asyncpg.__version__}")
    print(f"Python version: {sys.version}")
    print()

    # Attempt 1: exactly what the app does (DSN's own sslmode, default timeout).
    print("Attempt 1: plain asyncpg.connect(dsn), 10s timeout...")
    try:
        conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=10)
        print("  SUCCESS")
        await conn.close()
        return
    except asyncio.TimeoutError:
        print("  Timed out after 10s (hung, no error from the server).")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")

    # Attempt 2: force IPv4 explicitly, in case IPv6 resolution/routing is
    # the actual dead end (Test-NetConnection and asyncpg can resolve to
    # different address families).
    print("\nAttempt 2: same, but forcing IPv4 resolution...")
    import socket
    try:
        host_part = dsn.split("@")[1].split("/")[0]
        host = host_part.split(":")[0]
        port = int(host_part.split(":")[1].split("?")[0]) if ":" in host_part else 5432
        infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        ipv4_addr = infos[0][4][0]
        print(f"  Resolved {host} -> {ipv4_addr} (IPv4)")
        conn = await asyncio.wait_for(
            asyncpg.connect(dsn, host=ipv4_addr), timeout=10
        )
        print("  SUCCESS via explicit IPv4")
        await conn.close()
        return
    except asyncio.TimeoutError:
        print("  Also timed out over IPv4 -- not an IPv6 routing issue.")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")

    # Attempt 3: a bare TLS handshake with the stdlib ssl module, no
    # Postgres protocol at all -- isolates whether the TLS layer itself
    # (independent of asyncpg/asyncio) can complete against this host.
    print("\nAttempt 3: raw TLS handshake only (stdlib ssl, no Postgres protocol)...")
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        raw_sock = socket.create_connection((host, port), timeout=10)
        tls_sock = ctx.wrap_socket(raw_sock, server_hostname=host)
        print(f"  SUCCESS -- TLS handshake completed, negotiated {tls_sock.version()}")
        tls_sock.close()
    except socket.timeout:
        print("  Raw TLS handshake itself hung/timed out -- this confirms something on the")
        print("  network path is specifically interfering with the TLS handshake, not the")
        print("  Postgres protocol or asyncpg/Python. Try a different network (phone hotspot)")
        print("  to confirm, or contact your network administrator about port 5432 TLS traffic.")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")

    print("\nAll attempts above that succeeded/failed should narrow down the cause.")


if __name__ == "__main__":
    asyncio.run(main())
