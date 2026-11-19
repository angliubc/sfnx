#!/usr/bin/env python3
"""TCP relay (1110): sfnx-gw@24.23 -> local -> every pooled sfnx (qwen3.8-flash-next) machine.

  18093 -> <FLEET_HOST>:8093   atlas2 instance A (merged in from relay_a2.py, 1110 quiet window)
  18094 -> <FLEET_HOST>:8094   atlas2 instance B
  18095 -> <FLEET_HOST>:8093   atlas1 instance A (newly deployed)
  18096 -> <FLEET_HOST>:8094   atlas1 instance B
  18097 -> <FLEET_HOST>:8093   atlas8 instance A (recalled from the retired -p40 line)
  18098 -> <FLEET_HOST>:8094   atlas8 instance B

Needed because 24.23 has no direct route to <FLEET_HOST>/24 (only via this box).
This is now the single path for the whole pool: systemd --user unit relay-sfnx.service,
enabled with lingering on, so a reboot of this box self-heals all six backends.
Restarting it drops in-flight streams — check first:
  ss -tn state established '( sport = :18093 or sport = :18094 or sport = :18095 or sport = :18096 or sport = :18097 or sport = :18098 )'
relay_a2.py is retired (kept for reference); its cold unit relay-a2.service is masked so
nobody re-enables it into a port conflict."""
import asyncio

PORTS = {
    18095: ("<FLEET_HOST>", 8093),
}
LOG = "/home/ang/relay_sfnx.log"


def _log(line):
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _close(*objs):
    for o in objs:
        try:
            o.close()
        except Exception:
            pass


async def pipe(src, dst):
    try:
        while True:
            data = await src.read(65536)
            if not data:
                break
            dst.write(data)
            await dst.drain()
    except Exception:
        pass
    finally:
        _close(src, dst)


async def handle(creader, cwriter, lp, target):
    host, tp = target
    try:
        ureader, uwriter = await asyncio.wait_for(
            asyncio.open_connection(host, tp), timeout=10)
    except Exception as e:
        _log(f"upstream fail {lp}->{host}:{tp}: {e}")
        _close(cwriter)
        return
    _log(f"conn {lp}->{host}:{tp}")
    await asyncio.gather(pipe(creader, uwriter), pipe(ureader, cwriter))


async def main():
    servers = []
    for lp, target in PORTS.items():
        srv = await asyncio.start_server(
            lambda r, w, lp=lp, target=target: handle(r, w, lp, target),
            host="0.0.0.0", port=lp, reuse_address=True)
        servers.append(srv)
        print(f"relay 0.0.0.0:{lp} -> {target[0]}:{target[1]}", flush=True)
    await asyncio.gather(*(s.serve_forever() for s in servers))


if __name__ == "__main__":
    asyncio.run(main())
