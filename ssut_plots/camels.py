from pathlib import Path


root = Path("/mnt/ceph/users/camels/Sims/")

oneP_sims = [f"1P_p{n}_{v}" for n in range(1, 7) for v in ["n2", "n1", "0", "1", "2"]]


SIMBA = {"1P": {k: root / "SIMBA/1P" / k for k in oneP_sims}}
