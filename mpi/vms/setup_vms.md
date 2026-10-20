%%writefile /content/drive/MyDrive/TP-Cent/mpi/setup_vms.md
This is the complete, step-by-step deployment guide for **Version 2 (Virtual Machines)** from the project specification.

To save time, we will **configure the Master VM completely first** (installing all software and libraries), and then **clone it to create the Worker/Slave VM**.

---

### Cluster Architecture Plan

| Node Name | Role | Static IP | vCPUs | RAM |
| :--- | :--- | :--- | :--- | :--- |
| `mpi-master` | Master / Rank 0 | `192.168.10.21` | 2 | 2 GB |
| `mpi-slave1` | Worker / Rank 1 | `192.168.10.22` | 2 | 2 GB |

* **User account (mandatory rule for MPI):** Both VMs must have the **exact same username** (e.g., `mpiuser`).
* **Directory structure:** Both VMs will keep project files in the **exact same path** (`/home/mpiuser/hpc_project`).

---

# Phase 1: VirtualBox Installation & Base VM Setup

### 1. Download Tools
1. Download and install **[VirtualBox](https://www.virtualbox.org/)** (and its Extension Pack).
2. Download **[Ubuntu Server 22.04 LTS (or 24.04 LTS) ISO](https://ubuntu.com/download/server)**.

### 2. Create the Master VM (`mpi-master`)
1. Open VirtualBox $\rightarrow$ Click **New**.
2. **Name:** `mpi-master`
3. **Type:** Linux | **Version:** Ubuntu (64-bit).
4. **ISO Image:** Select your downloaded Ubuntu Server ISO.
5. Check **Skip Unattended Installation** (so we can manually set username and disk).
6. **Hardware:**
   * **Base Memory:** `2048 MB` (or `4096 MB` if your host PC has 16GB+ RAM).
   * **Processors:** `2 CPUs`.
7. **Hard Disk:** `20 GB` (Dynamically Allocated).
8. Click **Finish**.

### 3. Configure VM Network Card (Bridged Mode)
*Before powering on the VM:*
1. Select `mpi-master` $\rightarrow$ Click **Settings** $\rightarrow$ **Network**.
2. Adapter 1:
   * Attached to: **Bridged Adapter**
   * Name: Select your physical network adapter (Ethernet preferred, as stated in Section 5).
   * Advanced $\rightarrow$ Promiscuous Mode: **Allow All**.
3. Click **OK**.

*(Note: If you are at home without a switch or router, choose **Host-only Adapter** instead. Both Master and Slave will still communicate seamlessly).*

---

# Phase 2: Ubuntu Server Installation

1. Start `mpi-master`.
2. Select your language and keyboard layout.
3. Choose **Ubuntu Server (minimized)** when prompted (boots faster, consumes less RAM).
4. **Network connections:** Leave DHCP for now (we set static IPs via netplan later).
5. **Storage configuration:** Use default entire disk.
6. **Profile Setup (IMPORTANT):**
   * **Your name:** `mpiuser`
   * **Your server's name:** `mpi-master`
   * **Pick a username:** `mpiuser`
   * **Choose a password:** `mpi123` (choose something simple, remember it).
7. **SSH Setup:** Check **[X] Install OpenSSH server** (CRITICAL).
8. Skip featured server snaps $\rightarrow$ Click **Done** and let the installation complete.
9. When finished, select **Reboot Now** (press Enter to unmount the ISO if asked).

---

# Phase 3: Install MPI, Python, and Dependencies (On Master)

Log in as `mpiuser` in the VM terminal, then run:

### 1. Update System & Install Compiler Tools
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential gcc gfortran libopenmpi-dev openmpi-bin openmpi-common git curl
```

### 2. Install Python 3, Pip, and Virtual Environment
```bash
sudo apt install -y python3 python3-pip python3-venv python3-dev
```

### 3. Set up the Python Environment
```bash
# Create project folder
mkdir -p /home/mpiuser/hpc_project
cd /home/mpiuser/hpc_project

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required HPC libraries inside venv
pip install --upgrade pip
pip install numpy pandas matplotlib mpi4py
```

### 4. Verify Open MPI Installation
Run:
```bash
mpirun --version
python3 -c "from mpi4py import MPI; print('MPI Vendor:', MPI.get_vendor())"
```
You should see: `MPI Vendor: ('Open MPI', ...)` without errors.

---

# Phase 4: Configure Static IP & Hostnames

### 1. Set Static IP for `mpi-master`
Find your network interface name using:
```bash
ip link
```
*(Common names are `enp0s3` or `eth0`). Assume it is `enp0s3`.*

Edit the Netplan config:
```bash
sudo nano /etc/netplan/00-installer-config.yaml
```
Replace the content with the following (adjust interface name if not `enp0s3`):
```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp0s3:
      dhcp4: no
      addresses:
        - 192.168.10.21/24
      routes:
        - to: default
          via: 192.168.10.1
      nameservers:
        addresses: [8.8.8.8, 1.1.1.1]
```
Apply the configuration:
```bash
sudo netplan apply
```
Verify the IP:
```bash
ip addr show enp0s3
```
It should show `inet 192.168.10.21/24`.

---

# Phase 5: Clone the VM to Create `mpi-slave1`

Because all dependencies are already installed, we can clone `mpi-master` to create the worker node.

1. **Shut down `mpi-master`:**
   ```bash
   sudo poweroff
   ```
2. In VirtualBox:
   * Right-click `mpi-master` $\rightarrow$ Click **Clone...**
   * **Name:** `mpi-slave1`
   * **MAC Address Policy:** Select **Generate new MAC addresses for all network adapters** (CRITICAL!).
   * **Clone type:** **Full clone** $\rightarrow$ Click **Finish**.

### Configure `mpi-slave1`
1. Start **`mpi-slave1`** in VirtualBox and log in as `mpiuser`.
2. **Change its Hostname:**
   ```bash
   sudo hostnamectl set-hostname mpi-slave1
   ```
3. **Change its Static IP to `.22`:**
   ```bash
   sudo nano /etc/netplan/00-installer-config.yaml
   ```
   Change `192.168.10.21/24` to:
   ```yaml
         addresses:
           - 192.168.10.22/24
   ```
   Apply it:
   ```bash
   sudo netplan apply
   ```
4. Start **`mpi-master`** so **both VMs are running simultaneously**.

---

# Phase 6: DNS Resolution & Passwordless SSH Setup

MPI needs to connect between nodes silently without prompting for a password.

### 1. Update `/etc/hosts` on BOTH VMs
Run this command on **BOTH** `mpi-master` and `mpi-slave1`:
```bash
sudo nano /etc/hosts
```
Add these lines at the bottom:
```text
192.168.10.21 mpi-master
192.168.10.22 mpi-slave1
```

Test network reachability from `mpi-master`:
```bash
ping -c 3 mpi-slave1
```

---

### 2. Configure SSH Keys (From `mpi-master`)
Open a terminal on **`mpi-master`**:

1. Generate SSH key pair (press **Enter** to accept all defaults; do **not** set a passphrase):
   ```bash
   ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
   ```

2. Copy the key to **itself** (master must SSH to master passwordlessly):
   ```bash
   ssh-copy-id mpiuser@mpi-master
   ```
   *(Type `yes` and enter `mpi123`).*

3. Copy the key to **the slave**:
   ```bash
   ssh-copy-id mpiuser@mpi-slave1
   ```
   *(Type `yes` and enter `mpi123`).*

4. **Verify Passwordless Access:**
   Test logging into the slave without a password:
   ```bash
   ssh mpi-slave1 "hostname"
   ```
   It should return: `mpi-slave1` immediately **without asking for a password**. Then test localhost:
   ```bash
   ssh mpi-master "hostname"
   ```

---

# Phase 7: Deploying the Code to Both VMs

To keep files synchronized, you can either set up an NFS share or sync via `rsync`. For this mini-project, **`rsync`** is reliable, fast, and does not require complex NFS daemon configuration.

On **`mpi-master`**:
```bash
cd /home/mpiuser/hpc_project
```

### 1. Create `matmul_mpi_worker.py`
```bash
nano matmul_mpi_worker.py
```
Paste the MPI worker code:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import json
import argparse
import numpy as np
from mpi4py import MPI

def run_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=4)
    args = parser.parse_args()

    n = args.n
    reps = args.reps

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if n % size != 0:
        if rank == 0:
            sys.stderr.write(f"[-] n={n} not divisible by size={size}\n")
        sys.exit(1)

    rows_per_proc = n // size

    # Buffers
    rep_total = []
    rep_bcast = []
    rep_scatter = []
    rep_compute = []
    rep_gather = []

    for r in range(reps):
        if rank == 0:
            A = np.random.rand(n, n).astype(np.float64)
            B = np.random.rand(n, n).astype(np.float64)
            C = np.empty((n, n), dtype=np.float64)
        else:
            A = None
            B = np.empty((n, n), dtype=np.float64)
            C = None

        A_loc = np.empty((rows_per_proc, n), dtype=np.float64)
        C_loc = np.empty((rows_per_proc, n), dtype=np.float64)

        comm.Barrier()
        t_total_start = MPI.Wtime()

        # Bcast B
        comm.Barrier()
        t0 = MPI.Wtime()
        comm.Bcast(B, root=0)
        t_bcast = MPI.Wtime() - t0

        # Scatter A
        comm.Barrier()
        t0 = MPI.Wtime()
        comm.Scatter(A, A_loc, root=0)
        t_scatter = MPI.Wtime() - t0

        # Compute
        comm.Barrier()
        t0 = MPI.Wtime()
        C_loc = A_loc @ B
        t_compute = MPI.Wtime() - t0

        # Gather C
        comm.Barrier()
        t0 = MPI.Wtime()
        comm.Gather(C_loc, C, root=0)
        t_gather = MPI.Wtime() - t0

        comm.Barrier()
        t_total = MPI.Wtime() - t_total_start

        max_bcast = comm.reduce(t_bcast, op=MPI.MAX, root=0)
        max_scatter = comm.reduce(t_scatter, op=MPI.MAX, root=0)
        max_compute = comm.reduce(t_compute, op=MPI.MAX, root=0)
        max_gather = comm.reduce(t_gather, op=MPI.MAX, root=0)
        max_total = comm.reduce(t_total, op=MPI.MAX, root=0)

        if rank == 0:
            if r == 0:
                C_ref = A @ B
                assert np.allclose(C, C_ref), "Matrix verification failed!"
            rep_total.append(max_total)
            rep_bcast.append(max_bcast)
            rep_scatter.append(max_scatter)
            rep_compute.append(max_compute)
            rep_gather.append(max_gather)

    if rank == 0:
        res = {
            "n": n,
            "p": size,
            "rep_1": rep_total[0],
            "rep_2": rep_total[1],
            "rep_3": rep_total[2],
            "rep_4": rep_total[3],
            "median_total": float(np.median(rep_total)),
            "bcast_median": float(np.median(rep_bcast)),
            "scatter_median": float(np.median(rep_scatter)),
            "compute_median": float(np.median(rep_compute)),
            "gather_median": float(np.median(rep_gather)),
        }
        print(f"__RESULT_JSON__{json.dumps(res)}")

if __name__ == "__main__":
    run_worker()
```

---

### 2. Create the MPI `hostfile`
```bash
nano /home/mpiuser/hpc_project/hostfile
```
Add your cluster topology (each VM has 2 CPU slots):
```text
mpi-master slots=2
mpi-slave1 slots=2
```

---

### 3. Sync the Project Directory to `mpi-slave1`
Run this sync command on `mpi-master`:
```bash
rsync -avz --exclude 'venv' /home/mpiuser/hpc_project/ mpiuser@mpi-slave1:/home/mpiuser/hpc_project/
```

---

# Phase 8: Cluster Verification & Running the Benchmark

### 1. Sanity Test: Distributed Hostname Check
Run this on `mpi-master`:
```bash
mpirun --hostfile hostfile -n 4 hostname
```
**Expected output:**
```text
mpi-master
mpi-master
mpi-slave1
mpi-slave1
```
If you see output from both machines, **your distributed VM cluster is fully functional.**

---

### 2. Create the Orchestrator Script (`run_cluster_benchmark.py`)
On `mpi-master`:
```bash
nano run_cluster_benchmark.py
```
Paste this orchestrator:

```python
import subprocess
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_cluster_suite():
    matrix_sizes = [1024, 2048]
    process_counts = [1, 2, 4]  # 4 ranks across two 2-core VMs
    reps = 4
    results = []

    print("==================================================================")
    print("      RUNNING VERSION 2: DISTRIBUTED VM CLUSTER BENCHMARK         ")
    print("==================================================================")

    for n in matrix_sizes:
        print(f"\n>>> Matrix Dimension n = {n} <<<")
        t1_reference = None

        for p in process_counts:
            # We explicitly specify python inside our venv to ensure uniform paths
            cmd = [
                "mpirun",
                "--hostfile", "hostfile",
                "-n", str(p),
                "/home/mpiuser/hpc_project/venv/bin/python",
                "matmul_mpi_worker.py",
                "--n", str(n),
                "--reps", str(reps)
            ]

            print(f"[*] Running: mpirun -n {p} across VMs...")
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if proc.returncode != 0:
                print(f"[-] Execution failed on ranks p={p}:\n{proc.stderr}")
                continue

            data = None
            for line in proc.stdout.splitlines():
                if line.startswith("__RESULT_JSON__"):
                    data = json.loads(line.replace("__RESULT_JSON__", ""))
                    break

            if not data:
                print(f"[-] No valid JSON output captured for p={p}")
                continue

            if p == 1:
                t1_reference = data["median_total"]
                speedup = 1.0
                efficiency = 1.0
            else:
                speedup = t1_reference / data["median_total"]
                efficiency = speedup / p

            comm_total = data["bcast_median"] + data["scatter_median"] + data["gather_median"]
            data["speedup"] = speedup
            data["efficiency"] = efficiency
            data["comm_total_median"] = comm_total

            print(f"    -> [p={p}] Total: {data['median_total']:.4f}s | "
                  f"Compute: {data['compute_median']:.4f}s | "
                  f"Network Comm: {comm_total:.4f}s | "
                  f"Speedup: {speedup:.2f}x | Efficiency: {efficiency:.2%}")

            results.append(data)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("vm_cluster_results.csv", index=False)
    print("\n[+] Benchmark finished! Saved to 'vm_cluster_results.csv'")

if __name__ == "__main__":
    run_cluster_suite()
```

### 3. Run the Distributed Benchmark
On `mpi-master`:
```bash
cd /home/mpiuser/hpc_project
source venv/bin/activate
python run_cluster_benchmark.py
```

---

# Phase 9: Network Ping-Pong Test (Deliverable Requirement)

Section 6 of the proposal requires:
> *"Run a ping-pong test between two nodes on each target to measure latency and bandwidth."*

Create **`ping_pong.py`** on `mpi-master`:
```python
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# 1. LATENCY TEST (8-byte small message)
iterations = 1000
data_small = np.zeros(1, dtype=np.float64)

comm.Barrier()
t0 = MPI.Wtime()
for _ in range(iterations):
    if rank == 0:
        comm.Send(data_small, dest=1, tag=11)
        comm.Recv(data_small, source=1, tag=12)
    elif rank == 1:
        comm.Recv(data_small, source=0, tag=11)
        comm.Send(data_small, dest=0, tag=12)
t_latency = (MPI.Wtime() - t0) / (2 * iterations)

# 2. BANDWIDTH TEST (10 MB payload)
size_mb = 10
n_elements = (size_mb * 1024 * 1024) // 8
data_large = np.ones(n_elements, dtype=np.float64)
bw_iterations = 50

comm.Barrier()
t0 = MPI.Wtime()
for _ in range(bw_iterations):
    if rank == 0:
        comm.Send(data_large, dest=1, tag=21)
        comm.Recv(data_large, source=1, tag=22)
    elif rank == 1:
        comm.Recv(data_large, source=0, tag=21)
        comm.Send(data_large, dest=0, tag=22)
total_time = MPI.Wtime() - t0
# Total bytes sent back and forth
total_bytes = 2 * bw_iterations * size_mb
bandwidth_MBps = total_bytes / total_time

if rank == 0:
    print(f"\n=== VM-to-VM Ping-Pong Results ===")
    print(f"Network Latency  : {t_latency * 1e6:.2f} microseconds")
    print(f"Network Bandwidth: {bandwidth_MBps:.2f} MB/s\n")
```

Sync it to the slave:
```bash
rsync -avz ping_pong.py mpiuser@mpi-slave1:/home/mpiuser/hpc_project/
```

Run it across nodes:
```bash
mpirun --hostfile hostfile -n 2 /home/mpiuser/hpc_project/venv/bin/python ping_pong.py
```

You now have:
1. Two connected VMs communicating over bridged IPs.
2. Verified passwordless SSH and Open MPI communication.
3. Latency & bandwidth numbers for your report.
4. Correctness-checked distributed matrix multiplication timing data saved to `vm_cluster_results.csv`.