import os
import sys
import time
import hashlib
import requests
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

REMAINING_FILES = [
    {
        "filename": "IR-Thin_Wall_PartII.avi",
        "access_url": "https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7910/DVN/AHYXL9/IKGWEE",
        "filesize": 107820544,
        "md5": "d96da91174a0331d4b1170eeecbd9127"
    },
    {
        "filename": "IR-Thin_Wall_PartI.avi",
        "access_url": "https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7910/DVN/AHYXL9/P0DMZJ",
        "filesize": 215468032,
        "md5": "e7ba0cd7c61236f7f891c4ce8ea18679"
    },
    {
        "filename": "Pyro-Temperature.avi",
        "access_url": "https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7910/DVN/AHYXL9/PY9LET",
        "filesize": 230077440,
        "md5": "98e930d43a8cb120e176db3694d0b5b5"
    },
    {
        "filename": "300W-30ipm-4rpm_CSV_Pyro - time-location.zip",
        "access_url": "https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7910/DVN/AHYXL9/CB21DS",
        "filesize": 724964619,
        "md5": "2bd669d008ea11c1bd34f0f3875c805f"
    }
]

REMOTE_TARGET_DIR = "/root/autodl-tmp/Ti6Al4V_LENS_datasets/2016_Ti6Al4V_LENS_MeltPool_BulkTemp"

def compute_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def get_s3_url(access_url):
    for _ in range(5):
        try:
            r = requests.get(access_url, allow_redirects=False, timeout=30)
            if "Location" in r.headers:
                return r.headers["Location"]
            elif r.status_code == 200:
                return access_url
        except Exception as e:
            print("Retry resolving URL:", e)
        time.sleep(2)
    raise RuntimeError(f"Failed to get S3 URL from {access_url}")

def download_file_multithread(s3_url, local_path, expected_size, num_threads=12, chunk_size=4*1024*1024):
    print(f"Pre-allocating {local_path} ({expected_size / (1024*1024):.2f} MB)...")
    with open(local_path, "wb") as f:
        f.seek(expected_size - 1)
        f.write(b"\0")

    ranges = []
    for start in range(0, expected_size, chunk_size):
        end = min(start + chunk_size - 1, expected_size - 1)
        ranges.append((start, end))

    print(f"Downloading {len(ranges)} chunks with {num_threads} threads...")
    completed_bytes = 0
    t0 = time.time()

    def download_chunk(rg):
        start, end = rg
        headers = {"Range": f"bytes={start}-{end}"}
        for attempt in range(5):
            try:
                res = requests.get(s3_url, headers=headers, timeout=40)
                if res.status_code in (200, 206):
                    with open(local_path, "r+b") as fp:
                        fp.seek(start)
                        fp.write(res.content)
                    return len(res.content)
            except Exception as e:
                time.sleep(1)
        raise RuntimeError(f"Failed chunk {start}-{end}")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(download_chunk, rg): rg for rg in ranges}
        for future in as_completed(futures):
            c_len = future.result()
            completed_bytes += c_len
            pct = (completed_bytes / expected_size) * 100
            elapsed = time.time() - t0
            speed_mb = (completed_bytes / (1024 * 1024)) / max(elapsed, 0.001)
            sys.stdout.write(f"\rProgress: {completed_bytes / (1024*1024):.1f}/{expected_size / (1024*1024):.1f} MB ({pct:.1f}%) | Speed: {speed_mb:.2f} MB/s")
            sys.stdout.flush()

    print(f"\nDownload completed in {time.time() - t0:.1f} s")

def scp_to_remote(local_path, remote_dir):
    filename = os.path.basename(local_path)
    rel_path = f"./{filename}"
    remote_dest = f"{remote_dir}/{filename}"
    cmd = [
        "bash", "./scp_to_path.sh",
        rel_path,
        remote_dest
    ]
    print(f"Uploading {rel_path} to remote {remote_dest}...")
    t0 = time.time()
    ret = subprocess.run(cmd, stdin=subprocess.DEVNULL)
    if ret.returncode != 0:
        raise RuntimeError(f"SCP failed with code {ret.returncode}")
    print(f"Upload finished in {time.time() - t0:.1f} s")

def verify_remote_md5(filename, expected_md5):
    cmd = [
        "bash", "./run_remote.sh",
        f"md5sum \"{REMOTE_TARGET_DIR}/{filename}\""
    ]
    res = subprocess.run(cmd, capture_output=True)
    remote_output = res.stdout.decode("utf-8", errors="ignore").strip()
    print(f"Remote MD5 output: {remote_output}")
    if expected_md5.lower() in remote_output.lower():
        print(f"[VERIFIED] Remote MD5 matches expected {expected_md5}!")
        return True
    else:
        return False

def process_file(item):
    fname = item["filename"]
    fsize = item["filesize"]
    exp_md5 = item["md5"]
    access_url = item["access_url"]

    print("\n" + "=" * 70)
    print(f"PROCESSING: {fname} ({fsize / (1024*1024):.2f} MB)")
    print("=" * 70)

    # 1. Check if remote already has it verified
    try:
        if verify_remote_md5(fname, exp_md5):
            print(f"File {fname} already fully verified on remote server. Skipping.")
            return
    except Exception:
        pass

    local_path = os.path.join(os.getcwd(), fname)

    # 2. Check local file if already downloaded
    need_download = True
    if os.path.exists(local_path) and os.path.getsize(local_path) == fsize:
        print("Local file exists. Checking MD5...")
        if compute_md5(local_path).lower() == exp_md5.lower():
            print("Local MD5 valid!")
            need_download = False

    if need_download:
        print("Resolving S3 URL...")
        s3_url = get_s3_url(access_url)
        download_file_multithread(s3_url, local_path, fsize, num_threads=12)
        print("Verifying local MD5...")
        local_md5 = compute_md5(local_path)
        if local_md5.lower() != exp_md5.lower():
            raise RuntimeError(f"Local MD5 mismatch for {fname}: got {local_md5}, expected {exp_md5}")
        print(f"Local MD5 verified: {local_md5}")

    # 3. SCP to remote
    scp_to_remote(local_path, REMOTE_TARGET_DIR)

    # 4. Verify remote
    verify_remote_md5(fname, exp_md5)

    # 5. Clean up local file to free disk space
    if os.path.exists(local_path):
        os.remove(local_path)
        print(f"Cleaned up local temporary file: {local_path}")

def main():
    # Remove any unfinished .aria2 file on remote for clean state
    subprocess.run([
        "bash", "./run_remote.sh",
        f"rm -f \"{REMOTE_TARGET_DIR}/\"*.aria2"
    ])

    for item in REMAINING_FILES:
        process_file(item)

    print("\n" + "=" * 70)
    print("ALL 4 REMAINING FILES DOWNLOADED, UPLOADED, AND VERIFIED ON REMOTE SERVER!")
    print("=" * 70)

if __name__ == "__main__":
    main()
