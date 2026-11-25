import os
import sys
import time
import subprocess

TARGET_PID = 8283
PROJECT_DIR = "/root/autodl-tmp/Ti6Al4V_LENS_project"
LOG_FILE = os.path.join(PROJECT_DIR, "watcher.log")

def log(msg):
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now_str}] {msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)

def is_pid_running(pid):
    try:
        # 0 signal checks if process exists
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def get_gpu_memory_used():
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return int(res.stdout.strip())
    except Exception:
        return 99999

def main():
    log(f"=== GPU WATCHER DAEMON STARTED ===")
    log(f"Monitoring Target PID: {TARGET_PID}")
    log(f"Project Directory: {PROJECT_DIR}")
    
    idle_count = 0
    while True:
        pid_alive = is_pid_running(TARGET_PID)
        vram_mb = get_gpu_memory_used()

        if not pid_alive:
            if vram_mb < 500:
                idle_count += 1
                log(f"[IDLE DETECTED] Target PID {TARGET_PID} terminated. GPU Memory: {vram_mb} MiB. Idle Count: {idle_count}/3")
                if idle_count >= 3:
                    log(">>> GPU IS STABLY IDLE! AUTO-TRIGGERING STAGE B EXPERIMENT PIPELINE! <<<")
                    break
            else:
                log(f"[WAITING] Target PID {TARGET_PID} exited, but GPU Memory is still {vram_mb} MiB. Checking again...")
                idle_count = 0
        else:
            idle_count = 0
            log(f"[RUNNING] Target PID {TARGET_PID} active | GPU Memory: {vram_mb} MiB | GPU Busy")

        time.sleep(20)

    # Launch Stage B
    script_path = os.path.join(PROJECT_DIR, "run_stageB_mamba_thermal_dl.py")
    log_path = os.path.join(PROJECT_DIR, "stageB_experiment.log")
    log(f"Launching command: python {script_path} > {log_path} 2>&1")
    
    env = os.environ.copy()
    env["PATH"] = f"/root/miniconda3/bin:{env.get('PATH', '')}"

    with open(log_path, "w", encoding="utf-8") as out_f:
        p = subprocess.Popen(["python", script_path], cwd=PROJECT_DIR, stdout=out_f, stderr=subprocess.STDOUT, env=env)
        log(f"Stage B Experiment Process Launched with PID: {p.pid}!")
        p.wait()
        log(f"Stage B Experiment Process PID {p.pid} Finished with Exit Code: {p.returncode}!")

    log("=== GPU WATCHER DAEMON TERMINATED NORMALLY ===")

if __name__ == "__main__":
    main()
