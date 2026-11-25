#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoDL / SeetaCloud SSH & SFTP Helper Tool
"""
import sys
import os
import argparse
import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = 36551
USER = "root"
PASS = "bJSG3IB1MRXwK9UWT8S2/JiU6cX19Hgp"

def get_ssh_client():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=HOST, port=PORT, username=USER, password=PASS, timeout=15)
    return client

def run_remote_command(command, print_output=True):
    client = get_ssh_client()
    try:
        stdin, stdout, stderr = client.exec_command(command, get_pty=True)
        out = []
        for line in iter(stdout.readline, ""):
            if print_output:
                print(line, end="")
            out.append(line)
        exit_status = stdout.channel.recv_exit_status()
        return exit_status, "".join(out)
    finally:
        client.close()

def upload_file(local_path, remote_path):
    client = get_ssh_client()
    try:
        sftp = client.open_sftp()
        print(f"Uploading {local_path} -> {remote_path}...")
        sftp.put(local_path, remote_path)
        print("Upload completed successfully.")
        sftp.close()
    finally:
        client.close()

def download_file(remote_path, local_path):
    client = get_ssh_client()
    try:
        sftp = client.open_sftp()
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        print(f"Downloading {remote_path} -> {local_path}...")
        sftp.get(remote_path, local_path)
        print("Download completed successfully.")
        sftp.close()
    finally:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSH Client Helper for SeetaCloud")
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")

    exec_p = subparsers.add_parser("exec", help="Execute remote command")
    exec_p.add_argument("cmd", type=str, help="Command to run")

    up_p = subparsers.add_parser("upload", help="Upload local file to remote")
    up_p.add_argument("local", type=str, help="Local file path")
    up_p.add_argument("remote", type=str, help="Remote file path")

    down_p = subparsers.add_parser("download", help="Download remote file to local")
    down_p.add_argument("remote", type=str, help="Remote file path")
    down_p.add_argument("local", type=str, help="Local file path")

    args = parser.parse_args()
    if args.action == "exec":
        status, _ = run_remote_command(args.cmd)
        sys.exit(status)
    elif args.action == "upload":
        upload_file(args.local, args.remote)
    elif args.action == "download":
        download_file(args.remote, args.local)
    else:
        parser.print_help()
