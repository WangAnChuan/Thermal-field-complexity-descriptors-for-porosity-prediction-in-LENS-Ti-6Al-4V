import os
import sys
import time
import hashlib
import subprocess
import requests

DATASETS_CONFIG = [
    {
        "name": "2016_Ti6Al4V_LENS_MeltPool_BulkTemp",
        "persistent_id": "doi:10.7910/DVN/AHYXL9",
        "title": "Data indicating melt pool and bulk part temperature during the additive manufacture of a Ti-6Al-4V thin-walled structure via Laser Engineered Net Shaping",
        "year": "2016",
        "files": [
            {
                "access_url": "https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7910/DVN/AHYXL9/4R71ZH",
                "filename": "300W-30ipm-4rpm_CSV_IR - time-location.zip",
                "filesize": 132715657,
                "md5": "0d5783969de0c0f3cd8715e7b4d8a6a2",
                "description": "IR recorded temperature data of bulk part during LENS fabrication of Ti-6Al-4V thin wall."
            },
            {
                "access_url": "https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7910/DVN/AHYXL9/CB21DS",
                "filename": "300W-30ipm-4rpm_CSV_Pyro - time-location.zip",
                "filesize": 724964619,
                "md5": "2bd669d008ea11c1bd34f0f3875c805f",
                "description": "Pyrometer recorded temperature data of melt pool during LENS fabrication of Ti-6Al-4V thin wall."
            },
            {
                "access_url": "https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7910/DVN/AHYXL9/P0DMZJ",
                "filename": "IR-Thin_Wall_PartI.avi",
                "filesize": 215468032,
                "md5": "e7ba0cd7c61236f7f891c4ce8ea18679",
                "description": "Video reconstructed from in-situ monitoring of Ti-6Al-4V thin-wall build with the IR camera (Part I)."
            },
            {
                "access_url": "https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7910/DVN/AHYXL9/IKGWEE",
                "filename": "IR-Thin_Wall_PartII.avi",
                "filesize": 107820544,
                "md5": "d96da91174a0331d4b1170eeecbd9127",
                "description": "Video reconstructed from in-situ monitoring of Ti-6Al-4V thin-wall build with the IR camera (Part II)."
            },
            {
                "access_url": "https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7910/DVN/AHYXL9/PY9LET",
                "filename": "Pyro-Temperature.avi",
                "filesize": 230077440,
                "md5": "98e930d43a8cb120e176db3694d0b5b5",
                "description": "Video of the melt pool reconstructed from in-situ monitoring of Ti-6Al-4V thin-wall build with the pyrometer."
            }
        ]
    },
    {
        "name": "2023_Ti6Al4V_Thermal_Porosity_Cropped",
        "persistent_id": "doi:10.7910/DVN/BWHYEH",
        "title": "Thermal-Porosity Characterization Data of Additively Manufactured Ti–6Al–4V Thin-walled Structure via Laser Engineered Net Shaping",
        "year": "2023",
        "files": [
            {
                "access_url": "https://dataverse.harvard.edu/api/access/datafile/7172366",
                "filename": "cropped_thinwall_CSV_pyrometer.zip",
                "filesize": 54580473,
                "md5": "b50903ebb39c29b0b1e4d0d73af57194",
                "description": "In-process thermal melt pool pyrometer CSV images cropped around >1000C melt pool."
            },
            {
                "access_url": "https://dataverse.harvard.edu/api/access/datafile/7172365",
                "filename": "thermal-porosity-table.xlsx",
                "filesize": 122347,
                "md5": "9f385aba819cb2f3d0a985d8b425df24",
                "description": "XCT internal porosity labeled table registered within 0.5 mm of melt pool image coordinate."
            }
        ]
    }
]

def compute_md5(filepath, chunk_size=1024*1024):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def get_s3_url(access_url, max_retries=5):
    for attempt in range(max_retries):
        try:
            r = requests.get(access_url, allow_redirects=False, timeout=30)
            if "Location" in r.headers:
                return r.headers["Location"]
            elif r.status_code == 200:
                return access_url
            else:
                print(f"  [Attempt {attempt+1}] Status {r.status_code}, no Location header.")
        except Exception as e:
            print(f"  [Attempt {attempt+1}] Error querying Dataverse: {e}")
        time.sleep(2)
    raise RuntimeError(f"Failed to resolve download URL from {access_url}")

def download_file_aria2(s3_url, dest_dir, filename, expected_size, max_retries=3):
    dest_path = os.path.join(dest_dir, filename)
    os.makedirs(dest_dir, exist_ok=True)

    for attempt in range(max_retries):
        print(f"  -> aria2c downloading {filename} (Attempt {attempt+1}/{max_retries})...")
        cmd = [
            "aria2c",
            "-c",
            "-x", "16",
            "-s", "16",
            "-j", "16",
            "-k", "1M",
            "--file-allocation=none",
            "--summary-interval=10",
            "-d", dest_dir,
            "-o", filename,
            s3_url
        ]
        ret = subprocess.run(cmd)
        if ret.returncode == 0 and os.path.exists(dest_path):
            return True
        else:
            print(f"  [WARN] aria2c returned {ret.returncode}, retrying...")
            # If S3 URL expired or interrupted, re-query S3 URL
            time.sleep(3)
    return False

def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/Ti6Al4V_LENS_datasets"
    os.makedirs(base_dir, exist_ok=True)
    print("=" * 80)
    print(f"HARVARD DATAVERSE DATASET DOWNLOADER")
    print(f"Target Root Directory: {base_dir}")
    print("=" * 80)

    summary_records = []

    for ds in DATASETS_CONFIG:
        ds_name = ds["name"]
        ds_dir = os.path.join(base_dir, ds_name)
        os.makedirs(ds_dir, exist_ok=True)
        print(f"\nProcessing Dataset: {ds['title']} ({ds['year']})")
        print(f"Persistent ID: {ds['persistent_id']}")
        print(f"Target Directory: {ds_dir}")
        print("-" * 80)

        for f_info in ds["files"]:
            access_url = f_info["access_url"]
            fname = f_info["filename"]
            fsize = f_info["filesize"]
            exp_md5 = f_info["md5"]
            dest_file = os.path.join(ds_dir, fname)

            print(f"\nChecking: {fname} ({fsize / (1024*1024):.2f} MB)")
            if os.path.exists(dest_file):
                curr_size = os.path.getsize(dest_file)
                if curr_size == fsize:
                    print("  File size matches. Verifying MD5 checksum...")
                    curr_md5 = compute_md5(dest_file)
                    if curr_md5.lower() == exp_md5.lower():
                        print(f"  [SKIP] File already exists and MD5 verified: {curr_md5}")
                        summary_records.append({
                            "dataset": ds_name,
                            "filename": fname,
                            "size_mb": round(fsize / (1024*1024), 2),
                            "md5": curr_md5,
                            "status": "Verified (Cached)"
                        })
                        continue
                    else:
                        print(f"  [WARN] MD5 mismatch ({curr_md5} != {exp_md5}). Re-downloading...")
                else:
                    print(f"  [WARN] Partial file detected ({curr_size}/{fsize} bytes). Resuming/Re-downloading...")

            s3_url = get_s3_url(access_url)
            ok = download_file_aria2(s3_url, ds_dir, fname, fsize)
            if not ok:
                print(f"  [ERROR] Failed to download {fname} via aria2c!")
                sys.exit(1)

            # MD5 Verification
            print("  Verifying downloaded file MD5...")
            downloaded_md5 = compute_md5(dest_file)
            if downloaded_md5.lower() == exp_md5.lower():
                print(f"  [SUCCESS] MD5 match confirmed: {downloaded_md5}")
                summary_records.append({
                    "dataset": ds_name,
                    "filename": fname,
                    "size_mb": round(fsize / (1024*1024), 2),
                    "md5": downloaded_md5,
                    "status": "Verified (Downloaded)"
                })
            else:
                print(f"  [FATAL ERROR] MD5 Mismatch: got {downloaded_md5}, expected {exp_md5}")
                sys.exit(2)

    # Generate README.md
    readme_path = os.path.join(base_dir, "README.md")
    readme_content = f"""# Ti-6Al-4V LENS 增材制造热成像与孔隙缺陷多模态数据集汇总说明

本项目数据集目录包含两套来自 Harvard Dataverse 的权威公开增材制造（Additive Manufacturing, AM）原位监测与缺陷检测金标准数据集：

---

## 1. 2016年 原生原位熔池与宏观温度监测数据集 (Raw Thermal Monitoring)
- **DOI**: [10.7910/DVN/AHYXL9](https://doi.org/10.7910/DVN/AHYXL9)
- **数据集全称**: *Data indicating melt pool and bulk part temperature during the additive manufacture of a Ti-6Al-4V thin-walled structure via Laser Engineered Net Shaping*
- **研究机构**: Mississippi State University & Center for Advanced Vehicular Systems (CAVS)
- **第一作者**: Garrett Marshall, Scott M. Thompson, Nima Shamsaei
- **制造工艺与材料**: OPTOMEC LENS 750 系统，同轴送粉激光直接能量沉积（DLD），Ti-6Al-4V 薄壁结构
- **传感器配置**:
  1. **双波长比色高温计 (Dual-wavelength Pyrometer)**: 752×480 分辨率，精确测量熔池及其热影响区（HAZ）原位瞬态温度；
  2. **红外热像仪 (Infrared IR Camera)**: 320×240 分辨率，测量薄壁宏观整体瞬态温度场分布；
  3. **重建视频**: 熔池与整体原位沉积全过程 AVI 视频。
- **目录位置**: `2016_Ti6Al4V_LENS_MeltPool_BulkTemp/`

---

## 2. 2023年 熔池裁剪与 XCT 内部孔隙配准处理数据集 (Processed & Porosity Labeled)
- **DOI**: [10.7910/DVN/BWHYEH](https://doi.org/10.7910/DVN/BWHYEH)
- **数据集全称**: *Thermal-Porosity Characterization Data of Additively Manufactured Ti–6Al–4V Thin-walled Structure via Laser Engineered Net Shaping*
- **研究机构**: Mississippi State University & Arizona State University
- **第一作者**: Christian Zamiela, Wenmeng Tian, Shenghan Guo, Linkan Bian
- **与2016年数据集的核心渊源与技术处理**:
  - **裁剪对齐 (Cropping)**: 对 2016 年原生比色高温计 CSV 图像进行裁剪，以熔池中心为核心，聚焦提取温度 > 1000℃ 的有效熔池形态与温度场分布；
  - **金标准孔隙配准 (XCT Porosity Ground Truth)**: 采用 Nikon X-Ray Computed Tomography (XCT XT H225) 对沉积成形后的薄壁内部未熔合（Lack of Fusion, LoF）微观孔隙缺陷（0.05 mm ~ 1.00 mm）进行离线三维无损重构；
  - **多模态时空配准**: 将 XCT 内部孔隙缺陷精确配准到原位比色高温计图像坐标（0.5 mm 容差内），并在 Excel 表中提供精准时间戳、熔池坐标、偏心率、峰值温度及孔隙尺寸/标签。
- **目录位置**: `2023_Ti6Al4V_Thermal_Porosity_Cropped/`

---

## 3. 已归档文件校验清单 (MD5 Integrity Ledger)

| 目录 | 文件名 | 大小 (MB) | MD5 校验码 | 完整性状态 |
| :--- | :--- | :---: | :--- | :---: |
"""
    for r in summary_records:
        readme_content += f"| `{r['dataset']}` | `{r['filename']}` | {r['size_mb']} MB | `{r['md5']}` | {r['status']} |\n"

    readme_content += """
---
*本说明文件自动生成于数据集下载与 MD5 闭环校验流水线。*
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("\n" + "=" * 80)
    print(f"DOWNLOAD AND VERIFICATION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Summary README generated at: {readme_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
