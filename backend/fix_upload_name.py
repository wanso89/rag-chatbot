#!/usr/bin/env python
"""
uploads/ 디렉터리 안에서
  <hex12>_<hex12>_<hex12>_<UUID>_원본 → <UUID>_원본
으로 일괄 리네임한 뒤
변경 전·후 매핑을 CSV 로 기록한다.
실행 전 반드시 uploads/ 백업을 만들어 두세요!
"""
import re, csv, os, shutil
from pathlib import Path

UPLOAD_DIR = Path("app/static/uploads")
BACKUP_DIR = Path("app/static/uploads_backup")
MAPPING_CSV = "rename_mapping.csv"

# 12자리 16진수 + '_' 패턴
HASH_RE = re.compile(r"^(?:[0-9a-f]{12}_)+")

def main():
    if not UPLOAD_DIR.exists():
        print("🛑 uploads 디렉터리를 찾을 수 없습니다.")
        return

    print("📦 uploads 전체를 백업…")
    if not BACKUP_DIR.exists():
        shutil.copytree(UPLOAD_DIR, BACKUP_DIR)

    mapping = []
    for f in UPLOAD_DIR.iterdir():
        if not f.is_file():
            continue
        new_name = HASH_RE.sub("", f.name)  # 앞쪽 hash_ 들 제거
        if new_name != f.name:
            target = UPLOAD_DIR / new_name
            # 이미 같은 이름이 있으면 skip
            if target.exists():
                print(f"⚠️  {target.name} 이미 존재, {f.name} 삭제만 수행")
                f.unlink()
                continue
            print(f"🔄 {f.name}  →  {new_name}")
            f.rename(target)
            mapping.append([f.name, new_name])

    # 매핑 CSV 기록
    with open(MAPPING_CSV, "w", newline="", encoding="utf-8") as cf:
        wr = csv.writer(cf)
        wr.writerow(["old_name", "new_name"])
        wr.writerows(mapping)

    print(f"✅ 완료! 리네임 {len(mapping)}건. 매핑: {MAPPING_CSV}")

if __name__ == "__main__":
    main()
