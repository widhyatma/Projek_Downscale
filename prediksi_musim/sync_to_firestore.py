#!/usr/bin/env python3
"""
sync_to_firestore.py
====================
Sinkronisasi data prediksi musim dan katalog klimatologi Kabupaten Kebumen
ke Google Cloud Firestore (Namespace: 'musim_kebumen').

Fitur Utama:
1. Menggunakan service account resmi (staklimjerukagung-b852a12a367e.json).
2. Arsitektur Chunked: 1 Dokumen Kecamatan = Data Kecamatan + Array Desa (~15-25 desa).
3. Sangat hemat kuota: Seluruh 26 kecamatan diunggah dalam 1 kali Batch Commit (26 write).
4. Dual-Season Support: Menyimpan parameter Musim Hujan (MH) dan Musim Kemarau (MK).
5. Opsi --dry-run untuk memvalidasi payload sebelum menulis ke cloud.

Penggunaan:
    conda run -n tensorflow python prediksi_musim/sync_to_firestore.py
    conda run -n tensorflow python prediksi_musim/sync_to_firestore.py --dry-run
    conda run -n tensorflow python prediksi_musim/sync_to_firestore.py --sync-all-archives
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sync_firestore")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_KEY_PATH = BASE_DIR / "staklimjerukagung-b852a12a367e.json"
DEFAULT_DASHBOARD_DATA = Path(r"D:\Github\Dashboard_Jerukagung-Meteorologi\public\data")


def parse_args():
    parser = argparse.ArgumentParser(description="Sinkronisasi Prediksi Musim Kebumen ke Cloud Firestore")
    parser.add_argument("--key", type=str, default=str(DEFAULT_KEY_PATH), help="Path ke berkas serviceAccountKey.json")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DASHBOARD_DATA), help="Path ke public/data dashboard")
    parser.add_argument("--dry-run", action="store_true", help="Validasi payload tanpa menulis ke Firestore")
    parser.add_argument("--sync-all-archives", action="store_true", help="Unggah seluruh 11 tahun arsip (default: hanya musim aktif)")
    parser.add_argument("--season-id", type=str, default=None, help="Spesifik season_id untuk disinkronkan (misal: 2026_2027)")
    return parser.parse_args()


def init_firestore(key_path: Path):
    """Menginisialisasi klien Firestore menggunakan service account."""
    if not key_path.exists():
        raise FileNotFoundError(f"Service account key tidak ditemukan di: {key_path}")
    
    import firebase_admin
    from firebase_admin import credentials, firestore

    # Cegah double-initialization
    if not firebase_admin._apps:
        cred = credentials.Certificate(str(key_path))
        firebase_admin.initialize_app(cred)
    
    return firestore.client()


def load_json(filepath: Path):
    """Membaca berkas JSON dengan encoding UTF-8."""
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_kecamatan_chunk(kec_hujan_rec: dict, kec_kemarau_rec: dict, desa_hujan_list: list, desa_kemarau_list: list) -> dict:
    """Menggabungkan data kecamatan dan seluruh desanya ke dalam satu payload dokumen chunked."""
    kec_name = kec_hujan_rec.get("kecamatan") or (kec_kemarau_rec.get("kecamatan") if kec_kemarau_rec else "Unknown")
    code = kec_hujan_rec.get("code") or (kec_kemarau_rec.get("code") if kec_kemarau_rec else "")

    return {
        "kecamatan": kec_name,
        "code": code,
        "hujan": kec_hujan_rec,
        "kemarau": kec_kemarau_rec,
        "total_desa": len(desa_hujan_list),
        "desa_hujan": desa_hujan_list,
        "desa_kemarau": desa_kemarau_list,
    }


def sync_season(db, season_id: str, season_folder: Path, is_active: bool, dry_run: bool = False):
    """Menyinkronkan satu musim (overview + 26 kecamatan) ke Firestore."""
    logger.info(f"==> Memproses Season ID: {season_id} (Folder: {season_folder.name or 'root'})")

    kec_file = season_folder / "prediksi_musim.json"
    desa_file = season_folder / "prediksi_musim_desa.json"

    data_kec = load_json(kec_file)
    data_desa = load_json(desa_file)

    if not data_kec:
        logger.warning(f"Berkas {kec_file} tidak ditemukan, lewati season {season_id}.")
        return False

    # 1. Dokumen Overview Season (Metadata & Overlays)
    overview_payload = {
        "season_id": season_id,
        "is_active": is_active,
        "metadata": data_kec.get("metadata", {}),
        "overlays": data_kec.get("overlays", []),
    }

    # 2. Kelompokkan Desa berdasarkan Kecamatan
    desa_hujan_by_kec = {}
    desa_kemarau_by_kec = {}

    if data_desa:
        for dh in data_desa.get("hujan", []):
            k = dh.get("kecamatan")
            if k:
                desa_hujan_by_kec.setdefault(k, []).append(dh)
        for dk in data_desa.get("kemarau", []):
            k = dk.get("kecamatan")
            if k:
                desa_kemarau_by_kec.setdefault(k, []).append(dk)

    # 3. Bentuk Payload 26 Kecamatan
    kec_hujan_map = {r["kecamatan"]: r for r in data_kec.get("hujan", []) if "kecamatan" in r}
    kec_kemarau_map = {r["kecamatan"]: r for r in data_kec.get("kemarau", []) if "kecamatan" in r}

    all_kec_names = sorted(list(set(list(kec_hujan_map.keys()) + list(kec_kemarau_map.keys()))))
    logger.info(f"Ditemukan {len(all_kec_names)} kecamatan untuk season {season_id}.")

    if dry_run:
        logger.info(f"[DRY-RUN] Overview payload valid: {overview_payload['metadata'].get('kabupaten')}")
        logger.info(f"[DRY-RUN] Akan menulis 1 overview doc + {len(all_kec_names)} kecamatan docs.")
        for k in all_kec_names[:3]:
            chunk = build_kecamatan_chunk(
                kec_hujan_map.get(k, {}),
                kec_kemarau_map.get(k, {}),
                desa_hujan_by_kec.get(k, []),
                desa_kemarau_by_kec.get(k, []),
            )
            logger.info(f"[DRY-RUN] Sample chunk {k}: code={chunk['code']}, desa_count={chunk['total_desa']}")
        return True

    # 4. Tulis ke Cloud Firestore
    season_col = db.collection("musim_kebumen").document(season_id)

    # Tulis Overview
    season_col.set(overview_payload)

    # Batch Commit untuk 26 Kecamatan
    batch = db.batch()
    kec_col = season_col.collection("kecamatan")

    for kec_name in all_kec_names:
        chunk = build_kecamatan_chunk(
            kec_hujan_map.get(kec_name, {}),
            kec_kemarau_map.get(kec_name, {}),
            desa_hujan_by_kec.get(kec_name, []),
            desa_kemarau_by_kec.get(kec_name, []),
        )
        # ID dokumen disanitasi: lowercase, replace spasi dengan underscore
        doc_id = kec_name.lower().replace(" ", "_").replace(".", "")
        doc_ref = kec_col.document(doc_id)
        batch.set(doc_ref, chunk)

    batch.commit()
    logger.info(f"[SUKSES] Berhasil menyimpan overview dan {len(all_kec_names)} dokumen kecamatan untuk {season_id}!")
    return True


def sync_catalog(db, data_dir: Path, dry_run: bool = False):
    """Menyinkronkan master katalog musim (musim_catalog.json)."""
    catalog_file = data_dir / "musim_catalog.json"
    catalog_data = load_json(catalog_file)

    if not catalog_data:
        logger.error(f"Berkas katalog {catalog_file} tidak ditemukan!")
        return False

    if dry_run:
        logger.info(f"[DRY-RUN] Master katalog valid dengan {catalog_data.get('total_seasons')} musim.")
        return True

    doc_ref = db.collection("musim_kebumen").document("catalog")
    doc_ref.set(catalog_data)
    logger.info(f"[SUKSES] Master katalog ({catalog_data.get('total_seasons')} musim) tersimpan di 'musim_kebumen/catalog'!")
    return True


def main():
    args = parse_args()
    key_path = Path(args.key)
    data_dir = Path(args.data_dir)

    logger.info("=" * 70)
    logger.info("MEMULAI SINKRONISASI PREDIKSI MUSIM KE CLOUD FIRESTORE")
    logger.info(f"Target Data Dir : {data_dir}")
    logger.info(f"Dry Run Mode    : {args.dry_run}")
    logger.info("=" * 70)

    db = None
    if not args.dry_run:
        try:
            db = init_firestore(key_path)
            logger.info("Berhasil mengautentikasi ke Google Cloud Firestore.")
        except Exception as e:
            logger.error(f"Gagal menghubungkan ke Firestore: {e}")
            sys.exit(1)

    # 1. Sinkronisasi Katalog Master
    sync_catalog(db, data_dir, dry_run=args.dry_run)

    # 2. Sinkronisasi Musim Aktif (2026/2027)
    catalog_file = data_dir / "musim_catalog.json"
    catalog_data = load_json(catalog_file) or {}
    seasons = catalog_data.get("seasons", [])

    if args.season_id:
        # Spesifik 1 musim
        target_seasons = [s for s in seasons if s.get("season_id") == args.season_id]
    elif args.sync_all_archives:
        # Seluruh musim
        target_seasons = seasons
    else:
        # Default: hanya musim aktif
        target_seasons = [s for s in seasons if s.get("is_active", False)]

    logger.info(f"Total musim yang akan disinkronkan: {len(target_seasons)}")

    for s in target_seasons:
        s_id = s.get("season_id")
        folder_rel = s.get("folder", "")
        s_folder = data_dir / folder_rel if folder_rel else data_dir
        is_active = s.get("is_active", False)

        sync_season(db, s_id, s_folder, is_active=is_active, dry_run=args.dry_run)

    logger.info("=" * 70)
    logger.info("SINKRONISASI FIRESTORE SELESAI DENGAN SUKSES!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
