import os
import shutil
import zipfile
import pandas as pd
import argparse


def ensure_unzipped(raw_dir):
    zips = [f for f in os.listdir(raw_dir) if f.lower().endswith('.zip')]
    if not zips:
        return
    print(f"Found {len(zips)} zip file(s), extracting...")
    for z in zips:
        zpath = os.path.join(raw_dir, z)
        extract_to = os.path.join(raw_dir, os.path.splitext(z)[0])
        if os.path.exists(extract_to):
            print(f"  {z} -> already extracted, skip")
            continue
        with zipfile.ZipFile(zpath, 'r') as zf:
            zf.extractall(extract_to)
        print(f"  {z} -> {extract_to}")
    print("Extraction complete.")


def get_inner_dir(parent_dir):
    """
    ISIC2018 解压后嵌套了一层同名子文件夹：
    Training_Input/Training_Input/xxx.jpg
    返回第二层目录
    """
    if not os.path.isdir(parent_dir):
        return None
    # 查找 parent_dir 下的子文件夹（排除隐藏文件）
    subdirs = [d for d in os.listdir(parent_dir)
               if os.path.isdir(os.path.join(parent_dir, d)) and not d.startswith('.')]
    if len(subdirs) == 1:
        return os.path.join(parent_dir, subdirs[0])
    elif len(subdirs) > 1:
        # 如果有多个，找包含实际数据（图片或CSV）的那个
        for d in subdirs:
            inner = os.path.join(parent_dir, d)
            for f in os.listdir(inner):
                if f.endswith(('.jpg', '.jpeg', '.png', '.csv')):
                    return inner
    return parent_dir  # 兜底


def prepare_isic2018(raw_dir, output_dir):
    ensure_unzipped(raw_dir)

    splits = [
        ('train', 'ISIC2018_Task3_Training_Input', 'ISIC2018_Task3_Training_GroundTruth'),
        ('val',   'ISIC2018_Task3_Validation_Input', 'ISIC2018_Task3_Validation_GroundTruth'),
        ('test',  'ISIC2018_Task3_Test_Input', 'ISIC2018_Task3_Test_GroundTruth'),
    ]

    for split_key, input_folder, gt_folder in splits:
        out_split_dir = os.path.join(output_dir, split_key)
        os.makedirs(os.path.join(out_split_dir, 'images'), exist_ok=True)

        # 第一层路径
        input_outer = os.path.join(raw_dir, input_folder)
        gt_outer = os.path.join(raw_dir, gt_folder)

        if not os.path.exists(input_outer):
            print(f"  [SKIP] {split_key}: {input_folder} not found")
            continue
        if not os.path.exists(gt_outer):
            print(f"  [SKIP] {split_key}: {gt_folder} not found")
            continue

        # 进入第二层（处理嵌套）
        input_dir = get_inner_dir(input_outer)
        gt_dir = get_inner_dir(gt_outer)

        # 在 gt_dir 里找 CSV
        gt_csv = None
        for f in os.listdir(gt_dir):
            if f.endswith('.csv'):
                gt_csv = os.path.join(gt_dir, f)
                break

        if gt_csv is None:
            print(f"  [SKIP] {split_key}: No CSV found in {gt_dir}")
            continue

        print(f"  [{split_key}] Input: {input_dir}")
        print(f"  [{split_key}] GroundTruth: {gt_csv}")

        # 复制图像
        copied = 0
        for img_name in os.listdir(input_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(input_dir, img_name)
                dst = os.path.join(out_split_dir, 'images', img_name)
                shutil.copy2(src, dst)
                copied += 1

        # 读取并转换 CSV
        df = pd.read_csv(gt_csv)
        id_col = df.columns[0]

        if 'label' in df.columns and len(df.columns) == 2:
            df_out = df.copy()
            df_out.columns = ['image', 'label']
        else:
            label_cols = [c for c in df.columns if c not in [id_col, 'image']]
            if len(label_cols) == 0:
                raise ValueError(f"No label columns in {gt_csv}")
            df['label'] = df[label_cols].values.argmax(axis=1)
            df_out = df[[id_col, 'label']].copy()
            df_out.columns = ['image', 'label']

        # 去掉扩展名
        df_out['image'] = df_out['image'].astype(str).str.replace(r'\.[^.]*$', '', regex=True)
        df_out.to_csv(os.path.join(out_split_dir, 'labels.csv'), index=False)

        print(f"  [{split_key}] Done: {copied} images, {len(df_out)} labels")

    print(f"\nDone! Output: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', default='./data/isic2018_raw')
    parser.add_argument('--output_dir', default='./data/isic2018')
    args = parser.parse_args()

    if not os.path.exists(args.raw_dir):
        print(f"Error: {args.raw_dir} does not exist.")
        exit(1)

    prepare_isic2018(args.raw_dir, args.output_dir)