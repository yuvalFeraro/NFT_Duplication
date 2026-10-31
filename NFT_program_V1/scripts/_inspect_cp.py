from pathlib import Path

cp   = Path(r'C:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\data_sets\CryptoPunk_data')
orig = sorted((cp / 'original').glob('*.png'))
print(f"original/ : {len(orig)} images")
if orig:
    print(f"  sample: {orig[0].name}, {orig[-1].name}")

manip_dir = cp / 'mamipulations'
manip_items = sorted(manip_dir.iterdir())
print(f"\nmamipulations/ : {len(manip_items)} items")
for item in manip_items[:10]:
    if item.is_dir():
        files = list(item.glob('*.png'))
        print(f"  DIR  {item.name}/  : {len(files)} images" + (f"  e.g. {files[0].name}" if files else ""))
    else:
        print(f"  FILE {item.name}")

imgs_dir = Path(r'C:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\data_sets\imgs')
all_imgs = sorted(imgs_dir.glob('*.png'), key=lambda x: int(x.stem))
print(f"\nimgs/ : {len(all_imgs)} images  ({all_imgs[0].name} .. {all_imgs[-1].name})")
