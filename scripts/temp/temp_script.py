from pathlib import Path
import shutil

reference_0 = Path(r"C:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\data_sets\DISC-21\2409_data\references_0")
disc21 = Path(r"C:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\NFT_program_V2\Data_sets\DISC-21")
count = 0


for file in reference_0.iterdir():
    if (disc21/file.name).exists():
        continue
    else:
        shutil.copy(file, disc21/file.name)
        count+=1
        if count == 516:
            break

print("done")