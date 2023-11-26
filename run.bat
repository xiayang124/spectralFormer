
for %%i in (10) do (
      echo "runing Pavia"
      python demo.py --dataset="Pavia" --train_num=%%i --epoches=480 --train_time=4 --patches=7 --band_patches=3
)

for %%i in (10) do (
      echo "runing Honghu"
      python demo.py --dataset="Honghu" --train_num=%%i --epoches=300 --train_time=4 --patches=3 --band_patches=7
)