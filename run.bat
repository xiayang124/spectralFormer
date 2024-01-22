for %%i in (40, 50) do (
      echo "runing Honghu"
      python demo.py --dataset="Honghu" --train_num=%%i --epoches=300 --train_time=3 --patches=3 --band_patches=7
)