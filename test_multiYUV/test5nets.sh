python EDSR_1cha_noUp_4seq.py > preds+origin-s26f300yuv/EDSR_1cha_noUp_4seq.txt &
python EDSR_2cha_noUp_4seq.py > preds+origin-s26f300yuv/EDSR_2cha_noUp_4seq.txt &
python EDSR_1cha_Up_4seq.py > preds+origin-s26f300yuv/EDSR_1cha_Up_4seq.txt &
python EDSR_2cha_Up_4seq.py > preds+origin-s26f300yuv/EDSR_2cha_Up_4seq.txt &
python test_srcnn_4seq.py > preds+origin-s26f300yuv/test_srcnn_4seq.txt