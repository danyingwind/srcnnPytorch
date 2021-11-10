# python test_yuv.py "--weights-file" "/home/wangdanying/SRCNN/srcnnPytorch/debug/trainLog/x1/best.pth" "--lr-tex-yuv-path" "/home/wangdanying/SRCNN/yuv_to_get_dataset/R1_yuv_rec/texture/seq23/S23C2AIR01_F300_GOF9_texture_rec_1280x1280_8bit_p420.yuv" "--lr-occ-yuv-path" "/home/wangdanying/SRCNN/yuv_to_get_dataset/R1_yuv_rec/occupancy/seq23/S23C2AIR01_F300_GOF9_occupancy_rec_320x320_8bit_p420.yuv" "--hr-tex-yuv-path" "/home/wangdanying/SRCNN/yuv_to_get_dataset/R5_yuv/texture/seq23/S23C2AIR05_F300_GOF9_texture_1280x1280_8bit_p420.yuv" "--outputLR-path" "/home/wangdanying/SRCNN/yuv_to_get_dataset/LR_UVchannel.yuv" "--outputHR-path" "/home/wangdanying/SRCNN/yuv_to_get_dataset/HR_UVchannel.yuv"

python test_yuv.py "--weights-file" "/home/wangdanying/SRCNN/srcnnPytorch/debug/trainLog/x1/best.pth" "--lr-tex-yuv-path" "/home/wangdanying/SRCNN/yuv_to_test_network/R1_yuv_rec/texture/seq23/S23C2AIR01_F1_GOFp_texture_1280x1280_8bit_p420.yuv" "--lr-occ-yuv-path" "/home/wangdanying/SRCNN/yuv_to_test_network/R1_yuv_rec/occupancy/seq23/S23C2AIR01_F1_GOFp_occupancy_320x320_8bit_p420.yuv" "--hr-tex-yuv-path" "/home/wangdanying/SRCNN/yuv_to_test_network/R1_yuv_rec/texture/seq23/S23C2AIR01_F1_GOFp_texture_1280x1280_8bit_p420.yuv"  "--outputLR-path" "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq23/r1_process/LR_UVchannel.yuv" "--outputHR-path" "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq23/r1_process/HR_UVchannel.yuv"





# /home/wangdanying/SRCNN/srcnnPytorch/0-bestparas/srcnnEDSR_res3_feats6_best.pth