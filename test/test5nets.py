import os
seq26_paths_tex_lr = [
    "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq26/r1/S26C2AIR01_F32_GOF0_texture_rec_1280x1296_8bit_p420.yuv"
]
seq26_paths_tex_hr = [
    "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq26/r5/S26C2AIR05_F32_GOF0_texture_1280x1296_8bit_p420.yuv"
]
seq26_paths_occ = [
    "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq26/r1/S26C2AIR01_F32_GOF0_occupancy_rec_320x324_8bit_p420.yuv"
]
new_s26_r1_paths = [
    "home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq26/r5/S26C2AIR05_F32_dec_GOF0_texture_rec_1280x1296_8bit_p420.yuv","home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq26/r4/S26C2AIR04_F32_dec_GOF0_texture_rec_1280x1296_8bit_p420.yuv","home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq26/r3/S26C2AIR03_F32_dec_GOF0_texture_rec_1280x1296_8bit_p420.yuv","home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq26/r2/S26C2AIR02_F32_dec_GOF0_texture_rec_1280x1296_8bit_p420.yuv","home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq26/r1/S26C2AIR01_F32_dec_GOF0_texture_rec_1280x1296_8bit_p420.yuv"
]

seq23_paths_tex_lr = [
    "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq23/r1/S23C2AIR01_F32_GOF0_texture_rec_1280x1280_8bit_p420.yuv"]
seq23_paths_tex_hr = [
    "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq23/r1/S23C2AIR01_F32_GOF0_texture_1280x1280_8bit_p420.yuv"]
seq23_paths_occ = [
    "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq23/r1/S23C2AIR01_F32_GOF0_occupancy_rec_320x320_8bit_p420.yuv"]
new_s23_r1_paths = [
    "home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/EDSR_test_1channel_Upsample/test/seq23/r1_process/S23C2AIR01_F32_dec_GOF0_texture_rec_1280x1280_8bit_p420.yuv",
    "home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/EDSR_test_1channel_noUpsample/test/seq23/r1_process/S23C2AIR01_F32_dec_GOF0_texture_rec_1280x1280_8bit_p420.yuv",
    "home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/EDSR_test_2channel_Upsample/test/seq23/r1_process/S23C2AIR01_F32_dec_GOF0_texture_rec_1280x1280_8bit_p420.yuv",
    "home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/EDSR_test_2channel_noUpsample/test/seq23/r1_process/S23C2AIR01_F32_dec_GOF0_texture_rec_1280x1280_8bit_p420.yuv",
    "home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test_srcnn/test/seq23/r1_process/S23C2AIR01_F32_dec_GOF0_texture_rec_1280x1280_8bit_p420.yuv"
]

seq24_paths_tex_lr = [
    "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq24/r1/S24C2AIR01_F32_GOF0_texture_rec_1280x1344_8bit_p420.yuv"]
seq24_paths_tex_hr = [
    "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq24/r1/S24C2AIR01_F32_GOF0_texture_1280x1344_8bit_p420.yuv"]
seq24_paths_occ = [
    "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq24/r1/S24C2AIR01_F32_GOF0_occupancy_rec_320x336_8bit_p420.yuv"]
new_s24_r1_paths = [
    "home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/EDSR_test_1channel_Upsample/test/seq24/r1_process/S24C2AIR01_F32_dec_GOF0_texture_rec_1280x1344_8bit_p420.yuv"
    ,"home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/EDSR_test_1channel_noUpsample/test/seq24/r1_process/S24C2AIR01_F32_dec_GOF0_texture_rec_1280x1344_8bit_p420.yuv"
    ,"home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/EDSR_test_2channel_Upsample/test/seq24/r1_process/S24C2AIR01_F32_dec_GOF0_texture_rec_1280x1344_8bit_p420.yuv"
    ,"home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/EDSR_test_2channel_noUpsample/test/seq24/r1_process/S24C2AIR01_F32_dec_GOF0_texture_rec_1280x1344_8bit_p420.yuv"
    ,"home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test_srcnn/test/seq24/r1_process/S24C2AIR01_F32_dec_GOF0_texture_rec_1280x1344_8bit_p420.yuv"]

seq25_paths_tex_lr = [
    "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq25/r1/S25C2AIR01_F32_GOF0_texture_rec_1280x1280_8bit_p420.yuv"]
seq25_paths_tex_hr = [
    "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq25/r1/S25C2AIR01_F32_GOF0_texture_1280x1280_8bit_p420.yuv"]
seq25_paths_occ = [
    "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq25/r1/S25C2AIR01_F32_GOF0_occupancy_rec_320x320_8bit_p420.yuv"]
new_s25_r1_paths = [
    "home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/EDSR_test_1channel_Upsample/test/seq25/r1_process/S25C2AIR01_F32_dec_GOF0_texture_rec_1280x1280_8bit_p420.yuv",
    "home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/EDSR_test_1channel_noUpsample/test/seq25/r1_process/S25C2AIR01_F32_dec_GOF0_texture_rec_1280x1280_8bit_p420.yuv",
    "home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/EDSR_test_2channel_Upsample/test/seq25/r1_process/S25C2AIR01_F32_dec_GOF0_texture_rec_1280x1280_8bit_p420.yuv",
    "home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/EDSR_test_2channel_noUpsample/test/seq25/r1_process/S25C2AIR01_F32_dec_GOF0_texture_rec_1280x1280_8bit_p420.yuv",
    "home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test_srcnn/test/seq25/r1_process/S25C2AIR01_F32_dec_GOF0_texture_rec_1280x1280_8bit_p420.yuv"]
# 参考输入名称"S23C2AIR01_F1_GOF0_texture_rec_1280x1280_8bit_p420"
# 参考输出名称"S23C2AIR01_F1_dec_GOF0_texture_rec_1280x1280_8bit_p420"
def make_path(input_path):
    input_path_parts = input_path.split("/")
    input_yuv = input_path_parts[-1]
    input_yuv_parts = input_yuv.split("_")
    idx = len(input_yuv_parts[0])+len(input_yuv_parts[1])+2
    output_yuv = input_yuv[0:idx]+"dec_"+input_yuv[idx:]
    input_path_parts[-1] = output_yuv
    output_path = ""
    for t in input_path_parts:
        output_path = os.path.join(output_path, t)
    return output_path
def make_paths(input_paths):
    output_paths = []
    for path in input_paths:
        tpath = make_path(path)
        output_paths.append(tpath)
    return output_paths;

# 参考输入路径"/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq26/r1/S26C2AIR01_F32_GOF0_texture_rec_1280x1296_8bit_p420.yuv"
# 参考输出路径,有两个改动分别是r1->r1_process，以及增加了EDSR_test_1channel_Upsample，EDSR_test_1channel_noUpsample，EDSR_test_2channel_Upsample，EDSR_test_2channel_noUpsample，test_srcnn这5个文件名
# "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/EDSR_test_1channel_Upsample/seq26/r1_process/S26C2AIR01_F32_dec_GOF0_texture_rec_1280x1296_8bit_p420.yuv",
# "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/EDSR_test_1channel_noUpsample/seq26/r1_process/S26C2AIR01_F32_dec_GOF0_texture_rec_1280x1296_8bit_p420.yuv",
# "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/EDSR_test_2channel_Upsample/seq26/r1_process/S26C2AIR01_F32_dec_GOF0_texture_rec_1280x1296_8bit_p420.yuv",
# "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/EDSR_test_2channel_noUpsample/seq26/r1_process/S26C2AIR01_F32_dec_GOF0_texture_rec_1280x1296_8bit_p420.yuv",
# "/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/test_srcnn/seq26/r1_process/S26C2AIR01_F32_dec_GOF0_texture_rec_1280x1296_8bit_p420.yuv",




def make_path2(input_path):
    input_path_parts = input_path.split("/")
    input_yuv = input_path_parts[-1]
    input_yuv_parts = input_yuv.split("_")
    idx = len(input_yuv_parts[0])+len(input_yuv_parts[1])+2
    output_yuv = input_yuv[0:idx]+"dec_"+input_yuv[idx:]
    input_path_parts[-1] = output_yuv
    input_path_parts[-2] = input_path_parts[-2] + "_process"
    folder_list = ["EDSR_test_1channel_Upsample","EDSR_test_1channel_noUpsample","EDSR_test_2channel_Upsample","EDSR_test_2channel_noUpsample","test_srcnn"]
    output_paths = []
    for i in range(5):
        output_path = ""
        input_path_parts.insert(5,folder_list[i])
        for t in input_path_parts:
            output_path = os.path.join(output_path, t)
        output_paths.append(output_path)
        del input_path_parts[5]
    return output_paths
def make_paths2(input_paths):
    output_paths = []
    for path in input_paths:
        tpath = make_path2(path)
        output_paths.extend(tpath)
    return output_paths;

tex_lr = ["/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq23/r1/S23C2AIR01_F32_GOF0_texture_rec_1280x1280_8bit_p420.yuv","/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq24/r1/S24C2AIR01_F32_GOF0_texture_rec_1280x1344_8bit_p420.yuv","/home/wangdanying/VPCC_2021/mpeg-pcc-tmc2/test/seq25/r1/S25C2AIR01_F32_GOF0_texture_rec_1280x1280_8bit_p420.yuv"]





if __name__ == '__main__':
    output_paths = make_paths2(tex_lr)
    for path in output_paths:
        print(path)
