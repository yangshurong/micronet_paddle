export DATA_PATH=C:\\Users\\Administrator\\Desktop\\micronet-main\\quant_100imgs\\val
# export DATA_PATH=C:\\Users\\Administrator\\Desktop\\Mydata\\test
export OUTPUT_PATH=C:\\Users\\Administrator\\Desktop\\micronet-main\\imagenet
export WEIGHT_PATH=C:\\Users\\Administrator\\Desktop\\micronet-main\\backbone\\micronet-m3.pth

CUDA_VISIBLE_DEVICES=0 python C:\\Users\\Administrator\\Desktop\\micronet-main\\main.py --arch MicroNet -d $DATA_PATH -c $OUTPUT_PATH --input-size 224 -b 16 -e --weight $WEIGHT_PATH \
                                                         MODEL.MICRONETS.BLOCK DYMicroBlock \
                                                         MODEL.MICRONETS.NET_CONFIG msnx_dy12_exp6_20M_020 \
                                                         MODEL.MICRONETS.STEM_CH 12 \
                                                         MODEL.MICRONETS.STEM_GROUPS 4,3\
                                                         MODEL.MICRONETS.STEM_DILATION 1 \
                                                         MODEL.MICRONETS.STEM_MODE spatialsepsf \
                                                         MODEL.MICRONETS.OUT_CH 1024 \
                                                         MODEL.MICRONETS.DEPTHSEP True \
                                                         MODEL.MICRONETS.POINTWISE group \
                                                         MODEL.MICRONETS.DROPOUT 0.1 \
                                                         MODEL.ACTIVATION.MODULE DYShiftMax \
                                                         MODEL.ACTIVATION.ACT_MAX 2.0 \
                                                         MODEL.ACTIVATION.LINEARSE_BIAS False \
							 MODEL.ACTIVATION.INIT_A_BLOCK3 1.0,0.0 \
                                                         MODEL.ACTIVATION.INIT_A 1.0,0.5 \
                                                         MODEL.ACTIVATION.INIT_B 0.0,0.5 \
                                                         MODEL.ACTIVATION.REDUCTION 8 \
                                                         MODEL.MICRONETS.SHUFFLE True \

