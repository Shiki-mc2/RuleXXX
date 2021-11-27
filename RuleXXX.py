# -*- coding: utf-8 -*-

import numpy as np
import cv2
import time
from numba import jit

@jit
def plot(row, col, ppu, cmg):
    img = np.full((row*ppu, col*ppu, 3), 0, dtype=np.uint8)
    blank = int(ppu/7)
    for i in range(row):
        for j in range(col):
            if cmg[i,j] > 0:
                img[i*ppu:(i+1)*ppu-blank, j*ppu:(j+1)*ppu-blank, 0] = 255
                img[i*ppu:(i+1)*ppu-blank, j*ppu:(j+1)*ppu-blank, 1] = 255
                img[i*ppu:(i+1)*ppu-blank, j*ppu:(j+1)*ppu-blank, 2] = 255
           
    return img

@jit
def run(row, col, cmg, i, bit):
    
    for j in range(col):
        if j == 0:
            state = [cmg[i,-1], cmg[i,0], cmg[i,1]]
        elif j < col - 1:
            state = list(cmg[i, j-1:j+2])
        else:
            state = [cmg[i,col-2], cmg[i,-1],cmg[i,0]]
            
        if   state == [1,1,1]:
            cmg[i+1, j] = bit[0]
        elif state == [1,1,0]: 
            cmg[i+1, j] = bit[1]
        elif state == [1,0,1]: 
            cmg[i+1, j] = bit[2]
        elif state == [1,0,0]: 
            cmg[i+1, j] = bit[3]
        elif state == [0,1,1]: 
            cmg[i+1, j] = bit[4]
        elif state == [0,1,0]: 
            cmg[i+1, j] = bit[5]
        elif state == [0,0,1]: 
            cmg[i+1, j] = bit[6]
        elif state == [0,0,0]: 
            cmg[i+1, j] = bit[7]
    return cmg
            
def main(rule):
    fps     = 30
    ppu     = 4
    row_cal = int(960/ppu)
    col_cal = int(np.ceil(int(1920/ppu)))
    n       = row_cal
    row_img = row_cal*ppu
    col_img = col_cal*ppu

    bit_str = format(rule,"08b")
    bit = []
    for j in range(8):
        bit.append(int(bit_str[j]))
    bit = np.array(bit)
    
    wname = f"Rule{rule:03d}"

    fourcc  = cv2.VideoWriter_fourcc(*"h264")
    video  = cv2.VideoWriter(f"{wname}.mp4", fourcc, fps, (col_img, row_img))
        
    cmg = np.zeros((row_cal, col_cal),dtype=np.uint32)
    cmg[0, col_cal//2] = 1

    sw1 = time.perf_counter()
    for i in range(n):
        img = plot(row_cal, col_cal, ppu, cmg)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video.write(img)
        
        if i == 0:
            cv2.imwrite(f"{wname}_start.png",img)
        
        sw2 = time.perf_counter()
        print(f"{i:4d}", f"{(sw2-sw1):.4f}")
        sw1 = sw2
        
        if i < n - 1:
            cmg = run(row_cal, col_cal, cmg, i, bit)
            
    cv2.imwrite(f"{wname}_end.png",img)
    video.release()
   
if __name__ == "__main__":
    main(90)
