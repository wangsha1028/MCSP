
from PIL import Image
import numpy as np
import math
import time
#需要嵌入的信息,用整形0,1两种数值，分别表示二进制的0,1
np.random.seed(1203)
s_data = np.random.randint(0,2,18000)

# 从路径打开
fname = r"E:\office files\投稿\信息隐藏论文\图片\512\1.png"
img = Image.open(fname,"r")
img = img.convert('L')
#img.show()

# 将二维数组，转换为一维数组
img_array1 = np.array(img)
img_array2 = img_array1.reshape(img_array1.shape[0] * img_array1.shape[1])
#print(img_array2)
# 将二维数组，转换为一维数组
img_array3 = img_array1.flatten()
#print(img_array3)


#metrics
#PSNR
def PSNR(image_array1,image_array2):
    #输入为两个图像数组，一维，大小相同
    assert(np.size(image_array1) == np.size(image_array2))
    n = np.size(image_array1)
    assert(n > 0)
    MSE = 0.0
    for i in range(0,n):
        MSE+=math.pow(int(image_array1[i]) - int(image_array2[i]),2)
    MSE = MSE / n 
    if MSE > 0:
        rtnPSNR = 10 * math.log10(255 * 255 / MSE)
    else:
        rtnPSNR = 100
    return rtnPSNR

def CSNR(image_array1,image_array2):
    #输入为两个图像数组，一维，大小相同
    assert(np.size(image_array1) == np.size(image_array2))
    n = np.size(image_array1)
    assert(n > 0)
    MSE = 0
    P2 = 0
    for i in range(0,n):
        MSE+=math.pow(image_array1[i] - image_array2[i],2)
        P2+=math.pow(image_array1[i],2)
    MSE = MSE 
    if MSE > 0:
        rtnCSNR = 10 * math.log10(P2 / MSE)
    else:
        rtnCSNR = 100
    return rtnCSNR

#十进制转二进制
#将一个十进制数x转换为n个bit的二进制数,高位在前
def dec2bin_higher_ahead(x,n):
    b_array1 = np.zeros(n)
    for i in range(0,n ,1):
        b_array1[i] = int(x % 2)
        x = x // 2
    b_array2 = np.zeros(n)
    for i in range(0,n ,1):
        b_array2[i] = b_array1[n - i] # n-1-i ？
    return b_array2

#十进制转二进制
#将一个十进制数x转换为n个bit的二进制数,低位在前
def dec2bin_lower_ahead(y,n):
    x = y
    b_array1 = np.zeros(n)
    for i in range(0,n ,1):
        b_array1[i] = int(x % 2)
        x = x // 2

    return b_array1

#信息隐藏算法

# ALGORITHM: EMD_2006 方法
def EMD_2006(image_array,secret_string,n):
    #image_array:输入的一维图像数组
    #n为一组像素的数量,我理解n只能取2，4,8,16等值，取其他值会导致嵌入的bit数不好确定
    assert(n == 2 or n == 4 or n == 8 or n == 16 or n == 32 or n == 64)
    moshu = 2 * n + 1  #模数的底
    #分成n个像素一组
    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups,n))
    i = 0
    while (i < num_pixel_groups):
        for j in range(0,n):
            if(i * n + j < image_array.size):
                 pixels_group[i,j] = image_array[i * n + j]
        i = i + 1
    #每一个像素组计算出一个fG值
    fG_array = np.zeros(num_pixel_groups)
    for i in range(0,num_pixel_groups):
        fG = 0
        for j in range(0,n):
            fG += (j + 1) * pixels_group[i,j] 
        fG_array[i] = fG % moshu
    #-----------------------------------------------------------------------------------
    #从待嵌入bit串数据中取出m个比特，作为一组。m=math.log((2*n),2),以2为底的对数
    m = int(math.log((2 * n),2))
    #分组
    num_secret_groups = math.ceil(secret_string.size / m)
    secret_group = np.zeros((num_secret_groups,m))
    i = 0
    while (i < num_secret_groups):
        for j in range(0,m):
            if(i * m + j < s_data.size):
                 secret_group[i,j] = s_data[i * m + j]
        i = i + 1
    #-----------------------------------------------------------------------------------

    #一组pixels_group嵌入一组secret_group的信息，多了不能嵌入,最后一组pixel不用于嵌入以防止错误
    assert(np.shape(secret_group)[0] <= np.shape(pixels_group)[0] - 1)    
    #每一组secret_group计算得到一个d值，d为（2n+1）进制的一个数
    d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        #d代表一个（2n+1）进制的一个数
        d = 0
        for j in range(0,m):
            d += secret_group[i,j] * (2 ** (m - 1 - j))
            d_array[i] = d
    #-----------------------------------------------------------------------------------
    #开始进行嵌入
    embeded_pixels_group = pixels_group.copy()
    for i in range(0,num_secret_groups):
        d = d_array[i]
        fG = fG_array[i]
        j = int(d - fG) % moshu
        if (j > 0): #如果为0的话，则不进行修改
            if (j <= n) :
                embeded_pixels_group[i , j - 1]+=1
            else:
                embeded_pixels_group[i ,(2 * n + 1 - j) - 1]+=-1

    #-----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        fG = 0
        for j in range(0,n):
            fG += (j + 1) * embeded_pixels_group[i,j] 
        recover_d_array[i] = fG % moshu

    # 恢复出的和以前的应该是一致的
    assert(int((recover_d_array - d_array).sum()) == 0)

    #-----------------------------------------------------------------------------------
    #输出图像
    img_out = embeded_pixels_group.flatten()
    img_out = img_out[:512 * 512] #取前面的pixel
    #计算PSNR
    img_array_out = img_out.copy()
    psnr = PSNR(image_array,img_array_out)
    print('EMD_2006 PSNR: %.2f' % psnr)
    #csnr = CSNR(image_array,img_array_out)
    #print('EMD_2006 CSNR: %.2f' % csnr)
    #重组图像
    img_out = img_out.reshape(512,512)
    img_out = Image.fromarray(img_out)
    #img_out.show()



    return 0

# ALGORITHM: LWC_2007 方法
def LWC_2007(image_array,secret_string,n=2):
    #image_array:输入的一维图像数组
    #分成2个像素一组
    n = 2
    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups,n))
    i = 0
    while (i < num_pixel_groups):
        for j in range(0,n):
            if(i * n + j < image_array.size):
                 pixels_group[i,j] = image_array[i * n + j]
        i = i + 1
    #-----------------------------------------------------------------------------------
    #从待嵌入bit串数据中取出3个比特，作为一组
    #分组
    num_secret_groups = math.ceil(secret_string.size / 3)
    secret_group = np.zeros((num_secret_groups,3))
    i = 0
    while (i < num_secret_groups):
        for j in range(0,3):
            if(i * 3 + j < s_data.size):
                 secret_group[i,j] = s_data[i * 3 + j]
        i = i + 1

        #一组pixels_group嵌入一组secret_group的信息，多了不能嵌入,最后一组pixel不用于嵌入以防止错误
    assert(np.shape(secret_group)[0] <= np.shape(pixels_group)[0] - 1)    
    #每一组secret_group计算得到一个d值，d为十进制的一个数
    d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        #d代表一个（2n+1）进制的一个数
        d = 0
        for j in range(0,3):
            d += secret_group[i,j] * (2 ** (3 - 1 - j))
        d_array[i] = d

    #-----------------------------------------------------------------------------------
    #开始进行嵌入
    embeded_pixels_group = pixels_group.copy()
    for i in range(0,num_secret_groups):
        fe = 1 * pixels_group[i,0] + 3 * pixels_group[i,1]
        fe = fe % 8   
        if (int(fe) == int(d_array[i])):
            #不修改
            embeded_pixels_group[i,0] = pixels_group[i,0]
            embeded_pixels_group[i,1] = pixels_group[i,1]
        else:
            fe = 1 * (pixels_group[i,0] + 1) + 3 * pixels_group[i,1]
            fe = fe % 8 
            if (int(fe) == int(d_array[i])):
                embeded_pixels_group[i,0]+=1
            else:
                fe = 1 * (pixels_group[i,0] - 1) + 3 * pixels_group[i,1]
                fe = fe % 8 
                if (int(fe) == int(d_array[i])):
                    embeded_pixels_group[i,0]+=-1
                else:
                    fe = 1 * pixels_group[i,0] + 3 * (pixels_group[i,1] + 1)
                    fe = fe % 8 
                    if (int(fe) == int(d_array[i])):
                        embeded_pixels_group[i,1]+=1
                    else:
                        fe = 1 * pixels_group[i,0] + 3 * (pixels_group[i,1] - 1)
                        fe = fe % 8 
                        if (int(fe) == int(d_array[i])):
                            embeded_pixels_group[i,1]+=-1
                        else:
                            fe = 1 * (pixels_group[i,0] + 1) + 3 * (pixels_group[i,1] + 1)
                            fe = fe % 8 
                            if (int(fe) == int(d_array[i])):
                                embeded_pixels_group[i,0]+=1
                                embeded_pixels_group[i,1]+=1
                            else:
                                fe = 1 * (pixels_group[i,0] + 1) + 3 * (pixels_group[i,1] - 1)
                                fe = fe % 8 
                                if (int(fe) == int(d_array[i])):
                                    embeded_pixels_group[i,0]+=1
                                    embeded_pixels_group[i,1]+=-1
                                else:
                                    fe = 1 * (pixels_group[i,0] - 1) + 3 * (pixels_group[i,1] + 1)
                                    fe = fe % 8 
                                    if (int(fe) == int(d_array[i])):
                                        embeded_pixels_group[i,0]+=-1
                                        embeded_pixels_group[i,1]+=1
    #-----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        fe = 1 * embeded_pixels_group[i,0] + 3 * embeded_pixels_group[i,1] 
        recover_d_array[i] = fe % 8

    # 恢复出的和以前的应该是一致的
    assert(int((recover_d_array - d_array).sum()) == 0)

    #-----------------------------------------------------------------------------------
    #输出图像
    img_out = embeded_pixels_group.flatten()
    img_out = img_out[:512 * 512] #取前面的pixel
    #计算PSNR
    img_array_out = img_out.copy()
    psnr = PSNR(image_array,img_array_out)
    print('LWC_2007 PSNR: %.2f' % psnr)
    #csnr = CSNR(image_array,img_array_out)
    #print('LWC_2007 CSNR: %.2f' % csnr)
    #重组图像
    img_out = img_out.reshape(512,512)
    img_out = Image.fromarray(img_out)
    #img_out.show()

    return 0

# ALGORITHM: JY_2009 方法
def JY_2009(image_array,secret_string,n=1):
    #image_array:输入的一维图像数组
    #n = 1 #此算法在一个像素中嵌入
    num_pixel_groups = image_array.size
    #-----------------------------------------------------------------------------------
    #从待嵌入bit串数据中取出k个比特，作为一组
    k = 1
    moshu = 2 * k + 1
    #分组
    num_secret_groups = math.ceil(secret_string.size / k)
    secret_group = np.zeros((num_secret_groups,k))
    for i in range(0,num_secret_groups,1):
        for j in range(0,k,1):
            if(i * k + j < secret_string.size):
                 secret_group[i,j] = s_data[i * k + j]

    #一组pixels_group嵌入一组secret_group的信息，多了不能嵌入,最后一组pixel不用于嵌入以防止错误
    assert(num_pixel_groups > num_secret_groups)    
    #每一组secret_group计算得到一个d值，d为十进制的一个数
    secret_d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        #d代表一个（2n+1）进制的一个数
        d = 0
        for j in range(0,k,1):
            d += secret_group[i,j] * (2 ** j)  #将secret视为低位在前
        secret_d_array[i] = d

    #-----------------------------------------------------------------------------------
    #开始进行嵌入
    embeded_pixels_group = image_array.copy()
    pixels_group = image_array.copy()
    for i in range(0,num_secret_groups):
        x = 0
        if (pixels_group[i] >= 0) and (pixels_group[i] <= 1):
            for x in range(0,moshu,1):
                fg = (pixels_group[i] + x) % moshu
                if int(fg) == int(secret_d_array[i]):
                    embeded_pixels_group[i] = pixels_group[i] + x
        else:
            if (pixels_group[i] >= 254) and (pixels_group[i] <= 254):
                for x in range(-1 * moshu + 1,1,1):
                    fg = (pixels_group[i] + x) % moshu
                    if int(fg) == int(secret_d_array[i]):
                        embeded_pixels_group[i] = pixels_group[i] + x
            else:
                for x in range(-1 * moshu,moshu + 1,1):
                    fg = (pixels_group[i] + x) % moshu
                    if int(fg) == int(secret_d_array[i]):
                        embeded_pixels_group[i] = pixels_group[i] + x
                    
    #-----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        recover_d_array[i] = embeded_pixels_group[i] % moshu

    # 恢复出的和以前的应该是一致的
    assert(int((recover_d_array - secret_d_array).sum()) == 0)

    #-----------------------------------------------------------------------------------
    #输出图像
    img_out = embeded_pixels_group.flatten()
    img_out = img_out[:512 * 512] #取前面的pixel
    #计算PSNR
    img_array_out = img_out.copy()
    psnr = PSNR(image_array,img_array_out)
    print('JY_2009 PSNR: %.2f' % psnr)
    #csnr = CSNR(image_array,img_array_out)
    #print('JY_2009 CSNR: %.2f' % csnr)
    #重组图像
    img_out = img_out.reshape(512,512)
    img_out = Image.fromarray(img_out)
    #img_out.show()

    return 0

# ALGORITHM: GEMD_2013 方法
def GEMD_2013(image_array,secret_string,n):
    #image_array:输入的一维图像数组
    #n为一组像素的数量

    #将一个十进制数x转换为（n+1）个bit的二进制数,低位在前
    def dec2bin_lower_ahead(x,n):
        b_array1 = np.zeros(n + 1)
        for i in range(0,n + 1,1):
            b_array1[i] = int(x % 2)
            x = x // 2
        # 没有这个功能 b_array.reverse()
        #b_array2 = np.zeros(n + 1)
        #for i in range(0,n + 1,1):
        #    b_array2[i] = b_array1[n - i]
        return b_array1

    moshu = 2 ** (n + 1)  #模数的底
    #分成n个像素一组
    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups,n))
    i = 0
    while (i < num_pixel_groups):
        for j in range(0,n):
            if(i * n + j < image_array.size):
                 pixels_group[i,j] = image_array[i * n + j]
        i = i + 1
    #每一个像素组计算出一个fGEMD值
    fGEMD_array = np.zeros(num_pixel_groups)
    for i in range(0,num_pixel_groups):
        fGEMD = 0
        for j in range(0,n):
            fGEMD += (2 ** (j + 1) - 1) * pixels_group[i,j] 
        fGEMD_array[i] = fGEMD % moshu
    #-----------------------------------------------------------------------------------
    #从待嵌入bit串数据中取出m个比特，作为一组
    m = n + 1
    #分组
    num_secret_groups = math.ceil(secret_string.size / m)
    secret_group = np.zeros((num_secret_groups,m))
    i = 0
    while (i < num_secret_groups):
        for j in range(0,m):
            if(i * m + j < s_data.size):
                 secret_group[i,j] = s_data[i * m + j]
        i = i + 1
    #-----------------------------------------------------------------------------------

    #一组pixels_group嵌入一组secret_group的信息，多了不能嵌入,最后一组pixel不用于嵌入以防止错误
    assert(np.shape(secret_group)[0] <= np.shape(pixels_group)[0] - 1)    
    #每一组secret_group计算得到一个d值，d为（2n+1）进制的一个数
    d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        #d代表一个十进制的一个数
        d = 0
        for j in range(0,m):
            d += secret_group[i,j] * (2 ** (m - 1 - j))
        d_array[i] = d
    #-----------------------------------------------------------------------------------
    #开始进行嵌入
    embeded_pixels_group = pixels_group.copy()
    diff_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        d = d_array[i]
        fGEMD = fGEMD_array[i]
        assert(fGEMD < 33)
        diff_array[i] = int(d - fGEMD) % moshu

    for i in range(0,num_secret_groups):
        diff = int(diff_array[i])
        if (diff == 2 ** n) :
            embeded_pixels_group[i,0] = pixels_group[i,0] + 1
            embeded_pixels_group[i,n - 1] = pixels_group[i,n - 1] + 1
        if  (diff > 0) and (diff < 2 ** n) :
            #将diff转换为（n+1）个二进制数
            b_array = np.zeros(n + 1)
            b_array = dec2bin_lower_ahead(diff,n)
            for j in range(n,0,-1): #倒序
                if (int(b_array[j]) == 0) and (int(b_array[j - 1]) == 1) :
                    embeded_pixels_group[i,j - 1] = pixels_group[i,j - 1] + 1
                if (int(b_array[j]) == 1) and (int(b_array[j - 1]) == 0) :
                    embeded_pixels_group[i,j - 1] = pixels_group[i,j - 1] - 1
        if (diff > 2 ** n) and (diff < 2 ** (n + 1)) :
            #将diff转换为（n+1）个二进制数
            b_array = np.zeros(n + 1)
            b_array = dec2bin_lower_ahead(2 ** (n + 1) - diff,n)
            for j in range(n,0,-1): #倒序
                if (int(b_array[j]) == 0) and (int(b_array[j - 1]) == 1) :
                    embeded_pixels_group[i,j - 1] = pixels_group[i,j - 1] - 1
                if (int(b_array[j]) == 1) and (int(b_array[j - 1]) == 0) :
                    embeded_pixels_group[i,j - 1] = pixels_group[i,j - 1] + 1
   

    #-----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        fGEMD = 0
        for j in range(0,n):
            fGEMD += (2 ** (j + 1) - 1) * embeded_pixels_group[i,j] 
        recover_d_array[i] = fGEMD % moshu

    # 恢复出的和以前的应该是一致的
    assert(int((recover_d_array - d_array).sum()) == 0)

    #-----------------------------------------------------------------------------------
    #输出图像
    img_out = embeded_pixels_group.flatten()
    img_out = img_out[:512 * 512] #取前面的pixel
    #计算PSNR
    img_array_out = img_out.copy()
    psnr = PSNR(image_array,img_array_out)
    print('GEMD_2013 PSNR: %.2f' % psnr)
    #csnr = CSNR(image_array,img_array_out)
    #print('GEMD_2013 CSNR: %.2f' % csnr)
    #重组图像
    img_out = img_out.reshape(512,512)
    img_out = Image.fromarray(img_out)
    #img_out.show()

    return 0

# ALGORITHM: KKWW_2016 方法
def KKWW_2016(image_array,secret_string,n):
    #image_array:输入的一维图像数组
    #n为一组像素的数量

    #将一个十进制数x转换为n个bit的（2**k）进制数,低位在前
    def dec_2k_lower_ahead(x,n):
        b_array1 = np.zeros(n)
        for i in range(0,n,1):
            b_array1[i] = int(x) % (2 ** k)
            x = x // (2 ** k)
        # 没有这个功能 b_array.reverse()
        #b_array2 = np.zeros(n + 1)
        #for i in range(0,n + 1,1):
        #    b_array2[i] = b_array1[n - i]
        return b_array1
    
    k = 3 #k是一个参数，表示一个pixel嵌入多少个bit
    moshu = 2 ** (n * k + 1)  #模数的底
    #参数c
    c_array = np.zeros(n)
    c_array[0] = 1
    for i in range(1,n):
        c_array[i] = (2 ** k) * c_array[i - 1] + 1

    #分成n个像素一组
    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups,n))
    i = 0
    while (i < num_pixel_groups):
        for j in range(0,n):
            if(i * n + j < image_array.size):
                 pixels_group[i,j] = image_array[i * n + j]
        i = i + 1
    #每一像素组计算出一个fG值
    fG_array = np.zeros(num_pixel_groups)
    for i in range(0,num_pixel_groups):
        fG = 0
        for j in range(0,n):
            fG += c_array[j] * pixels_group[i,j] 
        fG_array[i] = int(fG) % moshu
    #-----------------------------------------------------------------------------------
    #从待嵌入bit串数据中取出n*k+1个比特，作为一组
    m = n * k + 1
    #分组
    num_secret_groups = math.ceil(secret_string.size / m)
    secret_group = np.zeros((num_secret_groups,m))
    for i in range(0,num_secret_groups):
        for j in range(0,m):
            if(i * m + j < secret_string.size):
                 secret_group[i,j] = secret_string[i * m + j]
    #-----------------------------------------------------------------------------------
    #一组pixels_group嵌入一组secret_group的信息，多了不能嵌入,最后一组pixel不用于嵌入以防止错误
    assert(np.shape(secret_group)[0] <= np.shape(pixels_group)[0] - 1)    
    #每一组secret_group计算得到一个d值，d为（2n+1）进制的一个数
    d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        #d代表一个十进制的一个数
        d = 0
        for j in range(0,m):
            d += secret_group[i,j] * (2 ** (m - 1 - j))
        d_array[i] = d
   #-----------------------------------------------------------------------------------
    #开始进行嵌入
    embeded_pixels_group = pixels_group.copy()
    diff_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        d = d_array[i]
        fG = fG_array[i]
        diff_array[i] = int(d - fG) % moshu
    #开始
    for i in range(0,num_secret_groups):
        #print(i,end=" ")
        diff = diff_array[i]
        if int(diff) > 0:
            if int(diff) == 2 ** (n * k):
                embeded_pixels_group[i,n - 1] = pixels_group[i,n - 1] + (2 ** k - 1)
                embeded_pixels_group[i,0] = pixels_group[i,0] + 1
            else:
                if int(diff) < 2 ** (n * k):
                   d_transfromed = np.zeros(n)
                   d_transfromed = dec_2k_lower_ahead(diff,n)
                   for j in range(n - 1,-1,-1):
                       embeded_pixels_group[i,j] = embeded_pixels_group[i,j] + d_transfromed[j]
                       if j > 0:
                            embeded_pixels_group[i,j - 1] = embeded_pixels_group[i,j - 1] - d_transfromed[j]
                else:
                    if int(diff) > 2 ** (n * k):
                       d_transfromed = np.zeros(n)
                       d_transfromed = dec_2k_lower_ahead((2 ** (n * k + 1)) - diff,n)
                       for j in range(n - 1,-1,-1):
                           embeded_pixels_group[i,j] = embeded_pixels_group[i,j] - d_transfromed[j]
                           if j > 0:
                               embeded_pixels_group[i,j - 1] = embeded_pixels_group[i,j - 1] + d_transfromed[j]
    #-----------------------------------------------------------------------------------
    pixels_changed = num_secret_groups * n
    #-----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        fG = 0
        for j in range(0,n):
            fG += c_array[j] * embeded_pixels_group[i,j] 
        recover_d_array[i] = int(fG) % moshu
        #assert(int((recover_d_array[i] - d_array[i]).sum()) == 0)

    # 恢复出的和以前的应该是一致的
    assert(int((recover_d_array - d_array).sum()) == 0)
    #-----------------------------------------------------------------------------------
    #输出图像
    img_out = embeded_pixels_group.flatten()
    img_out = img_out[:512 * 512] #取前面的pixel
    #计算PSNR
    img_array_out = img_out.copy()
    psnr = PSNR(image_array,img_array_out)
    print('KKWW_2016 PSNR: %.2f' % psnr)
    #csnr = CSNR(image_array,img_array_out)
    #print('KKWW_2016 CSNR: %.2f' % csnr)
    print('KKWW_2016 pixels used: %d' % pixels_changed)
    #重组图像
    img_out = img_out.reshape(512,512)
    img_out = Image.fromarray(img_out)
    #img_out.show()

    return 0

# ALGORITHM: SB_2019 方法
def SB_2019(image_array,secret_string,n=1):
    #image_array:输入的一维图像数组
    #n = 1 #此算法在一个像素中嵌入
    num_pixel_groups = image_array.size
    #-----------------------------------------------------------------------------------
    #从待嵌入bit串数据中取出k个比特，作为一组
    k = 5 #k应该是可调的
    moshu = k* k 
    #分组
    num_secret_groups = math.ceil(secret_string.size / k)
    secret_group = np.zeros((num_secret_groups,k))
    for i in range(0,num_secret_groups,1):
        for j in range(0,k,1):
            if(i * k + j < secret_string.size):
                 secret_group[i,j] = s_data[i * k + j]

    #一组pixels_group嵌入一组secret_group的信息，多了不能嵌入,最后一组pixel不用于嵌入以防止错误
    assert(num_pixel_groups > num_secret_groups)    
    #每一组secret_group计算得到一个d值，d为十进制的一个数
    secret_d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups,1):
        #d代表一个（2n+1）进制的一个数
        d = 0
        for j in range(0,k,1):
            d += secret_group[i,j] * (2 ** j)  #将secret视为低位在前
        secret_d_array[i] = d

    #-----------------------------------------------------------------------------------
    #开始进行嵌入
    embeded_pixels_group = image_array.copy()
    pixels_group = image_array.copy()
    for i in range(0,num_secret_groups):
        x = 0
        for x in range(-1*math.floor(moshu/2),math.floor(moshu/2)+1,1):
            f = (pixels_group[i] + x) % moshu
            if int(f) == int(secret_d_array[i]):
                embeded_pixels_group[i] = pixels_group[i] + x
                    
    #-----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        recover_d_array[i] = embeded_pixels_group[i] % moshu

    # 恢复出的和以前的应该是一致的
    assert(int((recover_d_array - secret_d_array).sum()) == 0)

    #-----------------------------------------------------------------------------------
    #输出图像
    img_out = embeded_pixels_group.flatten()
    img_out = img_out[:512 * 512] #取前面的pixel
    #计算PSNR
    img_array_out = img_out.copy()
    psnr = PSNR(image_array,img_array_out)
    print('SB_2019 PSNR: %.2f' % psnr)
    #csnr = CSNR(image_array,img_array_out)
    #print('SB_2019 CSNR: %.2f' % csnr)
    #重组图像
    img_out = img_out.reshape(512,512)
    img_out = Image.fromarray(img_out)
    img_out.show()

    return 0

def SB_2019_1(image_array, secret_string, n=1):
    # image_array:输入的一维图像数组
    # n = 1 #此算法在一个像素中嵌入
    num_pixel_groups = image_array.size
    # -----------------------------------------------------------------------------------
    # 从待嵌入bit串数据中取出k个比特，作为一组
    k = 2 # k应该是可调的
    moshu = 2**k
    # 分组
    num_secret_groups = math.ceil(secret_string.size / k)
    secret_group = np.zeros((num_secret_groups, k))
    for i in range(0, num_secret_groups, 1):
        for j in range(0, k, 1):
            if (i * k + j < secret_string.size):
                secret_group[i, j] = s_data[i * k + j]

    # 一组pixels_group嵌入一组secret_group的信息，多了不能嵌入,最后一组pixel不用于嵌入以防止错误
    assert (num_pixel_groups > num_secret_groups)
    # 每一组secret_group计算得到一个d值，d为十进制的一个数
    secret_d_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups, 1):
        # d代表一个（2n+1）进制的一个数
        d = 0
        for j in range(0, k, 1):
            d += secret_group[i, j] * (2 ** j)  # 将secret视为低位在前
        secret_d_array[i] = d
    distance = math.floor(num_pixel_groups / num_secret_groups)
    # -----------------------------------------------------------------------------------
    # 开始进行嵌入
    embeded_pixels_group = image_array.copy()
    pixels_group = image_array.copy()
    for i in range(0, num_secret_groups):
        x = 0
        for x in range(-1 * math.floor(moshu / 2), math.floor(moshu / 2) + 1, 1):
            f = (pixels_group[i*distance] + x) % moshu
            if int(f) == int(secret_d_array[i]):
                embeded_pixels_group[i*distance] = pixels_group[i*distance] + x

    # -----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0, num_secret_groups):
        recover_d_array[i] = embeded_pixels_group[i*distance] % moshu

    # 恢复出的和以前的应该是一致的
    assert (int((recover_d_array - secret_d_array).sum()) == 0)

    # -----------------------------------------------------------------------------------
    # 输出图像
    img_out = embeded_pixels_group.flatten()
    img_out = img_out[:512 * 512]  # 取前面的pixel
    # 计算PSNR
    img_array_out = img_out.copy()
    psnr = PSNR(image_array, img_array_out)
    print('SB_2019_1 PSNR: %.2f' % psnr)
    # csnr = CSNR(image_array,img_array_out)
    # print('SB_2019 CSNR: %.2f' % csnr)
    # 重组图像
    img_out = img_out.reshape(512, 512)
    img_out = Image.fromarray(img_out)
    img_out.show()

    return 0

#ALGORITHM: MaxPSNR5 our new 方法
def MaxPSNR5(image_array,secret_string,n=5):
    #image_array:输入的一维图像数组
    #n为一组像素的数量,在本算法中，固定为5
    n = 5
    #分成n个像素一组,保证整数组，不足的补零
    num_pixel_groups = int(image_array.size / n)
    while num_pixel_groups % n > 0:
        num_pixel_groups+=1
    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups,n))
    i = 0
    while (i < num_pixel_groups):
        for j in range(0,n):
            if(i * n + j < image_array.size):
                 pixels_group[i,j] = image_array[i * n + j]
        i = i + 1

    #-----------------------------------------------------------------------------------
    #从待嵌入bit串数据中取出k个比特，作为一组
    k = 4 #尝试固定值
    #分组
    num_secret_groups = int(secret_string.size / k)
    while num_secret_groups % k > 0:#保证整数组，不足的补零
        num_secret_groups+=1
    while num_secret_groups % 4 > 0: #保证整数组，不足的补零
        num_secret_groups+=1
    secret_group = np.zeros((num_secret_groups,k))
    i = 0
    while (i < num_secret_groups):
        for j in range(0,k):
            if(i * k + j < s_data.size):
                 secret_group[i,j] = s_data[i * k + j]
        i = i + 1

    #每一组secret_group计算得到一个d值，d为十进制
    d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        #d代表一个十进制的一个数
        d = 0
        for j in range(0,k):
            d += secret_group[i,j] * (2 ** (k - 1 - j))
        d_array[i] = d
    #-----------------------------------------------------------------------------------
    #确保能嵌入所有的秘密数据
    assert(num_pixel_groups * 4 >= num_secret_groups)
    #-----------------------------------------------------------------------------------
    #metrics
    def PSNR5(image_array1,image_array2):
        #输入为两个图像数组，一维，大小相同
        assert(np.size(image_array1) == np.size(image_array2))
        n = np.size(image_array1)
        assert(n > 0)
        MSE = 0
        for i in range(0,n):
            MSE+=math.pow(image_array1[i] - image_array2[i],2)
        MSE = MSE / n 
        if MSE > 0:
            rtnPSNR = 10 * math.log10(255 * 255 / MSE)
        else:
            rtnPSNR = 100
        return rtnPSNR
    #-----------------------------------------------------------------------------------

    #开始进行嵌入
    #每一组pixels_group，嵌入四组d_array
    embeded_pixels_group = pixels_group.copy()

    for i in range(0,int(num_secret_groups / 4),1):
        sected_index = 0 #选中的方案索引
        tmpCount = int('0b10111',2) + 1
        G5p = np.zeros((tmpCount,5))
        MaxPSNR5 = 0
        tmpPSNR5 = np.zeros(tmpCount)
        for possible_index in range(0,tmpCount): #计算所有可能的组合，看哪一个最合适
            if possible_index == int('0b00000',2):
                d_embeded1 = d_array[i * 4 + 0]
                d_embeded2 = d_array[i * 4 + 1]
                d_embeded3 = d_array[i * 4 + 2]
                d_embeded4 = d_array[i * 4 + 3]
                d_embeded5 = int('0b00000',2)
            if possible_index == int('0b00001',2):
                d_embeded1 = d_array[i * 4 + 0]
                d_embeded2 = d_array[i * 4 + 1]
                d_embeded3 = d_array[i * 4 + 3]
                d_embeded4 = d_array[i * 4 + 2]
                d_embeded5 = int('0b00001',2)
            if possible_index == int('0b00010',2):
                d_embeded1 = d_array[i * 4 + 0]
                d_embeded2 = d_array[i * 4 + 2]
                d_embeded3 = d_array[i * 4 + 1]
                d_embeded4 = d_array[i * 4 + 3]
                d_embeded5 = int('0b00010',2)
            if possible_index == int('0b00011',2):
                d_embeded1 = d_array[i * 4 + 0]
                d_embeded2 = d_array[i * 4 + 2]
                d_embeded3 = d_array[i * 4 + 3]
                d_embeded4 = d_array[i * 4 + 1]
                d_embeded5 = int('0b00011',2)
            if possible_index == int('0b00100',2):
                d_embeded1 = d_array[i * 4 + 0]
                d_embeded2 = d_array[i * 4 + 3]
                d_embeded3 = d_array[i * 4 + 1]
                d_embeded4 = d_array[i * 4 + 2]
                d_embeded5 = int('0b00100',2)
            if possible_index == int('0b00101',2):
                d_embeded1 = d_array[i * 4 + 0]
                d_embeded2 = d_array[i * 4 + 3]
                d_embeded3 = d_array[i * 4 + 2]
                d_embeded4 = d_array[i * 4 + 1]
                d_embeded5 = int('0b00101',2)
            if possible_index == int('0b00110',2):
                d_embeded1 = d_array[i * 4 + 1]
                d_embeded2 = d_array[i * 4 + 0]
                d_embeded3 = d_array[i * 4 + 2]
                d_embeded4 = d_array[i * 4 + 3]
                d_embeded5 = int('0b00110',2)
            if possible_index == int('0b00111',2):
                d_embeded1 = d_array[i * 4 + 1]
                d_embeded2 = d_array[i * 4 + 0]
                d_embeded3 = d_array[i * 4 + 3]
                d_embeded4 = d_array[i * 4 + 2]
                d_embeded5 = int('0b00111',2)
            if possible_index == int('0b01000',2):
                d_embeded1 = d_array[i * 4 + 1]
                d_embeded2 = d_array[i * 4 + 2]
                d_embeded3 = d_array[i * 4 + 0]
                d_embeded4 = d_array[i * 4 + 3]
                d_embeded5 = int('0b01000',2)
            if possible_index == int('0b01001',2):
                d_embeded1 = d_array[i * 4 + 1]
                d_embeded2 = d_array[i * 4 + 2]
                d_embeded3 = d_array[i * 4 + 3]
                d_embeded4 = d_array[i * 4 + 0]
                d_embeded5 = int('0b01001',2)
            if possible_index == int('0b01010',2):
                d_embeded1 = d_array[i * 4 + 1]
                d_embeded2 = d_array[i * 4 + 3]
                d_embeded3 = d_array[i * 4 + 0]
                d_embeded4 = d_array[i * 4 + 2]
                d_embeded5 = int('0b01010',2)
            if possible_index == int('0b01011',2):
                d_embeded1 = d_array[i * 4 + 1]
                d_embeded2 = d_array[i * 4 + 3]
                d_embeded3 = d_array[i * 4 + 2]
                d_embeded4 = d_array[i * 4 + 0]
                d_embeded5 = int('0b01011',2)
            if possible_index == int('0b01100',2):
                d_embeded1 = d_array[i * 4 + 2]
                d_embeded2 = d_array[i * 4 + 0]
                d_embeded3 = d_array[i * 4 + 1]
                d_embeded4 = d_array[i * 4 + 3]
                d_embeded5 = int('0b01100',2)
            if possible_index == int('0b01101',2):
                d_embeded1 = d_array[i * 4 + 2]
                d_embeded2 = d_array[i * 4 + 0]
                d_embeded3 = d_array[i * 4 + 3]
                d_embeded4 = d_array[i * 4 + 1]
                d_embeded5 = int('0b01101',2)
            if possible_index == int('0b01110',2):
                d_embeded1 = d_array[i * 4 + 2]
                d_embeded2 = d_array[i * 4 + 1]
                d_embeded3 = d_array[i * 4 + 0]
                d_embeded4 = d_array[i * 4 + 3]
                d_embeded5 = int('0b01110',2)
            if possible_index == int('0b01111',2):
                d_embeded1 = d_array[i * 4 + 2]
                d_embeded2 = d_array[i * 4 + 1]
                d_embeded3 = d_array[i * 4 + 3]
                d_embeded4 = d_array[i * 4 + 0]
                d_embeded5 = int('0b01111',2)
            if possible_index == int('0b10000',2):
                d_embeded1 = d_array[i * 4 + 2]
                d_embeded2 = d_array[i * 4 + 3]
                d_embeded3 = d_array[i * 4 + 0]
                d_embeded4 = d_array[i * 4 + 1]
                d_embeded5 = int('0b10000',2)
            if possible_index == int('0b10001',2):
                d_embeded1 = d_array[i * 4 + 2]
                d_embeded2 = d_array[i * 4 + 3]
                d_embeded3 = d_array[i * 4 + 1]
                d_embeded4 = d_array[i * 4 + 0]
                d_embeded5 = int('0b10001',2)
            if possible_index == int('0b10010',2):
                d_embeded1 = d_array[i * 4 + 3]
                d_embeded2 = d_array[i * 4 + 0]
                d_embeded3 = d_array[i * 4 + 1]
                d_embeded4 = d_array[i * 4 + 2]
                d_embeded5 = int('0b10010',2)
            if possible_index == int('0b10011',2):
                d_embeded1 = d_array[i * 4 + 3]
                d_embeded2 = d_array[i * 4 + 0]
                d_embeded3 = d_array[i * 4 + 2]
                d_embeded4 = d_array[i * 4 + 1]
                d_embeded5 = int('0b10011',2)
            if possible_index == int('0b10100',2):
                d_embeded1 = d_array[i * 4 + 3]
                d_embeded2 = d_array[i * 4 + 1]
                d_embeded3 = d_array[i * 4 + 0]
                d_embeded4 = d_array[i * 4 + 2]
                d_embeded5 = int('0b10100',2)
            if possible_index == int('0b10101',2):
                d_embeded1 = d_array[i * 4 + 3]
                d_embeded2 = d_array[i * 4 + 1]
                d_embeded3 = d_array[i * 4 + 2]
                d_embeded4 = d_array[i * 4 + 0]
                d_embeded5 = int('0b10101',2)
            if possible_index == int('0b10110',2):
                d_embeded1 = d_array[i * 4 + 3]
                d_embeded2 = d_array[i * 4 + 2]
                d_embeded3 = d_array[i * 4 + 0]
                d_embeded4 = d_array[i * 4 + 1]
                d_embeded5 = int('0b10110',2)
            if possible_index == int('0b10111',2):
                d_embeded1 = d_array[i * 4 + 3]
                d_embeded2 = d_array[i * 4 + 2]
                d_embeded3 = d_array[i * 4 + 1]
                d_embeded4 = d_array[i * 4 + 0]
                d_embeded5 = int('0b10111',2)
            
            # 复制像素值
            for j in range(0,5):
                G5p[possible_index,j] = pixels_group[i,j]
            # 替换
            d_embeded = np.zeros(5)
            d_embeded[0] = d_embeded1
            d_embeded[1] = d_embeded2
            d_embeded[2] = d_embeded3
            d_embeded[3] = d_embeded4
            d_embeded[4] = d_embeded5
            for j in range(0,4):
                #将最低k位置零
                assert(k < 9)
                #a1 = int('0b11111100', 2) #这条代码只适用于k=2
                a1 = 255 - (pow(2,k) - 1)
                a2 = bin(int(G5p[possible_index,j])) #转换为2进制字符串
                a3 = int(a2,2)
                a4 = a1 & a3 #低k位置零
                G5p[possible_index,j] = a4 | int(bin(int(d_embeded[j])),2)
            #最后一个单独处理
            G5p[possible_index,4] = (int(bin(int(G5p[possible_index,4])),2) & int('0b11100000', 2)) | int(bin(int(d_embeded[4])),2)
            #计算PSNR5
            tmpPSNR5[possible_index] = PSNR5(pixels_group[i],G5p[possible_index])
            if tmpPSNR5[possible_index] > MaxPSNR5:
                MaxPSNR5 = tmpPSNR5[possible_index]
                sected_index = possible_index
        #修改
        embeded_pixels_group[i] = G5p[sected_index] 

    #-----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros(num_secret_groups)
    tmpG5p = np.zeros(5)
    for i in range(0,int(num_secret_groups / 4),1):
        tmpG5p = embeded_pixels_group[i]
        tmpG1p = int(tmpG5p[4]) #取出G1p
        tmpG1p = int(bin(tmpG1p),2) & int('0b00011111',2) #取出低五位
        if tmpG1p == int('0b00000',2):
            r_d_embeded0 = tmpG5p[0]
            r_d_embeded1 = tmpG5p[1]
            r_d_embeded2 = tmpG5p[2]
            r_d_embeded3 = tmpG5p[3]
        if tmpG1p == int('0b00001',2):
            r_d_embeded0 = tmpG5p[0]
            r_d_embeded1 = tmpG5p[1]
            r_d_embeded3 = tmpG5p[2]
            r_d_embeded2 = tmpG5p[3]
        if tmpG1p == int('0b00010',2):
            r_d_embeded0 = tmpG5p[0]
            r_d_embeded2 = tmpG5p[1]
            r_d_embeded1 = tmpG5p[2]
            r_d_embeded3 = tmpG5p[3]
        if tmpG1p == int('0b00011',2):
            r_d_embeded0 = tmpG5p[0]
            r_d_embeded2 = tmpG5p[1]
            r_d_embeded3 = tmpG5p[2]
            r_d_embeded1 = tmpG5p[3]
        if tmpG1p == int('0b00100',2):
            r_d_embeded0 = tmpG5p[0]
            r_d_embeded3 = tmpG5p[1]
            r_d_embeded1 = tmpG5p[2]
            r_d_embeded2 = tmpG5p[3]
        if tmpG1p == int('0b00101',2):
            r_d_embeded0 = tmpG5p[0]
            r_d_embeded3 = tmpG5p[1]
            r_d_embeded2 = tmpG5p[2]
            r_d_embeded1 = tmpG5p[3]
        if tmpG1p == int('0b00110',2):
            r_d_embeded1 = tmpG5p[0]
            r_d_embeded0 = tmpG5p[1]
            r_d_embeded2 = tmpG5p[2]
            r_d_embeded3 = tmpG5p[3]
        if tmpG1p == int('0b00111',2):
            r_d_embeded1 = tmpG5p[0]
            r_d_embeded0 = tmpG5p[1]
            r_d_embeded3 = tmpG5p[2]
            r_d_embeded2 = tmpG5p[3]
        if tmpG1p == int('0b01000',2):
            r_d_embeded1 = tmpG5p[0]
            r_d_embeded2 = tmpG5p[1]
            r_d_embeded0 = tmpG5p[2]
            r_d_embeded3 = tmpG5p[3]
        if tmpG1p == int('0b01001',2):
            r_d_embeded1 = tmpG5p[0]
            r_d_embeded2 = tmpG5p[1]
            r_d_embeded3 = tmpG5p[2]
            r_d_embeded0 = tmpG5p[3]
        if tmpG1p == int('0b01010',2):
            r_d_embeded1 = tmpG5p[0]
            r_d_embeded3 = tmpG5p[1]
            r_d_embeded0 = tmpG5p[2]
            r_d_embeded2 = tmpG5p[3]
        if tmpG1p == int('0b01011',2):
            r_d_embeded1 = tmpG5p[0]
            r_d_embeded3 = tmpG5p[1]
            r_d_embeded2 = tmpG5p[2]
            r_d_embeded0 = tmpG5p[3]
        if tmpG1p == int('0b01100',2):
            r_d_embeded2 = tmpG5p[0]
            r_d_embeded0 = tmpG5p[1]
            r_d_embeded1 = tmpG5p[2]
            r_d_embeded3 = tmpG5p[3]
        if tmpG1p == int('0b01101',2):
            r_d_embeded2 = tmpG5p[0]
            r_d_embeded0 = tmpG5p[1]
            r_d_embeded3 = tmpG5p[2]
            r_d_embeded1 = tmpG5p[3]
        if tmpG1p == int('0b01110',2):
            r_d_embeded2 = tmpG5p[0]
            r_d_embeded1 = tmpG5p[1]
            r_d_embeded0 = tmpG5p[2]
            r_d_embeded3 = tmpG5p[3]
        if tmpG1p == int('0b01111',2):
            r_d_embeded2 = tmpG5p[0]
            r_d_embeded1 = tmpG5p[1]
            r_d_embeded3 = tmpG5p[2]
            r_d_embeded0 = tmpG5p[3]
        if tmpG1p == int('0b10000',2):
            r_d_embeded2 = tmpG5p[0]
            r_d_embeded3 = tmpG5p[1]
            r_d_embeded0 = tmpG5p[2]
            r_d_embeded1 = tmpG5p[3]
        if tmpG1p == int('0b10001',2):
            r_d_embeded2 = tmpG5p[0]
            r_d_embeded3 = tmpG5p[1]
            r_d_embeded1 = tmpG5p[2]
            r_d_embeded0 = tmpG5p[3]
        if tmpG1p == int('0b10010',2):
            r_d_embeded3 = tmpG5p[0]
            r_d_embeded0 = tmpG5p[1]
            r_d_embeded1 = tmpG5p[2]
            r_d_embeded2 = tmpG5p[3]
        if tmpG1p == int('0b10011',2):
            r_d_embeded3 = tmpG5p[0]
            r_d_embeded0 = tmpG5p[1]
            r_d_embeded2 = tmpG5p[2]
            r_d_embeded1 = tmpG5p[3]
        if tmpG1p == int('0b10100',2):
            r_d_embeded3 = tmpG5p[0]
            r_d_embeded1 = tmpG5p[1]
            r_d_embeded0 = tmpG5p[2]
            r_d_embeded2 = tmpG5p[3]
        if tmpG1p == int('0b10101',2):
            r_d_embeded3 = tmpG5p[0]
            r_d_embeded1 = tmpG5p[1]
            r_d_embeded2 = tmpG5p[2]
            r_d_embeded0 = tmpG5p[3]
        if tmpG1p == int('0b10110',2):
            r_d_embeded3 = tmpG5p[0]
            r_d_embeded2 = tmpG5p[1]
            r_d_embeded0 = tmpG5p[2]
            r_d_embeded1 = tmpG5p[3]
        if tmpG1p == int('0b10111',2):
            r_d_embeded3 = tmpG5p[0]
            r_d_embeded2 = tmpG5p[1]
            r_d_embeded1 = tmpG5p[2]
            r_d_embeded0 = tmpG5p[3]
        #取出最低2个bit
        recover1 = (pow(2,k) - 1) & int(bin(int(r_d_embeded0)),2)
        recover_d_array[i * 4 + 0] = recover1 
        recover1 = (pow(2,k) - 1) & int(bin(int(r_d_embeded1)),2)
        recover_d_array[i * 4 + 1] = recover1 
        recover1 = (pow(2,k) - 1) & int(bin(int(r_d_embeded2)),2)
        recover_d_array[i * 4 + 2] = recover1 
        recover1 = (pow(2,k) - 1) & int(bin(int(r_d_embeded3)),2)
        recover_d_array[i * 4 + 3] = recover1 
        assert(int((recover_d_array[i * 4 + 0:i * 4 + 3] - d_array[i * 4 + 0:i * 4 + 3]).sum()) == 0)

    # 恢复出的和以前的应该是一致的
    assert(int((recover_d_array - d_array).sum()) == 0)
    #-----------------------------------------------------------------------------------
    #输出图像
    img_out = embeded_pixels_group.flatten()
    img_out = img_out[:512 * 512] #取前面的pixel
    #计算PSNR
    img_array_out = img_out.copy()
    psnr = PSNR(image_array,img_array_out)
    print('MaxPSNR5 PSNR: %.2f' % psnr)
    csnr = CSNR(image_array,img_array_out)
    print('MaxPSNR5 CSNR: %.2f' % csnr)
    #重组图像
    img_out = img_out.reshape(512,512)
    img_out = Image.fromarray(img_out)
    #img_out.show()
    #-----------------------------------------------------------------------------------
    return

#ALGORITHM: VariableK our new 方法
def VariableK(image_array,secret_string,n=4):
    #image_array:输入的一维图像数组
    #n为一组像素的数量,在本算法中，固定为4
    def find_k(p_group):
        #p_group:三个像素
        assert(len(p_group) == 3)
        k = 2
        p = np.zeros(3)

        for i in range(0,3):
            p[i] = int('0b11000000',2) & int(p_group[i]) #取最高1位
        tmp_sum = 0
        for i in range(0,3):
            tmp_sum +=p[i] 

        if tmp_sum >= 192 * 3: #最高2位都为1
            k = 6
        else:
            if tmp_sum >= 2 * 192 + 64: #有一个最高位不是1
                k = 5
            else:
                if tmp_sum >= 192 + 2 * 64:
                    k = 4
                else:
                    if tmp_sum >= 3 * 64:
                        k = 3
                    else:
                        k = 2

        return k

    def find_k_OLD(p_group):
        #p_group:三个像素
        assert(len(p_group) == 3)
        tmp_sum = 0
        for i in range(0,3):
            tmp_sum +=p_group[i] 
        if tmp_sum > 3 * 192:
            k = 5
        else:
            if tmp_sum > 3 * 160:
                k = 4
            else:
                if tmp_sum > 3 * 96:
                    k = 3
                else:
                    k = 2
        return k

    n = 4
    #分成n个像素一组,保证整数组，不足的补零
    num_pixel_groups = int(image_array.size / n)
    while num_pixel_groups % 4 > 0:
        num_pixel_groups+=1
    num_pixel_groups = math.ceil(image_array.size / 4)
    pixels_group = np.zeros((num_pixel_groups,4))

    k_array = np.zeros(num_pixel_groups) #k值


    for i in range(0,num_pixel_groups):
        for j in range(0,4):
            tmp_sum = 0
            if(i * 4 + j < image_array.size):
                 pixels_group[i,j] = image_array[i * 4 + j]
            #确定k的值
        k_array[i] = find_k(pixels_group[i,0:3])
                        

    #-----------------------------------------------------------------------------------
    #metrics
    def CSNR4(image_array1,image_array2):
        #输入为两个图像数组，一维，大小相同
        assert(np.size(image_array1) == np.size(image_array2))
        n = np.size(image_array1)
        assert(n > 0)
        MSE = 0
        P2 = 0 
        for i in range(0,n):
            MSE+=math.pow(image_array1[i] - image_array2[i],2)
            P2+=math.pow(image_array1[i],2)
        if MSE > 0:
            rtnCSNR = 10 * math.log10(P2 / MSE)
        else:
            rtnCSNR = 100
        return rtnCSNR
    #-----------------------------------------------------------------------------------

    #开始进行嵌入
    secret_string_copy = secret_string.copy()
    embeded_pixels_group = pixels_group.copy()
    sec_index = 0
    pixel_group_index = 0
    
    while (sec_index < len(secret_string_copy)) & (pixel_group_index < num_pixel_groups):
        sected_index = 0 #选中的方案索引
        order_number = 8 #有八种次序
        G4p = np.zeros((order_number,4))
        MaxCSNR4 = 0
        tmpCSNR4 = np.zeros(order_number)
         
        #取出secret bits
        k = int(k_array[pixel_group_index])
        bits_count = int(3 * k)
        secret_group = np.zeros(bits_count)
        for j in range(0,bits_count):
            if sec_index + j < len(secret_string_copy):
                secret_group[j] = secret_string_copy[sec_index + j]
        #将其分为三组
        #第一组的10进制值
        d_group0 = 0 #第0组
        for j in range(0,k):
            d_group0 += 2 ** (j) * secret_group[j]
        d_group1 = 0#第1组
        for j in range(0,k):
            d_group1 += 2 ** (j) * secret_group[k + j]
        d_group2 = 0#第2组
        for j in range(0,k):
            d_group2 += 2 ** (j) * secret_group[k + k + j]

        for possible_index in range(0,order_number): #计算所有可能的组合，看哪一个最合适
                if possible_index == 0:
                    d_embeded0 = d_group0
                    d_embeded1 = d_group1
                    d_embeded2 = d_group2
                    d_embeded3 = 0
                if possible_index == 1:
                    d_embeded0 = d_group0
                    d_embeded1 = d_group1
                    d_embeded2 = d_group2
                    d_embeded3 = 1
                if possible_index == 2:
                    d_embeded0 = d_group0
                    d_embeded1 = d_group2
                    d_embeded2 = d_group1
                    d_embeded3 = 2
                if possible_index == 3:
                    d_embeded0 = d_group1
                    d_embeded1 = d_group0
                    d_embeded2 = d_group2
                    d_embeded3 = 3                
                if possible_index == 4:
                    d_embeded0 = d_group1
                    d_embeded1 = d_group2
                    d_embeded2 = d_group0
                    d_embeded3 = 4
                if possible_index == 5:
                    d_embeded0 = d_group2
                    d_embeded1 = d_group0
                    d_embeded2 = d_group1
                    d_embeded3 = 5
                if possible_index == 6:
                    d_embeded0 = d_group2
                    d_embeded1 = d_group1
                    d_embeded2 = d_group0
                    d_embeded3 = 6
                if possible_index == 7:
                    d_embeded0 = d_group2
                    d_embeded1 = d_group1
                    d_embeded2 = d_group0
                    d_embeded3 = 7                   
                # 复制像素值
                for j in range(0,4):
                    G4p[possible_index,j] = pixels_group[pixel_group_index,j]
                # 替换
                d_embeded = np.zeros(4)
                d_embeded[0] = d_embeded0
                d_embeded[1] = d_embeded1
                d_embeded[2] = d_embeded2
                d_embeded[3] = d_embeded3
      
                for j in range(0,3):
                    a1 = 255 - (pow(2,k) - 1)
                    a2 = bin(int(G4p[possible_index,j])) #转换为2进制字符串
                    a3 = int(a2,2)
                    a4 = a1 & a3 #低k位置零
                    G4p[possible_index,j] = a4 | int(bin(int(d_embeded[j])),2)
                #最后一个单独处理
                G4p[possible_index,3] = (int(bin(int(G4p[possible_index,3])),2) & int('0b11111000', 2)) | int(bin(int(d_embeded[3])),2)
                #计算PSNR5
                tmpCSNR4[possible_index] = CSNR4(pixels_group[i],G4p[possible_index])
                if tmpCSNR4[possible_index] > MaxCSNR4:
                    MaxCSNR4 = tmpCSNR4[possible_index]
                    sected_index = possible_index
        embeded_pixels_group[pixel_group_index] = G4p[sected_index]
        #进入下一次循环
        sec_index+=bits_count #取走了bits_count个bit
        pixel_group_index+=1
    #使用了多少pixel来进行嵌入
    pixels_changed = (pixel_group_index - 1) * 3
    #-----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_sec_index = 0
    pixel_group_index = 0
    recover_string = np.zeros(len(secret_string))
    tmpG4p = np.zeros(4)
    while (recover_sec_index < len(recover_string)) & (pixel_group_index < num_pixel_groups):
        tmpG4p = embeded_pixels_group[pixel_group_index]
        tmpG1p = int(tmpG4p[3]) #取出G1p
       

        tmpG1p = int(bin(tmpG1p),2) & int('0b00000111',2) #取出低3位
        if (tmpG1p == 0) | (tmpG1p == 1) :
            r_d_embeded0 = tmpG4p[0]
            r_d_embeded1 = tmpG4p[1]
            r_d_embeded2 = tmpG4p[2]
        if (tmpG1p == 2) :
            r_d_embeded0 = tmpG4p[0]
            r_d_embeded2 = tmpG4p[1]
            r_d_embeded1 = tmpG4p[2]
        if (tmpG1p == 3) :
            r_d_embeded1 = tmpG4p[0]
            r_d_embeded0 = tmpG4p[1]
            r_d_embeded2 = tmpG4p[2]
        if (tmpG1p == 4) :
            r_d_embeded1 = tmpG4p[0]
            r_d_embeded2 = tmpG4p[1]
            r_d_embeded0 = tmpG4p[2]
        if (tmpG1p == 5) :
            r_d_embeded2 = tmpG4p[0]
            r_d_embeded0 = tmpG4p[1]
            r_d_embeded1 = tmpG4p[2]
        if (tmpG1p == 6) | (tmpG1p == 7) :
            r_d_embeded2 = tmpG4p[0]
            r_d_embeded1 = tmpG4p[1]
            r_d_embeded0 = tmpG4p[2]
        
        k = int(find_k(tmpG4p[0:3]))
        #取出最低k个bit
        recover0 = (pow(2,k) - 1) & int(bin(int(r_d_embeded0)),2)
        for j_tmp in range(0,k):
            arr_tmp = dec2bin_lower_ahead(recover0,k)
            if recover_sec_index + j_tmp < len(recover_string):
                recover_string[recover_sec_index + j_tmp] = arr_tmp[j_tmp] 
        recover1 = (pow(2,k) - 1) & int(bin(int(r_d_embeded1)),2)
        for j_tmp in range(0,k):
            arr_tmp = dec2bin_lower_ahead(recover1,k)
            if recover_sec_index + k + j_tmp < len(recover_string):
                recover_string[recover_sec_index + k + j_tmp] = arr_tmp[j_tmp] 
        recover2 = (pow(2,k) - 1) & int(bin(int(r_d_embeded2)),2)
        for j_tmp in range(0,k):
            arr_tmp = dec2bin_lower_ahead(recover2,k)
            if recover_sec_index + k + k + j_tmp < len(recover_string):
                recover_string[recover_sec_index + k + k + j_tmp] = arr_tmp[j_tmp] 

        assert(int((recover_string[recover_sec_index:recover_sec_index + 3 * k] - secret_string[recover_sec_index:recover_sec_index + 3 * k]).sum()) == 0)
        #下一次循环
        recover_sec_index+=3 * k
        pixel_group_index+=1

    # 恢复出的和以前的应该是一致的
    diff = recover_string - secret_string
    #print(diff)
    assert(int((recover_string - secret_string).sum()) == 0)
    #-----------------------------------------------------------------------------------
    #输出图像
    img_out = embeded_pixels_group.flatten()
    img_out = img_out[:512 * 512] #取前面的pixel
    #计算PSNR
    img_array_out = img_out.copy()
    psnr = PSNR(image_array,img_array_out)
    print('VariableK PSNR: %.2f' % psnr)
    #csnr = CSNR(image_array,img_array_out)
    #print('VariableK CSNR: %.2f' % csnr)
    print('VariableK pixels used: %d' % pixels_changed)
    
    #重组图像
    img_out = img_out.reshape(512,512)
    img_out = Image.fromarray(img_out)
    #img_out.show()
    #-----------------------------------------------------------------------------------
    return

#ALGORITHM: PrimeNumberModulo our new 方法
def PrimeNumberModulo2(image_array,secret_string,n=2):
    #image_array:输入的一维图像数组
    #n为一组像素的数量,在本算法中，固定为2
    n = 2
    k = 3 #每组pixel嵌入nk+1个bit
    moshu = 2 ** (n * k + 1) #模数的底数
    c0 = 3
    c1 = 11
    #分成n个像素一组,保证整数组，不足的补零

    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups,n))
    
    for i in range(0,num_pixel_groups,1):
        for j in range(0,n,1):
            if i * n + j < image_array.size:
               pixels_group[i,j] = image_array[i * n + j]

    fG_array = np.zeros((num_pixel_groups))
    for i in range(0,num_pixel_groups,1):
        fG_array[i] = (c0 * pixels_group[i,0] + c1 * pixels_group[i,1]) % moshu

    num_BitsPerPixelsGoup = n * k + 1  #每组pixcel嵌入的bit数
    num_secret_groups = math.ceil(secret_string.size / num_BitsPerPixelsGoup)
    secret_group = np.zeros((num_secret_groups,num_BitsPerPixelsGoup))
    secret_string_copy = secret_string.copy()
    for i in range(0,num_secret_groups,1):
        for j in range(0,num_BitsPerPixelsGoup,1):
            if i * num_BitsPerPixelsGoup + j < secret_string.size:
               secret_group[i,j] = secret_string_copy[i * num_BitsPerPixelsGoup + j]

    secret_d_array = np.zeros(num_secret_groups) #待嵌入的secret值
    for i in range(0,num_secret_groups,1):
        for j in range(0,num_BitsPerPixelsGoup,1):
            secret_d_array[i]+=(2 ** j) * secret_group[i,j]

    #-----------------------------------------------------------------------------------
    #metrics
    def CSNR(image_array1,image_array2):
        #输入为两个图像数组，一维，大小相同
        assert(np.size(image_array1) == np.size(image_array2))
        n = np.size(image_array1)
        assert(n > 0)
        MSE = 0
        P2 = 0 
        for i in range(0,n):
            MSE+=math.pow(image_array1[i] - image_array2[i],2)
            P2+=math.pow(image_array1[i],2)
        if MSE > 0:
            rtnCSNR = 10 * math.log10(P2 / MSE)
        else:
            rtnCSNR = 100
        return rtnCSNR
    #-----------------------------------------------------------------------------------

    assert(num_pixel_groups > num_secret_groups)
    embeded_pixels_group = pixels_group.copy()
    for i in range(0,num_secret_groups,1):
        tmp_MaxPsnr = 0
        tmp_SlectedIndex0 = 0
        tmp_SlectedIndex1 = 0
        tmp_P = np.zeros(2)
        for j0 in range(-1 * moshu,moshu,1):
            for j1 in range(-1 * moshu,moshu,1):
                tmp_P[0] = (pixels_group[i,0] + j0)
                tmp_P[1] = (pixels_group[i,1] + j1)
                tmp = (c0 * tmp_P[0] + c1 * tmp_P[1]) % moshu
                if  (int(secret_d_array[i]) == int(tmp)):
                    tmp1 = CSNR(pixels_group[i],tmp_P)
                    if tmp1 > tmp_MaxPsnr:
                        tmp_MaxPsnr = tmp1
                        tmp_SlectedIndex0 = j0
                        tmp_SlectedIndex1 = j1
        assert(tmp_MaxPsnr > 0)
        embeded_pixels_group[i,0] = pixels_group[i,0] + tmp_SlectedIndex0
        embeded_pixels_group[i,1] = pixels_group[i,1] + tmp_SlectedIndex1


    #使用了多少pixel来进行嵌入
    pixels_changed = num_secret_groups * 2
    #-----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros(num_secret_groups) #待嵌入的secret值
    for i in range(0,num_secret_groups,1):
        for j in range(0,num_BitsPerPixelsGoup,1):
            tmp = (c0 * (embeded_pixels_group[i,0]) + c1 * (embeded_pixels_group[i,1])) % moshu
            recover_d_array[i] = tmp
    assert(int((recover_d_array - secret_d_array).sum()) == 0)
    #-----------------------------------------------------------------------------------
    #输出图像
    img_out = embeded_pixels_group.flatten()
    img_out = img_out[:512 * 512] #取前面的pixel
    #计算PSNR
    img_array_out = img_out.copy()
    psnr = PSNR(image_array,img_array_out)
    print('PrimeNumberModulo2 PSNR: %.2f' % psnr)
    #csnr = CSNR(image_array,img_array_out)
    #print('PrimeNumberModulo2 CSNR: %.2f' % csnr)
    print('PrimeNumberModulo2 pixels used: %d' % pixels_changed)
    
    #重组图像
    img_out = img_out.reshape(512,512)
    img_out = Image.fromarray(img_out)
    #img_out.show()
    #-----------------------------------------------------------------------------------
    return


def ModuleCoputing():
    #计算一个模运算，n个pixel，每个嵌入k个bit，
    n = 4
    k = 3
    moshu = 2 ** (n * k + 1)
    #随机生成n个pixel
    data_array = np.random.randint(0,255,n)
    #把n个pixel的低k个bit取出来
    for i in range(0,n):
        lower_k_bits = np.zeros(n)
        lower_k_bits[i] = (2 ** k - 1) & data_array[i] #

    r_list = []
    for i1 in range(0,255):
        for i2 in range(0,255):
            for i3 in range(0,255):
                a1 = i1 * i2 * i3 % moshu
                tmp = (c0 * tmp_P[0] + c1 * tmp_P[1]) % moshu
                if  (int(secret_d_array[i]) == int(tmp)):
                    tmp1 = CSNR(pixels_group[i],tmp_P)
                if a1 not in r_list:
                    r_list.append(a1)
    print(len(r_list))
    assert(len(r_list) == moshu)
    return


#调用函数
#EMD_2006(img_array3,s_data,2)
# LWC_2007(img_array3,s_data,n=2)
#JY_2009(img_array3,s_data,n=2)
# GEMD_2013(img_array3,s_data,n=4)
KKWW_2016(img_array3,s_data,n=2)
#SB_2019(img_array3,s_data,n=1)
SB_2019_1(img_array3,s_data,n=1)
#MaxPSNR5(img_array3,s_data,n=5)
#VariableK(img_array3,s_data,n = 4)
# PrimeNumberModulo2(img_array3,s_data,n = 4)
#ModuleCoputing()
time.sleep(10)
#img_out.save(fname + '.png','png')