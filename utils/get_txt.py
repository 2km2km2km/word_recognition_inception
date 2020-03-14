import os
import tqdm

def get_txt_CASIA(path,max_class_num=10,train_jpg_num=300):

    #get the train_txt and the test_txt
    #书写CASIA的train_dataset.txt和test_dataset.txt
    #path为CASIA数据集路径
    #max_class_num=最多种类的数量
    #the max number of the classes
    class_max_num=10

    test_path = path+"1.0test-gb1/"
    train_path = path+"1.0train-gb1/"

    train_txt = open('./train.txt',mode='w')
    test_txt = open('./test.txt', mode='w')

    #读取数据库中已有数据
    train_img_paths = os.walk(train_path)
    test_img_paths = os.walk(test_path)
    index=0
    train_img_paths=list(train_img_paths)
    test_img_paths = list(test_img_paths)

    # 训练集数据
    for l in tqdm.tqdm(train_img_paths[1:]):
        index+=1
        label = l[0].replace(train_path, "")
        for jpg_num,img_path in enumerate(l[2]):
            if jpg_num>=train_jpg_num:
                break
            train_txt.write(l[0]+"/"+img_path+label)
            train_txt.write("\n")

        # 测试集数据
        for j in test_img_paths[1:]:
            keyj=j[0].replace(test_path,"")
            if keyj==label:
                for img_path in j[2]:
                    test_txt.write(j[0] +"/"+ img_path+label)
                    test_txt.write("\n")
        if index>=max_class_num:
           break
    print("done")

#get_txt_CASIA("/media/xzl/Newsmy/数据集/CASIA-HWDB/Character Sample Data/1.0/",10,10)