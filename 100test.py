import tensorflow as tf
import os 
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

img_path='/Users/nanayana/Desktop/Black Sesame/question3/pics100'
text_file='/Users/nanayana/Desktop/Black Sesame/question3/labels100.txt'
file_path=os.listdir(img_path)
#将filepath排序
file_path.sort(key=lambda x:int(x[:-4]))

def creat_x_database(rootdir,resize_row,resize_col):
    #列出文件夹下所有的，目录和文件
    #创建一个随机矩阵，作为多个图片转换为矩阵后传入其中（arrange等差数列）
    database=np.arange(len(file_path)*resize_row*resize_col*3).reshape(len(file_path),resize_row,resize_col,3)
    for i in range(0,len(file_path)):
        path = os.path.join(rootdir,file_path[i])    #把目录和文件名合成一个路径
        if os.path.isfile(path):                ##判断路径是否为文件
            img_data = cv2.imread(path)#读取图片
            with tf.Session() as sess:
                #压缩图片矩阵为指定大小
                resized=tf.image.resize_images(img_data,[resize_row,resize_col],method=0)
                database[i]=resized.eval()
    return database       

def creat_y_database(pos,neg):
    with open(text_file,'r') as f:
        lab = [[0] * 2 for i in range(len(file_path))]
        #创建二维数组
        lines=f.readlines()
        for i in range(len(file_path)):
            no_n = lines[i].strip()
            no_t = no_n.split('\t')
            if no_t[1] == '1':
                lab[i] = pos
            else:
                lab[i] = neg
        lab_arr = np.array(lab)
    return lab_arr

#创建训练集 大小为 (96, 128, 128, 3) array not list
x_data = creat_x_database('/Users/nanayana/Desktop/Black Sesame/question3/pics100',128,128)
#创建标签集合 大小（96，2）array not list
y_data = creat_y_database([1,0],[0,1])
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.1,random_state=0)
#train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签。

#定义各参数变量并初始化            
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
 
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#计算准确率 
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result           

    
    
xs = tf.placeholder(tf.float32, [None, 128,128,3])/255 #归一化
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
#x_image = tf.reshape(xs, [-1, 50, 50, 3])
 
W_conv1 = weight_variable([5,5,3,32]) # patch 5x5, in size 3, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1) # output size 128x128x32
h_pool1 = max_pool_2x2(h_conv1)                          # output size 64x64x32
 
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 64x64x64
h_pool2 = max_pool_2x2(h_conv2) #32x32x64
 
W_fc1 = weight_variable([32*32*64, 1024])
b_fc1 = bias_variable([1024])
   
h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*64])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction+ 1e-10), reduction_indices=[1]))
#由于 prediction 可能为 0， 导致 log 出错，最后结果会出现 NA      
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)    
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

for i in range(10):
#    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.5})
    if i % 1 == 0:
        #print(compute_accuracy(x_test,y_test))
        print(sess.run(cross_entropy, feed_dict={xs: x_train, ys: y_train, keep_prob: 1}))
