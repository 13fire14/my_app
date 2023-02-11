# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 18:46:23 2023

@author: bianca
"""

#%% 功能实现的代码
def merge_pdfs(old_pdf_dir,new_pdf_dir,new_pdf_name):
    ###文件夹内合并
    #加载对应的库
    import pandas as pd
    import pdfplumber
    import PyPDF2
    import os
    from tqdm import tqdm
    import streamlit as st
    import time
    #获取待合并文件夹下面的所有pdf
    finall_time=time.time()
    pdf_name=os.listdir(old_pdf_dir)
    pdf_names=[]
    for i in pdf_name:
        if '.pdf' in i:
            pdf_names.append(os.path.join(old_pdf_dir,i))
    #总页数初始化为零
    new_pdf=PyPDF2.PdfWriter()
    pdf_numpages=0
    #待合并的读取进来
    for j in tqdm(pdf_names):
        old_pdf=PyPDF2.PdfReader(open(j,'rb'))
        pdf_pages=len(old_pdf.pages)
        #更新总页数
        pdf_numpages+=pdf_pages
        #每页读取进来
        for k in range(pdf_pages):
            new_pdf.add_page(old_pdf.pages[k])
    with open(os.path.join(new_pdf_dir,new_pdf_name),'wb') as p:
        new_pdf.write(p)
    use_time=time.time()-finall_time
    st.write('合并已经完成,耗时%d秒'%use_time)
    st.balloons()
#%% 页面呈现
#python小功能
import pandas as pd
import streamlit as st
import os
import re
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from streamlit.elements.image import image_to_url
from PIL import Image
import matplotlib.pyplot as plt


st.title('功能区')

# # 背景图片的网址
# img_url = image_to_url(plt.imread('D:/16照片/毕业照/DSC_0361.jpg'),width=-3,clamp
 
# # 修改背景样式
# st.markdown('''<style>.css-fg4pbf{background-image:url(''' + img_url + ''');
# background-size:100% 100%;background-attachment:fixed;}</style>
# ''', unsafe_allow_html=True) 


#%% 添加背景图
img_url=image_to_url(plt.imread('D:/pythonFile/image/烟雨江南.jpg'),width=-3,clamp=False,channels='RGB',output_format='auto',
                    image_id='')
st.markdown('''
<style>
.css-fg4pbf {background-image:url('''+img_url+''');}</style>
''',unsafe_allow_html=True)
choose=st.sidebar.selectbox('功能选择区', ['pdf合并','鸢尾花预测','其他小功能请等待开发'])
#%%pdf框
if choose=='pdf合并':
    old_pdf_dir=st.text_input('请输入即将要合并的文件夹：')
    new_pdf_dir=st.text_input('请输入合并完成后文件存放路径：')
    new_pdf_name=st.text_input('请输入合并后的文件名称：')
#判断新文件名是否已经有了
    name=os.listdir(new_pdf_dir)

    #判断
    if new_pdf_name not in name:
        if st.button('请点击开始合并' ):
            merge_pdfs(old_pdf_dir,new_pdf_dir,new_pdf_name) 
    else:
        st.write('已经存在该文件名，请换一个,如果不更新文件名，将会覆盖原文件')
        #获取新存放文件夹下的文件名单：提醒作用
        index_input=[]
        for i in range(len(name)):
            if name[i].startswith('合并')==True:
                index_input.append(name[i])
        expander=st.expander('查看类似的文件名')
        expander.write(index_input)
        if st.button('开始合并' ):
            merge_pdfs(old_pdf_dir,new_pdf_dir,new_pdf_name) 
      

#%% 鸢尾花数据集处理
iris=datasets.load_iris()
iris_x=iris.data
iris_y=iris.target
iris_x_train,iris_x_test,iris_y_train,iris_y_test=train_test_split(iris_x,iris_y,test_size=0.1,random_state=666)

def knn_predict():
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier()
    knn.fit(iris_x_train,iris_y_train)
    score=knn.score(iris_x_test, iris_y_test,sample_weight=None)
    return score,knn 

def tree_predict():
    from sklearn.tree import DecisionTreeClassifier
    clf=DecisionTreeClassifier()
    clf.fit(iris_x_train,iris_y_train)
    score=clf.score(iris_x_test, iris_y_test,sample_weight=None)
    return score,clf 


def inputmeter():
    sepal_length=st.sidebar.slider('sepal length',4.3,7.9,5.0)
    sepal_width=st.sidebar.slider('sepal width',2.0,4.4,3.0)
    petal_length=st.sidebar.slider('petal length',1.0,6.9,5.0)
    petal_width=st.sidebar.slider('petal width',0.1,2.5,1.0)
    data={
        "sepal_length":sepal_length,
        'sepal_width':sepal_width,
        'petal_length':petal_length,
        'petal_width':petal_width
        }
    return pd.DataFrame(data,index=[1])
if choose=='鸢尾花预测':
    st.sidebar.header('Input Parameter')
    data=inputmeter()
    st.write(data)  
    # choose1=st.selectbox('请选择模型',['knn','决策树'])
    col1,col2=st.columns(2)
    with col1:
        st.subheader('knn模型')
    # if choose1=='knn':
        score,knn=knn_predict()
        st.write('模型自测准确率为：',score)
        pred=knn.predict(data)
        st.write(pred)
        st.write('当前参数预测结果为：',iris.target_names[pred])
    with col2:
    # if choose1=='决策树':
        st.subheader('决策树模型')
        score,clf=tree_predict()
        st.write('模型自测准确率为：',score)
        pred=clf.predict(data)
        st.write(pred)
        st.write('当前参数预测结果为：',iris.target_names[pred])