1.anotations
  (1)检测框标注文件，每个json文件包含一张图片的鸽眼矩形框标注。
  (2)包括9979张图片的标注信息。
  (3)每个json是一个字典格式文件：
      img:图片文件名称
      weidth:图片的宽
      height:图片的高
      bbs:是一个列表,列表每个元素是一个字典
          label:检测框的类别
          bbx:[x1,y1,x2,y2]
          

2.blood.csv
  (1)血统对应的图片集合,共计有28910个血统(有可能有重复,由于网站血统id的编码方式并不统一)
  (2)每行代表一个血统,每行第一项为血统的id,其他项为该血统相关的图片id,所有项以逗号为分割符
  
3.pigeon.csv:每张图片对应的详情页(url)

4. details.txt: 图片对应详情页的文本内容。

5.relations.csv:从details提取图片和足环号的关系, 记录每个血统id和图片的关系。

5.blood.csv:从relations.csv 中提取血统聚类.格式: col_1:血统id, col_2:图片id,...,col_n:图片id        


