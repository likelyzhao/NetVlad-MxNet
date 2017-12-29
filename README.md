# NetVlad-MxNet
NetVlad Training with MxNet

## steps:
1. prepare video scripts
2. cut frame and generate feature files
3. generate the label files 
4. training 

## prepare video
json list.
each line contains one json 
the format should be contain sevral key words like: 
```
{
  "url":string,
  "type":"video",
  "label":
  {
    "name"：“video_terror”，
    “type“: "video",
    "version":"1",
    "data":[
            {
              "label":string,
              "segment":[float,float]# starttime,endtime
            },
            ...
     ]
  }
}
```
### 字段详细解释：

|字段|类型|详细解释|
|---|---|---|
|url|string|视频的url路径（本地或者web）|
|type|string|类型 视频为`video`|
|name|string|标示，BK视频分类设置为：`video_terror`|
|version|string|版本信息，目前为`1`|
|label|string|BK的类别|
|segment|[float float]|开始和结束时间|

## cut frame and generate feature files @baobao

## generate the label files  @baobao

## training

### netvlad symbol

### dataiter

### training hyper param


