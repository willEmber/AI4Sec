
|   |   |   |   |   |   |
|---|---|---|---|---|---|
|用户名|18782202187|密钥SecretKey|7fb456bc2ee9440a818d9c46b615dce1|已调用次数|131|
|快速生成接口|https://www.easyscholar.cc/open/getPublicationRank?secretKey=7fb456bc2ee9440a818d9c46b615dce1&publicationName=|测试接口|[调用接口](https://www.easyscholar.cc/open/getPublicationRank?secretKey=7fb456bc2ee9440a818d9c46b615dce1&publicationName=)|   |   |

# 期刊等级查询接口文档

## 产品概述

考虑到众多第三方应用有查询期刊等级的需求，于是开放此接口供用户免登录查询。此接口相比于扩展接口，速度更快，稳定性更强，逻辑更加清楚，免密操作更安全。

## 注意事项

1. 会员服务中所涉及的**期刊等级查询功能**并非此api接口。该接口不与会员服务绑定，目前免费对所有用户开放，祝各位科研工作者一切顺利！
2. 接口需要通过**密钥SecretKey**确认身份， 该SecretKey无法更改，请勿透露给任何人。
3. 该接口不涉及登录操作，不会挤占当前用户的登录。
4. 实际开发中请注意对期刊名进行encodeURIComponent()编码，防止&等符号影响传参
5. 实际开发中请对请求速度做限制，每秒最多2次请求。
6. 可以参考此文章对Zotero进行配置。[如何在Zotero上与easyScholar联动](https://www.easyscholar.cc/blogs/10007)

## API

接口地址：`https://www.easyscholar.cc/open/getPublicationRank` 请求方式：`get`

请求参数：

|参数名|类型|必填|说明|
|---|---|---|---|
|secretKey|string|是|根据此key确认身份|
|publicationName|string|是|期刊名称|

返回结果：

|参数名|类型|说明|
|---|---|---|
|code|int|成功为200，失败为其他|
|msg|string|成功为“SUCCESS”，失败为其他|
|data|嵌套对象|失败为null|

错误返回示例：

```
{
    "code": 40002,
    "msg": "Key错误",
    "data": null
}
```

正确返回结果：

```
{
    "code": 200,
    "msg": "SUCCESS",
    "data": {
        "customRank": {
            "rankInfo": [
                 {
                    "uuid": "1613157898765070336",
                    "abbName": "国自然管科",
                    "oneRankText": "A",
                    "twoRankText": "B"
                },
                {
                    "uuid": "1614986460329492480",
                    "abbName": "DUFE",
                    "oneRankText": "TOP",
                    "twoRankText": "A",
                    "threeRankText": "B",
                    "fourRankText": "C",
                    "fiveRankText": "D"
                },
            ],
            "rank": [
                "1614986460329492480&&&3",
            ]
        },
        "officialRank": {
            "all": {
                "swufe": "A",
                "cufe": "AA",
            },
            "select": {
                "cufe": "AA",
            }
        }
    }
}
```

正确结果解释：

`data`中包含两个对象，分别是`customRank`, `officialRank`，分别对应**自定义数据集**， **官方数据集**

- `officialRank`：
    - `all`, `select`类型相同，分别代表**所有数据等级**， **用户在扩展端选中的数据集**
    - `all`, `select`一共有37个字段（无结果的字段不返回），例如`swufe`, `sci`, `ssci`等，具体字段解释见**附录1**。
- `customRank`：
    - `rankInfo`代表用户添加的数据集信息，可能包含多个，具体字段解释见**附录2**。
    - `rank`代表该期刊在自定义数据集中的等级，结构构成为`{{rankInfo.uuid}}&&&{{rank}}`。首先通过`split()`函数获得`uuid`与`rank`，前者为数据集的`uuid`，通过此`uuid`前往`rankInfo`中寻找`uuid`相同的自定义数据集，获得其缩写；后者为等级，范围[1-5]，通过此范围前往自定义数据集中，获得对应的等级缩写。
    - 例如**结果示例**中`1614986460329492480&&&3`，`uuid`为`1614986460329492480`。通过此`uuid`获得`abbName`为**DUFE**（东北财经大学），再通过`3`获得`threeRankText`缩写为`B`。于是展示出结果：**DUFE B**

## 附录

### 附录1：

仅展示缩写对应的单位，具体数据集与等级划分，请前往官网查看

|缩写|解释|缩写|解释|缩写|解释|
|---|---|---|---|---|---|
|swufe|西南财经大学|cqu|重庆大学|sciif|SCI影响因子-JCR|
|cufe|中央财经大学|nju|南京大学|sci|SCI分区-JCR|
|uibe|对外经济贸易大学|xju|新疆大学|ssci|SSCI分区-JCR|
|sdufe|山东财经大学|cug|中国地质大学|jci|JCI指数-JCR|
|xdu|西安电子科技大学|ccf|中国计算机学会|sciif5|SCI五年影响因子-JCR|
|swjtu|西南交通大学|cju|长江大学（不是计量大学）|sciwarn|中科院预警|
|ruc|中国人民大学|zju|浙江大学|sciBase|SCI基础版分区-中科院|
|xmu|厦门大学|zhongguokejihexin|中国科技核心期刊|sciUp|SCI升级版分区-中科院|
|sjtu|上海交通大学|fms|FMS|ajg|ABS学术期刊指南|
|fdu|复旦大学|utd24|UTD24|ft50|FT50|
|hhu|河海大学|eii|EI检索|cscd|中国科学引文数据库|
|pku|北大核心|cssci|南大核心|ahci|A&HCI|
|scu|四川大学|sciUpSmall|中科院升级版小类分区|esi|ESI学科分类|
|sciUpTop|中科院升级版Top分区|cpu|中国药科大学|||

注意：**sciUpSmall、esi、sciUpTop、cpu**为2024年3月19日新增字段，在此之前创作的第三方软件，可能并不能识别这几条字段

### 附录2：

|字段名|解释|
|---|---|
|uuid|自定义数据集唯一ID|
|abbName|数据集缩写|
|oneRankText|等级一缩写|
|twoRankText|等级二缩写|
|threeRankText|等级三缩写|
|fourRankText|等级四缩写|
|fiveRankText|等级五缩写|
