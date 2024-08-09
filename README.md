# OptiNet-BCO
## 华为嵌入式大赛：光网络业务连续性优化难题
## OptiNet: Enhancing Optical Network Business Continuity
### 自定义结构体：
#### Edge：边结构体，保存信息包括该边的ID、边的两端节点、途经该边的业务ID、被占用的小通道、可用通道数、工作状态
#### Business：业务结构体，保存信息包括该业务ID、业务价值、业务状态、业务的开始节点和结束节点、经过的边数量、途经节点编号、途经边及占用的小通道
### 最短路线的搜寻：
#### step1 获取死亡路径上的业务，根据业务价值对业务进行排序
#### step2 通过算法Yen-s-k-shortest-paths-algorithm搜寻业务不经过死亡路径的前k条路径
#### step3 判断路径下的小通道是否满足通行需求
