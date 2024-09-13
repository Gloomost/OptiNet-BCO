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


### 题目难点：
#### 1. 题目中的几个关键概念：业务、大路径、大路径下的小通道、两条大路径间的节点
#### 2. 当某一路径死亡后，如果死亡路径上的业务没有得到重新规划（即没有为该业务找到新的工作路径），那么该业务就“死”在原路径上，占用的其他未死亡路径也不会释放。
#### 3. 每两点之间可能有多条大路径，每条大路径上有40个小通道，要对相同起始点的路径做区分（此处通过自定义结构体Edge解决，即每条路径都有自己特殊的编号）。
#### 4. 本题解决思路为先找到可能可行的大路径list，随后判断大路径下的小通道是否可以通行，如果可以通行，那么将新业务迁移到该路线，如果所有小通道均不满足通行条件， 则判断下一可行的大路径list是否满足条件。
#### 5. 不同大路径对应的小通道编号应当相同，即如果某一业务大路径list规划后为[1, 2, 3, 4, 5]且该业务宽度为5，那么该业务在这5条大路径上占用的小通道ID必须一样，即都是[1: 5]或[4: 9]等。但如果大路径1和大路径2中间的节点处有改变次数，那么可以消耗1次改变次数来使节点两边的通道ID不一样，即节点前的路径1所占通道数是[1: 5]，节点后的路径2所占通道数是[4: 9]。


#### 小通道的判断规则：考虑到超时或超空间的问题多次发生，因此不断简化小通道的判断规则，这在实际上丧失了相当一部分最优解。最终解决方案为按照[1: max_width]判断，如果可以通过，那么保留该条路径，反之，则舍弃该路径。


### Yen-s-k-shortest-paths-algorithm
#### C  [inf 3   2   inf inf inf]
#### D  [inf inf inf 4   inf inf]
#### E  [inf 1   inf 2   3   inf]
#### F  [inf inf inf inf 2   1  ]
#### G  [inf inf inf inf inf 2  ]
#### H  [inf inf inf inf inf inf]
####     C   D   E   F   G   H
#### 调用K条最短路径算法，源C，目的H，K为3。B为偏离路径集合。
#### 
#### 1.通过Dijkstra算法计算得到最短路径A^1：C-E-F-H，其中，花费为5，A[1] = C-E-F-H；
#### 
#### 2.将A[1]作为迭代路径，进行第一次迭代：
#### 
#### (1)以部分迭代路径(即A[1])C路径中，C点为起点，将C-E路径之间的权值设为无穷大，进行一次Dijkstra，得到路径A^2-1：C-D-F-H，花费为8，将A^2-1路径加入B；
#### 
#### (2)以部分迭代路径(即A[1])C-E路径中，E为起点，将E-F路径之间的权值设为无穷大，进行一次Dijkstra，得到路径A^2-2：C-E-G-H，花费为7，将A^2-2路径加入B；
#### 
#### (3)以部分迭代路径(即A[1])C-E-F路径中，F为起点，将F-H路径之间的权值设为无穷大，进行一次Dijkstra，得到路径A^2-3：C-E-F-G-H，花费为8，将A^2-3路径加入B；
#### 
#### 迭代完成，B集合中有三条路径：C-D-F-H，C-E-G-H，C-E-F-G-H；选出花费最小的偏离路径C-E-G-H，A[2] = C-E-G-H，移出B集合。
#### 
#### 3.将A[2]作为迭代路径，进行第二次迭代：
#### 
#### (1)以部分迭代路径(即A[2])C路径中，C点为起点，将C-E路径之间的权值设为无穷大，进行一次Dijkstra，得到路径A^3-1：C-D-F-H，但B集合已存在该路径，故不存在偏移路径；
#### 
#### (2)以部分迭代路径(即A[2])C-E路径中，E点为起点，将E-G、E-F路径之间的权值设为无穷大 (注意，这里设置两条路径的权值原因是这两条路径分别存在于A[1]和A[2]中)，进行一次Dijkstra，得到路径A^3-2：C-E-D-F-H，花费为8，将A^3-2加入B；
#### 
#### (3)以部分迭代路径(即A[2])C-E-G路径中，G点为起点，将C-H路径之间的权值设为无穷大，不存在偏移路径。
#### 
#### 迭代完成，B集合中有三条路径：C-D-F-H，C-E-F-G-H，C-E-D-F-H；由于三条路径花费均为8，则根据最小节点数进行判断，选出偏离路径C-D-F-H，A[3] = C-D-F-H。
#### 
#### 此时，选出了三条最短路径，分别是：
#### 
#### A[1] = C-E-F-H
#### 
#### A[2] = C-E-G-H
#### 
#### A[3] = C-D-F-H

### 缺点：
#### 在提交过程中遇到最多的问题是超时或者超空间，因此通过遍历所有解来寻找最优答案显然是不合理的。Yen-s-k-shortest-paths-algorithm算法是基于Dijkstra算法延伸出的前k条最短路径。Dijkstra算法解决的是有向图中的单源最短路问题,可证明Dijkstra算法所求路径是最短路径，Yen-s-k-shortest-paths-algorithm算法所求路径应是较优路径，但并不一定是排名前k的路径
#### 小通道的判断规则过于简单粗暴
#### 可以在对每条小通道做判断，即以小通道的路径为单位进行判断，这样可以节省时间复杂度，无需考虑后续的大路径。
