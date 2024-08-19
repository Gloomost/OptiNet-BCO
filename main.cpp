#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <map>
#include <string>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <stack>
#include <queue>
#include <limits>

using namespace std;

class Business {
public:
    Business() {};
    // 构造函数，初始化大部分成员，但pathEdges留空待后续设置
    Business(int businessId, int src, int snk, int edgeNum, int value)
        : businessId(businessId),
        value(value),
        status(true),
        startEnd(std::make_pair(src, snk)),
        edgeCount(edgeNum),
        pathNodes(), // 初始化为空
        pathEdges(), // 初始化为空，将在后续通过特定方法设置
        edgeSubPaths(){}

    // 设置经过的边编号列表，并依据[L, R]初始化占用通道列表
    void setPathEdges(const std::vector<int>& edgeIds, int L, int R) {
        pathEdges = edgeIds;
        for (int edgeId : edgeIds) {
            std::vector<int> subPathIds;
            for (int i = L; i <= R; ++i) {
                subPathIds.push_back(i);
            }
            edgeSubPaths[edgeId] = subPathIds;
        }
    }
    // 更新经过的边数目
    void updateEdgeCount(int newEdgeCount) {
        edgeCount = newEdgeCount;
    }

    // 更新途径的节点列表
    void updatePathNodes(const std::vector<int>& nodes) {
        pathNodes = nodes;
    }

    // 更新途径的边编号列表
    void updatePathEdges(const std::vector<int>& edges) {
        pathEdges = edges;
    }

    // 更新经过的边及其对应的占用小边编号列表
    void updateEdgeSubPaths(const std::map<int, std::vector<int>>& subPaths) {
        edgeSubPaths = subPaths;
    }

    void clearpathNodes() {
        pathNodes.clear();
    }

    void clearPathEdges() {
        pathEdges.clear();
    }

    // 清空 edgeSubPaths
    void clearEdgeSubPaths() {
        edgeSubPaths.clear();
    }

    // 获取业务编号
    int getBusinessId() const { return businessId; }

    // 获取业务价值
    int getValue() const { return value; }

    // 获取业务状态
    bool getStatus() const { return status; }

    // 获取起点终点
    std::pair<int, int> getStartEnd() const { return startEnd; }

    // 获取经过的边数目
    int getEdgeCount() const { return edgeCount; }

    // 获取途径的节点列表
    const std::vector<int>& getPathNodes() const { return pathNodes; }

    // 获取途径的边编号列表
    const std::vector<int>& getPathEdges() const { return pathEdges; }

    // 获取经过的边及其对应的占用小边编号列表
    const std::map<int, std::vector<int>>& getEdgeSubPaths() const { return edgeSubPaths; }

    //修改业务的状态
    void setStatus(bool newStatus) {
        status = newStatus;
    }

    int getWidth() {
        if (edgeSubPaths.empty()) {
            return 0; // 或者其他合适的默认值
        }
            return edgeSubPaths.begin()->second.size();
        }
    

    void print() const {
        std::cout << "Business ID: " << businessId << std::endl;
        std::cout << "Value: " << value << std::endl;
        std::cout << "Status: " << (status ? "Active" : "Inactive") << std::endl;
        std::cout << "Start-End: (" << startEnd.first << ", " << startEnd.second << ")" << std::endl;
        std::cout << "Edge Count: " << edgeCount << std::endl;

        std::cout << "Path Nodes: ";
        for (int node : pathNodes) {
            std::cout << node << " ";
        }
        std::cout << std::endl;

        std::cout << "Path Edges: ";
        for (int edge : pathEdges) {
            std::cout << edge << " ";
        }
        std::cout << std::endl;

        std::cout << "Edge SubPaths:" << std::endl;
        for (const auto& pair : edgeSubPaths) {
            std::cout << "  Edge ID: " << pair.first << " SubPath IDs: ";
            for (int subPathId : pair.second) {
                std::cout << subPathId << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    int businessId; // 业务编号
    int value; // 业务价值
    bool status; // 业务状态
    std::pair<int, int> startEnd; // 业务途径的起点和终点
    int edgeCount; // 经过的边数目
    std::vector<int> pathNodes; // 途径的节点列表
    std::vector<int> pathEdges; // 途径的边编号列表
    std::map<int, std::vector<int>> edgeSubPaths; // 经过的边及其对应的占用小边编号列表
};


class Edge {
public:
    Edge() {};
    // 构造函数
    Edge(int id, int startNode, int endNode)
        : edgeId(id),
        nodes(std::make_pair(startNode, endNode)),
        availableChannelsCount(40),
        workingStatus(true) {}

    // 获取Edge的ID
    int getId() const { return edgeId; }

    // 获取边的节点信息
    std::pair<int, int> getNodes() const {
        return nodes;
    }

    // 添加途径业务的编号
    void addBusinessId(int businessId) {
        routeBusinessIds.push_back(businessId);
    }

    // 添加占用的小道编号
    void occupyChannel(int channel) {
        occupiedChannels.insert(channel);
        availableChannelsCount = 40 - static_cast<int>(occupiedChannels.size());
    }

    // 获取途径业务的编号列表
    const std::vector<int>& getBusinessIds() const {
        return routeBusinessIds;
    }

    void removeBusinessId(int businessId) {
    // 使用 std::remove_if 将要删除的元素移到末尾，然后使用 erase 删除末尾的元素
        routeBusinessIds.erase(std::remove(routeBusinessIds.begin(), routeBusinessIds.end(), businessId), routeBusinessIds.end());
    }

    // 获取占用的小道集合
    const std::unordered_set<int>& getOccupiedChannels() const { return occupiedChannels; }

    // 获取不占用的小道集合
    std::vector<int> getUnoccupiedChannels() const {
        std::vector<int> fullSet;
        for (int i = 1; i <= 40; ++i) {
            fullSet.push_back(i);
        }
        
        std::vector<int> unoccupiedSet;
        std::set_difference(fullSet.begin(), fullSet.end(),
                            occupiedChannels.begin(), occupiedChannels.end(),
                            std::inserter(unoccupiedSet, unoccupiedSet.end()));
        
        return unoccupiedSet;
    }
    // 改变工作状态
    void setWorkingStatus(bool status) {
        workingStatus = status;
    }

    // 获取当前工作状态
    bool isWorking() const {
        return workingStatus;
    }

    // 设置边中断
    void clearAndSetOccupiedChannels(int count = 40) {
        occupiedChannels.clear();
        for (int i = 1; i <= count; ++i) {
            occupiedChannels.insert(i);
        }
        availableChannelsCount = 0;
    }

    // 释放占用的小道编号，并更新可用通道数
    void releaseChannel(int channel) {
        if (occupiedChannels.erase(channel) > 0) {
            availableChannelsCount = 40 - static_cast<int>(occupiedChannels.size());
        }
    }

private:
    int edgeId; // 边的ID
    std::pair<int, int> nodes; // 边的两端节点
    std::vector<int> routeBusinessIds; // 途径该边的业务编号列表
    std::unordered_set<int> occupiedChannels; // 被占用的小道编号集合
    int availableChannelsCount; // 可用通道数
    bool workingStatus; // 工作状态
};


int readInt() {
    int value;
    cin >> value;
    return value;
}

// 定义生成邻接矩阵的函数
vector<vector<int>> generateAdjacencyMatrix(const vector<Edge>& edges, int N) {
    vector<vector<int>> adjMatrix(N, vector<int>(N, 0)); // 初始化为全0矩阵

    for (const Edge& edge : edges) {
        auto nodes = edge.getNodes(); // 获取边的两个节点
        int node1 = nodes.first;
        int node2 = nodes.second;

        // 因为边是无向的，所以我们同时增加 node1->node2 和 node2->node1 的计数
        adjMatrix[node1 - 1][node2 - 1] += 1;
        adjMatrix[node2 - 1][node1 - 1] += 1; // 注意减1是因为索引从0开始
    }

    return adjMatrix;
}

/*定义重置为初始状态的函数
 在需要重置数据的地方调用...
resetToInitialState(edges, businesses, maxChannelsPerNode, adjMatrix,
    originalEdges, originalBusinesses, originalMaxChannelsPerNode, originalAdjMatrix);*/
void resetToInitialState(std::vector<Edge>& edges,
    std::vector<Business>& businesses,
    std::vector<int>& maxChannelsPerNode,
    std::vector<std::vector<int>>& adjMatrix,
    const std::vector<Edge>& originalEdges,
    const std::vector<Business>& originalBusinesses,
    const std::vector<int>& originalMaxChannelsPerNode,
    const std::vector<std::vector<int>>& originalAdjMatrix) {
    edges = originalEdges; // 还原edges到初始状态
    businesses = originalBusinesses; // 还原businesses到初始状态
    maxChannelsPerNode = originalMaxChannelsPerNode; // 还原maxChannelsPerNode到初始状态
    adjMatrix = originalAdjMatrix; // 还原adjMatrix到初始状态
}





void printAdjMatrix(const std::vector<std::vector<int>>& adjMatrix) {
    std::cout << "Adjacency Matrix:" << std::endl;
    for (const auto& row : adjMatrix) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


//打印结构体以检查的函数
void printEdges(const std::vector<Edge>& edges, const std::string& name) {
    std::cout << name << " (" << edges.size() << " edges):" << std::endl;
    for (size_t i = 0; i < edges.size(); ++i) {
        const auto& edge = edges[i];
        std::cout << "Edge [" << i << "]: ID = " << edge.getId()
            << ", Nodes = (" << edge.getNodes().first << ", " << edge.getNodes().second << ")"
            << ", Working Status = " << edge.isWorking()
            << ", Occupied Channels = ";
        for (const int channel : edge.getOccupiedChannels()) {
            std::cout << channel << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printBusinesses(const std::vector<Business>& businesses, const std::string& name) {
    std::cout << name << " (" << businesses.size() << " businesses):" << std::endl;
    for (size_t i = 0; i < businesses.size(); ++i) {
        const auto& business = businesses[i];
        std::cout << "Business [" << i << "]: ID = " << business.getBusinessId()
            << ", Value = " << business.getValue()
            << ", Status = " << (business.getStatus() ? "Active" : "Inactive")
            << ", Path Edges = ";
        for (const int edgeId : business.getPathEdges()) {
            std::cout << edgeId << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}




std::vector<int> setFaultStatus(int efailed, std::vector<Edge>& edges, std::vector<Business>& businesses, std::vector<std::vector<int>>& adjMatrix) {
    // 1. 改变对应Edge的workingStatus为false，并更新通道信息
    edges[efailed - 1].setWorkingStatus(false);
    edges[efailed - 1].clearAndSetOccupiedChannels();

    // 2. 查询经过该边的业务，设置其status为false，
    std::vector<int> affectedBusinessIds = edges[efailed - 1].getBusinessIds();

    // for (int businessId : affectedBusinessIds) {
    //     businesses[businessId - 1].setStatus(false); // 设置业务状态为false
    // }

    // 3.更新邻接矩阵
    std::pair<int, int> nodes = edges[efailed - 1].getNodes(); // 获取边的两个端点
    adjMatrix[nodes.first - 1][nodes.second - 1]--; // 减少一个计数
    adjMatrix[nodes.second - 1][nodes.first - 1]--; // 对称更新

    // 4.按照Value值从大到小排序并输出业务ID
    std::sort(affectedBusinessIds.begin(), affectedBusinessIds.end(),
        [&businesses](int a, int b) {
            return businesses[a - 1].getValue() > businesses[b - 1].getValue(); // 使用lambda表达式访问Business对象的价值进行比较
        });

    return affectedBusinessIds;
}

struct Path {
    vector<int> path;
    int cost;

    Path(const vector<int>& p, int c) : path(p), cost(c) {}

    bool operator<(const Path& other) const {
        return cost > other.cost;
    }
};

vector<int> dijkstra(const vector<vector<int>>& adjMatrix, int start, int end, const set<pair<int, int>>& forbiddenEdges = {}) {
    int n = adjMatrix.size();
    vector<int> dist(n, numeric_limits<int>::max());
    vector<int> prev(n, -1);
    dist[start] = 0;

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [currentDist, currentNode] = pq.top();
        pq.pop();

        if (currentNode == end) {
            vector<int> path;
            for (int at = end; at != -1; at = prev[at]) {
                path.insert(path.begin(), at);
            }
            return path;
        }

        if (currentDist > dist[currentNode]) continue;

        for (int neighbor = 0; neighbor < n; ++neighbor) {
            if (adjMatrix[currentNode][neighbor] != 0 && forbiddenEdges.find({currentNode, neighbor}) == forbiddenEdges.end()) {
                int newDist = currentDist + adjMatrix[currentNode][neighbor];
                if (newDist < dist[neighbor]) {
                    dist[neighbor] = newDist;
                    prev[neighbor] = currentNode;
                    pq.push({newDist, neighbor});
                }
            }
        }
    }

    return {}; // Return empty if no path found
}

vector<vector<int>> yenKShortestPaths(const vector<vector<int>>& adjMatrix, int start, int end, int k) {
    vector<vector<int>> paths;
    priority_queue<Path> pq;
    set<pair<int, int>> forbiddenEdges;

    vector<int> initialPath = dijkstra(adjMatrix, start, end);
    if (initialPath.empty()) return paths;

    int initialCost = 0;
    for (int i = 1; i < initialPath.size(); ++i) {
        initialCost += adjMatrix[initialPath[i - 1]][initialPath[i]];
    }
    pq.push(Path(initialPath, initialCost));

    while (!pq.empty() && paths.size() < k) {
        Path currentPath = pq.top();
        pq.pop();
        paths.push_back(currentPath.path);

        for (int i = 0; i < currentPath.path.size() - 1; ++i) {
            int spurNode = currentPath.path[i];
            vector<int> rootPath(currentPath.path.begin(), currentPath.path.begin() + i + 1);

            for (const auto& path : paths) {
                if (equal(path.begin(), path.begin() + rootPath.size(), rootPath.begin())) {
                    forbiddenEdges.insert({path[i], path[i + 1]});
                }
            }

            vector<int> spurPath = dijkstra(adjMatrix, spurNode, end, forbiddenEdges);
            if (!spurPath.empty()) {
                vector<int> totalPath = rootPath;
                totalPath.insert(totalPath.end(), spurPath.begin() + 1, spurPath.end());

                int totalCost = 0;
                for (int j = 1; j < totalPath.size(); ++j) {
                    totalCost += adjMatrix[totalPath[j - 1]][totalPath[j]];
                }

                pq.push(Path(totalPath, totalCost));
            }

            forbiddenEdges.clear();
        }
    }

    return paths;
}

// 自定义哈希函数，用于将 std::pair<int, int> 作为 unordered_map 的键
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

// 定义图的数据结构，使用 unordered_map 来存储边
using NodePairToEdgesMap = std::unordered_map<std::pair<int, int>, std::vector<const Edge*>, pair_hash>;

// 根据边的列表构建图
NodePairToEdgesMap buildGraph(const std::vector<Edge>& edges) {
    NodePairToEdgesMap graph;
    for (const Edge& edge : edges) {
        if (edge.isWorking()){
            graph[edge.getNodes()].push_back(&edge);
        }
    }
    return graph;
}


std::vector<std::vector<int>> findAllPossiblePaths(const std::vector<int>& nodeSequence,
    std::vector<Edge>& edges, Business& business, vector<int>& maxChannelsPerNode,
    std::function<bool(vector<int>&, vector<Edge>&, Business&, vector<int>&)> condition) {

    // 构建图
    NodePairToEdgesMap graph = buildGraph(edges);
    std::vector<std::vector<int>> result;
    std::vector<int> currentPath;

    std::stack<std::tuple<size_t, int, std::vector<int>>> dfs_stack;  // 使用栈来模拟递归调用
    dfs_stack.push({0, nodeSequence[0]+1, currentPath});  // 初始状态入栈

    while (!dfs_stack.empty()) {
        size_t currentIndex = std::get<0>(dfs_stack.top());
        int currentNode = std::get<1>(dfs_stack.top());
        std::vector<int> currentPath = std::get<2>(dfs_stack.top());
        dfs_stack.pop();

        if (currentIndex == nodeSequence.size() - 1) {
            if (condition(currentPath, edges, business, maxChannelsPerNode)) {
                result.push_back(currentPath);
            }
            continue;
        }

        std::pair<int, int> nodePair = { nodeSequence[currentIndex] + 1, nodeSequence[currentIndex + 1] + 1 };
        if (graph.find(nodePair) != graph.end()) {
            for (const Edge* edge : graph.at(nodePair)) {
                currentPath.push_back(edge->getId());
                dfs_stack.push({currentIndex + 1, edge->getId(), currentPath});
                currentPath.pop_back();
            }
        }
    }

    return result;

}


// bool judgeOk(vector<Edge>& edge, Business& business, vector<int>& maxChannelsPerNode)
// 函数用于计算vector<int>中元素的数量
int pathSize(const vector<int>& path) {
    return path.size();
}
// 按节点数对大路路径进行排序，将经过节点数少的路径放到前面
vector<vector<int>> bigPathByPots(vector<vector<int>> allPath) {
    // 使用自定义排序，以路径的大小为排序依据
    sort(allPath.begin(), allPath.end(), [](const vector<int>& a, const vector<int>& b) {
        return pathSize(a) < pathSize(b);
        });
    return allPath;
}


std::vector<std::vector<int>> getContinuous(const std::vector<int>& paths, int n) {
    std::vector<std::vector<int>> result;

    // Ensure paths has at least n elements
    if (paths.size() < n) {
        return result;  // Return an empty vector if paths has fewer than n elements
    }

    // Iterate over paths to find all continuous subarrays of length n
    for (size_t i = 0; i <= paths.size() - n; ++i) {
        // Check if the current subarray is continuous
        bool isContinuous = true;
        for (int j = 1; j < n; ++j) {
            if (paths[i + j] != paths[i + j - 1] + 1) {
                isContinuous = false;
                break;
            }
        }

        // If the subarray is continuous, add it to the result
        if (isContinuous) {
            result.push_back(std::vector<int>(paths.begin() + i, paths.begin() + i + n));
        }
    }

    return result;
}



// Function to find intersection of subarrays between v1 and v2
std::vector<std::vector<int>> getCommon(const std::vector<std::vector<int>>& v1, const std::vector<std::vector<int>>& v2) {
    std::vector<std::vector<int>> result;

    // Ensure v1 and v2 have the same size
    if (v1.size() != v2.size()) {
        return result;  // Return empty vector if sizes are different
    }

    // Iterate over each pair of corresponding subarrays in v1 and v2
    for (size_t i = 0; i < v1.size(); ++i) {
        // Check if v1[i] and v2[i] are equal
        if (v1[i] == v2[i]) {
            result.push_back(v1[i]); // or v2[i], since they are equal
        }
    }

    return result;
}


bool judgeOk(vector<int>& currentPath, vector<Edge>& edge, Business& business, vector<int>& maxChannelsPerNode) {
    vector<int> change_node;
    vector<vector<int>> temp;
    vector<vector<int>> channels;
    std::map<int, std::vector<int>> subPaths;
    int temp_num = 0;

    for (int i = 0; i < currentPath.size(); ++i) {
        vector<vector<int>> contin = getContinuous(edge[currentPath[i] - 1].getUnoccupiedChannels(), business.getWidth());
        if (contin.empty()) {
            return false; // 无法找到连续的小通道
        }

        if (i == 0) {
            temp = contin;
        } else {
            vector<vector<int>> cur = getCommon(temp, contin);
            if (cur.empty()) {
                int start_node = edge[currentPath[i] - 1].getNodes().first;
                int end_node = edge[currentPath[i] - 1].getNodes().second;
                if (maxChannelsPerNode[start_node] == 0) {
                    return false; // 节点上的最大通道数已用完
                }
                for (int j = temp_num; j < i; ++j) {
                    channels.push_back(temp[0]);
                }
                temp_num = i;
                change_node.push_back(start_node);
            } else {
                temp = cur;
            }
        }
    }

    // 将剩余的通道加入channels
    for (int j = temp_num; j < currentPath.size(); ++j) {
        channels.push_back(temp[0]);
    }


    for (int node : change_node) {
        maxChannelsPerNode[node - 1]--;
    }

    // 释放之前业务占据的通道
    std::map<int, std::vector<int>> oriSubPaths = business.getEdgeSubPaths();
    for (const auto& pair : oriSubPaths) {
        int key = pair.first;
        const std::vector<int>& value = pair.second;
        edge[key - 1].removeBusinessId(business.getBusinessId());
        for (int subPathId : value) {
            edge[key - 1].releaseChannel(subPathId);
        }
    }

    // 占用新的通道
    for (int i = 0; i < currentPath.size(); ++i) {
        edge[currentPath[i] - 1].addBusinessId(business.getBusinessId());
        for (int channel : channels[i]) {
            edge[currentPath[i] - 1].occupyChannel(channel);
            subPaths[currentPath[i]].push_back(channel);
        }
    }

    // 更新business信息
    business.clearpathNodes();
    business.clearPathEdges();
    business.clearEdgeSubPaths();
    business.updateEdgeSubPaths(subPaths);
    business.updateEdgeCount(currentPath.size());
    business.updatePathEdges(currentPath);
    // business.setStatus(true);

    return true;
}



int main() {
    // 读取第一行的 M 和 N
    int N, M;
    cin >> N >> M;

    //cout << "输入数据" << N << "---" << M;
    // 创建 Edge 列表
    vector<Edge> edges(M);


    // 读取并存储每个节点允许的变通道数
    // 现在从0开始，所以只需N大小
    vector<int> maxChannelsPerNode(N);

    // 因为索引从0开始，所以i也从0开始，到N-1
    for (int i = 0; i < N; ++i) {
        maxChannelsPerNode[i] = readInt(); // 用户输入时需提示从0开始输入
    }

    // 根据输入创建Edge实例
    for (int k = 0; k < M; ++k) {
        int node1, node2;
        cin >> node1 >> node2;
        edges[k] = Edge(k + 1, node1, node2);

        // 如果需要在创建Edge时考虑每个节点的最大通道数，传入调整后的索引
        // edges[k] = Edge(k + 1, node1, node2, maxChannelsPerNode[node1 - 1], maxChannelsPerNode[node2 - 1]);
    }


    // 至此，我们已经完成了Edge对象的创建，并读取了节点的最大通道数。
    int J;
    cin >> J;


    vector<Business> businesses(J);

    // 为每个Business实例读取和设置数据
    for (int j = 0; j < J; ++j) {
        // 读取单个业务的数据
        int src, snk, edgeNum, L, R, V;
        cin >> src >> snk >> edgeNum >>  L >> R >> V; 

        // 创建Business实例，暂时不设置pathEdges
        businesses[j] = Business(j + 1, src, snk, edgeNum, V);

        // 读取并设置该Business经过的边编号列表
        std::vector<int> edgeIds(edgeNum);
        for (int i = 0; i < edgeNum; ++i) {
            cin >> edgeIds[i];
        }

        // 设置Business的边编号列表及通道范围，并更新Edge状态
        for (int edgeId : edgeIds) {
            Edge& edge = edges[edgeId - 1];

            // 添加业务ID到该边的routeBusinessIds
            edge.addBusinessId(j + 1);

            // 占用通道并更新状态
            for (int channel = L; channel <= R; ++channel) {
                // businesses[j].setPathEdges(edgeIds, L, R);
                edge.occupyChannel(channel);
            }
        }
        
        
        // 假设通道范围L和R对于所有业务相同，或通过其他方式获取这两个值
        businesses[j].setPathEdges(edgeIds, L, R); // 设置经过的边编号列表及通道范围
        // cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        // businesses[j].print();
    }

    // 至此，所有Business对象已根据输入创建并配置完毕。


    // 调用函数生成邻接矩阵
    vector<vector<int>> adjMatrix = generateAdjacencyMatrix(edges, N);

    // 打印邻接矩阵
    // std::cout << "Adjacency Matrix:\n";
    // printAdjacencyMatrix(adjMatrix);


    // 假设已经完成了M, N, J, edges, businesses, maxChannelsPerNode, adjMatrix的初始化...

    // 创建“母版”数据结构
    std::vector<Edge> originalEdges = edges; // 深拷贝edges
    std::vector<Business> originalBusinesses = businesses; // 深拷贝businesses
    std::vector<int> originalMaxChannelsPerNode = maxChannelsPerNode; // 浅拷贝足够，因为vector<int>包含的是基本类型
    std::vector<std::vector<int>> originalAdjMatrix = adjMatrix; // 深拷贝二维vector


    int T;
    cin >> T; // 读取测试场景数量

    for (int testScenario = 1; testScenario <= T; ++testScenario) {
        while (true) {
            int efailed;

            cin >> efailed;//efailed即为死亡边编号
            if (efailed == -1) break; // 结束当前测试场景

            // 调用setFaultStatus并传递邻接矩阵参数，处理返回的排序业务ID序列，sortedBusinessIds即为根据业务附加值排好序的业务列表
            std::vector<int> sortedBusinessIds = setFaultStatus(efailed, edges, businesses, adjMatrix);
            
            // printEdges(edges, "Generated Edges");
            // printEdges(originalEdges, "Original Edges");

            // printBusinesses(businesses, "Generated Businesses");
            // printBusinesses(originalBusinesses, "Original Businesses");

            // cout << "Generated Adjacency Matrix:\n";
            // printAdjMatrix(adjMatrix);
            
            

            // cout << "num  " << sortedBusinessIds.size() << endl;
            std::vector<int> plannedBusinesses; // 成功规划的业务编号集合，与交互输出结果部分照应，先设定用以保存结果
            for (const int& element : sortedBusinessIds) { //这里原本是char & element，改成int
                if (!businesses[element - 1].getStatus()) { 
                    // printEdges(edges, "Generated Edges");
                    // printEdges(originalEdges, "Original Edges");

                    // businesses[element - 1].print();

                    // cout << "Generated Adjacency Matrix:\n";
                    // printAdjMatrix(adjMatrix);
                    continue; 
                }
                // cout << "element:" << element << endl;
                // step7.1
                // 根据element在结构体中查找该业务的startNode和endeNode，函数为getBusinessNode，输入应该为Business的结构体，输出为vector<int>的数据类型
                // 获取 startEnd 并将其分解为两个 int 变量
                std::pair<int, int> startEnd = businesses[element - 1].getStartEnd();
                int startNode = startEnd.first - 1;
                int endNode = startEnd.second - 1;

                // step7.2
                // 获取该业务下不经过死亡路径的所有可能的大路路径，按节点的路径
                vector<vector<int>> allPaths = yenKShortestPaths(adjMatrix, startNode, endNode, 5);
                // cout << "7.2" << endl;
                // for (const auto& row : allPaths) {
                //     for (int num : row) {
                //         std::cout << num << " ";
                //     }
                //     std::cout << std::endl;
                // }
                // step7.3
                // 对大路路径进行排序
                vector<vector<int>> allPathsByPotNum = bigPathByPots(allPaths);
                vector<vector<int>> okPath;
                // step7.4
                // 获取已经排好序的大路路径下最先成功的一条路径
                // vector<Edge> edge = edges;
                // cout << "findAllPossiblePaths" << endl;
                for (const auto& bigPath : allPathsByPotNum) {
                    okPath = findAllPossiblePaths(bigPath, edges, businesses[element - 1], maxChannelsPerNode, judgeOk);
                    if(okPath.empty()){
                        continue;
                    }else{
                        businesses[element - 1].updatePathNodes(bigPath);
                        break;
                    }
                }
                
                // std::vector<int> newEdges;
                    if (okPath.empty()) {
                        businesses[element - 1].setStatus(false);
                        continue;
                    } else{
                        businesses[element - 1].setStatus(true);
                        plannedBusinesses.push_back(element);
                        // newEdges = okPath[0];
                    }
            }


            // 输出规划成功的业务数量
            int R = plannedBusinesses.size();
            std::cout  << R << std::endl;
            fflush(stdout);
            // 遍历成功规划的业务并输出详细信息
            for (const int& businessIndex : plannedBusinesses) {
                const Business& business = businesses[businessIndex - 1];
                int businessId = business.getBusinessId(); // 获取业务编号
                int edgeCount = business.getEdgeCount(); // 获取经过的边数目

                // 输出业务编号和路径边数
                std::cout << businessId << " " << edgeCount << std::endl;
                fflush(stdout);
                // std::cout << "BussinessId: " << businessId << "  numOfEdge: " << edgeCount << std::endl;
                // fflush(stdout);

                const auto& edgeSubPaths = business.getEdgeSubPaths(); // 获取边及其对应的占用小边编号列表
                for (const auto& pair : edgeSubPaths) {
                    int edgeId = pair.first;
                    const std::vector<int>& subPathIds = pair.second;
                    int L = subPathIds.front();
                    int R = subPathIds.back();
                    // 输出每条边的编号以及占用的通道区间
                    std::cout << edgeId << " " << L << " " << R << " ";
                }
                std::cout << std::endl;
                fflush(stdout);
            }           
        }

        // 每个测试场景结束后重置状态
        resetToInitialState(edges, businesses, maxChannelsPerNode, adjMatrix,
            originalEdges, originalBusinesses, originalMaxChannelsPerNode, originalAdjMatrix);
    }

    return 0;
}

