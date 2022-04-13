#if 1
#include <iostream>
#include <algorithm>
#include <list>
#include <queue>
#include <unordered_map>
#include <unordered_set>
using namespace std;

// 核心点: 
//       1: 访问过的下标不能再访问了，否则会死循环
//		 2: 相同元素的下标访问需要考虑性能问题 需要注意的是对于测试用例如 [1, 1, 1, 1, 2 , 2]可简化为[1, 1, 2, 2]，同一段连续相同值只需首尾两个元素即可。

class Solution {
public:
    int minJumps(vector<int>& arr) {
        
        unordered_map<int, vector<int>> idxSameValue;
        for (int i = 0; i < arr.size(); i++) {
            idxSameValue[arr[i]].push_back(i);
        }
        
        unordered_set<int> visitedIndex;
        queue<pair<int, int>> q;

        q.emplace(0, 0);  // first: 下标 second: 步数
        visitedIndex.emplace(0);
        
        while (!q.empty()) {
            pair<int, int> pair = q.front();
            q.pop();
            
            if (pair.first == arr.size() - 1) {
                return pair.second;
            }
            
            int v = arr[pair.first];
            pair.second++;
            
            // arr[i] == arr[j] && i != j
            if (idxSameValue.count(v)) {
                for (auto& i : idxSameValue[v]) {
                    if (!visitedIndex.count(i)) {
                        visitedIndex.emplace(i);
                        q.emplace(i, pair.second);
                    }
                }

                idxSameValue.erase(v); // 清空所有子图， 目的是无需重复访问该子图的其他边了 时间优化O(n*n) -> O(n)
            }

            // i + 1
            if (pair.first + 1 < arr.size() && !visitedIndex.count(pair.first + 1)) {
                visitedIndex.emplace(pair.first + 1);
                q.emplace(pair.first + 1, pair.second);
            }

            // i - 1
            if (pair.first - 1 >= 0 && !visitedIndex.count(pair.first - 1)) {
                visitedIndex.emplace(pair.first - 1);
                q.emplace(pair.first - 1, pair.second);
            }
        }

        return -1;
    }
};

// 扑克牌:从牌顶拿出一张牌放到桌子上, 再从牌顶拿一张牌放在手上牌的底部，重复第一步、第二步的操作
class SolutionPokers {
public:
    /**
    * 正向，从手到桌子 t2 = {1,12,2,8,3,11,4,9,5,13,6,10,7};
    * 返回 ｛13,12,11,10,9,8,7,6,5,4,3,2,1};
    *
    * @param pokers
    */
    void sort2(vector<int> pokers) {
        
        // 13张牌转换成数组  方便操作，不用考虑太多
        list<int> pokerList;
        for (int poker : pokers) {
            pokerList.emplace_back(poker);
        }
        
        // 声明一个新的容器，在这里可以理解成桌子
        list<int> newPokers2;
        for (int i = 0; i < pokers.size(); i++) {
            
            // 将手牌中的第一张放在桌子上
            newPokers2.emplace_back(pokerList.front());
            pokerList.pop_front();
            
            // 假如这是最后一次循环手牌已经没有了就不需要进入这个判断了
            if (pokerList.size() > 0) {
                // 将第一张放在牌堆的最后
                pokerList.emplace_back(pokerList.front());
                pokerList.pop_front();
            }

            for (auto item : newPokers2) {
                cout << item << " ";
            }

            cout << endl;
        }
        
        // 循环打印到控制台，
        for (auto item : newPokers2) {
            cout << item << " ";
        }

        cout << endl;
    }

    /**
     * 这里的操作是从桌子把牌拿回到手上
     * 从桌子 到 手上 int[] t = {13,12,11,10,9,8,7,6,5,4,3,2,1};
     * 返回 {1,12,2,8,3,11,4,9,5,13,6,10,7}
     *
     * @param pokers
     */
    void sort(vector<int> pokers) {
        
        // 从数组转换成list,只是为了方便操作，不用考虑其它的
        list<int> pokerList;
        for (int poker : pokers) {
            pokerList.emplace_back(poker);
        }

        // 声明一个目标容器，理解成手
        list<int> newPokers2;

        for (int aPoker : pokerList) {
            
            // 核心点1: 判断手上的牌是否大于1张 
            if (newPokers2.size() > 1) {
                // 如果大于一张，则把手牌的最后一张放在最上面
                newPokers2.emplace_front(newPokers2.back());
                newPokers2.pop_back();
            }
            
            // 核心点2: 从桌子上拿一张牌放在手上
            newPokers2.emplace_front(aPoker);

            for (auto item : newPokers2) {
                cout << item << " ";
            }

            cout << endl;
        }

        // 循环打印到控制台
        for (auto item : newPokers2) {
            cout << item << " "; 
        }

        cout << endl;
    }
};

int main() {
	// Hard-LC1345
    // Solution s;
	// vector<int> nums = { 100,-23,-23,404,100,23,23,23,3,404 };
	// cout << s.minJumps(nums);

    // pokers
    vector<int> nums = /*{ 1, 12, 2, 8, 3, 11, 4, 9, 5, 13, 6, 10, 7 };*/{ 13,12,11,10,9,8,7,6,5,4,3,2,1 };
    SolutionPokers ps;
    ps.sort(nums); // 将牌从桌子上还原到手上. 核心逻辑:
    // ps.sort2(nums); // 将牌从手上放到桌子上. 原始逻辑
	return 0;
}
#endif