#if 1
#include <iostream>
#include <algorithm>
#include <list>
#include <queue>
#include <unordered_map>
#include <unordered_set>
using namespace std;

// ���ĵ�: 
//       1: ���ʹ����±겻���ٷ����ˣ��������ѭ��
//		 2: ��ͬԪ�ص��±������Ҫ������������ ��Ҫע����Ƕ��ڲ��������� [1, 1, 1, 1, 2 , 2]�ɼ�Ϊ[1, 1, 2, 2]��ͬһ��������ֵֻͬ����β����Ԫ�ؼ��ɡ�

class Solution {
public:
    int minJumps(vector<int>& arr) {
        
        unordered_map<int, vector<int>> idxSameValue;
        for (int i = 0; i < arr.size(); i++) {
            idxSameValue[arr[i]].push_back(i);
        }
        
        unordered_set<int> visitedIndex;
        queue<pair<int, int>> q;

        q.emplace(0, 0);  // first: �±� second: ����
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

                idxSameValue.erase(v); // ���������ͼ�� Ŀ���������ظ����ʸ���ͼ���������� ʱ���Ż�O(n*n) -> O(n)
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

// �˿���:���ƶ��ó�һ���Ʒŵ�������, �ٴ��ƶ���һ���Ʒ��������Ƶĵײ����ظ���һ�����ڶ����Ĳ���
class SolutionPokers {
public:
    /**
    * ���򣬴��ֵ����� t2 = {1,12,2,8,3,11,4,9,5,13,6,10,7};
    * ���� ��13,12,11,10,9,8,7,6,5,4,3,2,1};
    *
    * @param pokers
    */
    void sort2(vector<int> pokers) {
        
        // 13����ת��������  ������������ÿ���̫��
        list<int> pokerList;
        for (int poker : pokers) {
            pokerList.emplace_back(poker);
        }
        
        // ����һ���µ������������������������
        list<int> newPokers2;
        for (int i = 0; i < pokers.size(); i++) {
            
            // �������еĵ�һ�ŷ���������
            newPokers2.emplace_back(pokerList.front());
            pokerList.pop_front();
            
            // �����������һ��ѭ�������Ѿ�û���˾Ͳ���Ҫ��������ж���
            if (pokerList.size() > 0) {
                // ����һ�ŷ����ƶѵ����
                pokerList.emplace_back(pokerList.front());
                pokerList.pop_front();
            }

            for (auto item : newPokers2) {
                cout << item << " ";
            }

            cout << endl;
        }
        
        // ѭ����ӡ������̨��
        for (auto item : newPokers2) {
            cout << item << " ";
        }

        cout << endl;
    }

    /**
     * ����Ĳ����Ǵ����Ӱ����ûص�����
     * ������ �� ���� int[] t = {13,12,11,10,9,8,7,6,5,4,3,2,1};
     * ���� {1,12,2,8,3,11,4,9,5,13,6,10,7}
     *
     * @param pokers
     */
    void sort(vector<int> pokers) {
        
        // ������ת����list,ֻ��Ϊ�˷�����������ÿ���������
        list<int> pokerList;
        for (int poker : pokers) {
            pokerList.emplace_back(poker);
        }

        // ����һ��Ŀ��������������
        list<int> newPokers2;

        for (int aPoker : pokerList) {
            
            // ���ĵ�1: �ж����ϵ����Ƿ����1�� 
            if (newPokers2.size() > 1) {
                // �������һ�ţ�������Ƶ����һ�ŷ���������
                newPokers2.emplace_front(newPokers2.back());
                newPokers2.pop_back();
            }
            
            // ���ĵ�2: ����������һ���Ʒ�������
            newPokers2.emplace_front(aPoker);

            for (auto item : newPokers2) {
                cout << item << " ";
            }

            cout << endl;
        }

        // ѭ����ӡ������̨
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
    ps.sort(nums); // ���ƴ������ϻ�ԭ������. �����߼�:
    // ps.sort2(nums); // ���ƴ����Ϸŵ�������. ԭʼ�߼�
	return 0;
}
#endif