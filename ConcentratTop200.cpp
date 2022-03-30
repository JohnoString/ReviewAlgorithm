#if 0
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
using namespace std;

// 1 ����֮��
vector<int> TwoSum(const vector<int>& nums, int target) {
	if (nums.empty()) {
		return {};
	}

	// ��һ�棺˫ѭ��

	* for (int i = 0; i < nums.size(); ++i) {
		for (int j = 0; j < i; ++j) {
			if (nums[i] + nums[j] == target) {
				return {i, j};
			}
		}
	}

	return {-1, -1};
	
	// �ڶ��棺hash�Ż� O(n2) -> O(n)
	// ���ģ�����������������ε�һ�δ�hash���ڶ��β���hash������оͿ����ҵõ���
	unordered_map<int, int> hash;
	for (int i = 0; i < nums.size(); ++i) {
		auto it = hash.find(target - nums[i]);
		if (hash.end() != it) {
			return { it->second, i };
		}

		hash[nums[i]] = i;
	}

	return { -1, -1 };
}

int main() {
	vector<int> nums = { 2, 8, 0, 7, 13 };
	vector<int> res = TwoSum(nums, 9);
	cout << res[0] << " " << res[1] << endl;
	return 0;
}

// 2 �������

// 3 ���ظ��ַ�����Ӵ�
#endif