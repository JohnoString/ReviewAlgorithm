#if 0
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
using namespace std;

// 1 两数之和
vector<int> TwoSum(const vector<int>& nums, int target) {
	if (nums.empty()) {
		return {};
	}

	// 第一版：双循环

	* for (int i = 0; i < nums.size(); ++i) {
		for (int j = 0; j < i; ++j) {
			if (nums[i] + nums[j] == target) {
				return {i, j};
			}
		}
	}

	return {-1, -1};
	
	// 第二版：hash优化 O(n2) -> O(n)
	// 核心：左右索引会访问两次第一次存hash，第二次查找hash。如果有就可以找得到了
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

// 2 两数相加

// 3 无重复字符的最长子串
#endif