#if 1
#include <iostream>
#include <algorithm>
#include <unordered_map>
using namespace std;

// lc 424

int getMaxLenOfReplacedString(string str, int k) {
	if (str.empty()) {
		return 0;
	}

	/*
	* // 第一版自行实现
	if (k <= 0) {
		
		// return ...
	}

	int len = str.size();
	int j = k;
	int l = 0;
	int i = 1;
	int maxLen = 0;

	unordered_map<unsigned char, bool> hash;

	for (int i = 0; i < len; ++i) {
		hash[i] = false;
	}

	while (i < len - 1) {
		while (str[i] == str[i - 1]) {
			i++;
			continue;
		}

		while (j > 0) {
			if (!hash[i]) {
				hash[i] = true;
			}

			i++;
			j--;
		}

		maxLen = max(maxLen, i - l + 1);
	}
	*/
}

// answer 
// 暴力 O(n3) 取所有字串  
/*
	1. 字串有很多公共的部分冗余重复扫描
	2. 找到长度为L且替换k次以后全部相等的子串, 就没有必要考虑长度小于L的子串;
	   如果找到长度为L且替换k次以后仍然不完全相等的子串, 没有必要再考虑长度大于L的子串.
*/

// Two Pointers 时间:O(n) 空间:O(26)=O(1)
int characterReplacement(string s, int k) {
	vector<int> nums(26); // 统计字符出现的次数
	
	int n = s.size();
	int maxn = 0;
	int left = 0, right = 0;

	while (right < n) {
		nums[s[right] - 'A']++;
		maxn = max(maxn, nums[s[right] - 'A']); // 维护一个到目前为止出现过的元素的次数的最大值

		// 注意：为什么用if可以, 这个判断只会进行一次, 因为这种情况下left会→移
		if (right - left + 1 - maxn > k) { // 目前为止出现的总元素个数 - 目前出现过的次数最多的元素 > 可变换次数 
			/* 
			由于调整k次也不能到达所有元素相同的状态, 
			所以以当前的left为窗口的左边界所能找到的目
			标子串的最大长度不会再增加了, 因为要求是连
			续的字串. 需要将left→移 
			*/
			nums[s[left] - 'A']--; // 记得当前元素出现的次数-1
			left++;
		}

		right++;
	}

	return right - left /*(right - 1) - left + 1 此时right多加了1*/;
}

int longestOnes(vector<int> nums, int k) {
	if (nums.empty()) {
		return 0;
	}

	vector<int> counts(2);

	int n = nums.size();
	int left = 0, right = 0;
	int maxNumsOfOne = 0; 

	while (right < n) {
		counts[nums[right]]++;
 		if (nums[right] == 1) {
			maxNumsOfOne = max(maxNumsOfOne, counts[nums[right]]); // 仅仅需要统计1的个数
		}

		if (right - left + 1 - maxNumsOfOne > k) {
			counts[nums[left]]--; // 注意: 别减成right了
			left++;
		}

		right++;
 	}

	return right - left;
}

int main() {
	// lc 424
	// cout << characterReplacement("ABMCDBASDFDSGSD", 2) << endl;
	
	// lc 1004
	// vector<int> nums = { 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0 };
	// cout << longestOnes(nums, 2) << endl;

	// lc 1208

	// lc 1993
	// lc 209
	// lc 76
	// lc 438
	// lc 567
	return 0;
}

#endif