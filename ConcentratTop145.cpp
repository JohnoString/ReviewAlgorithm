#if 1
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
	for (int i = 0; i < nums.size(); ++i) {
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

// 2 两数相加
// 3 无重复字符的最长子串
// 4 寻找两个有序数组的中位数
class Solution4 {
public:
	/* 
	 * 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
	 * 这里的 "/" 表示整除
	 * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
	 * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
	 * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
	 * 这样 pivot 本身最大也只能是第 k-1 小的元素
	 * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
	 * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
	 * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
	 */
	double getKthElement(vector<int> nums1, vector<int> nums2, int k) { // k从1开始
		int m = nums1.size();
		int n = nums2.size();
		int index1 = 0, index2 = 0;

		while (true) {
			// 边界情况
			if (index1 == m) {
				return nums2[index2 + k - 1];
			}

			if (index2 == n) {
				return nums1[index1 + k - 1];
			}

			if (k == 1) {
				return min(nums1[index1], nums2[index2]);
			}

			// 正常情况
			int newIndex1 = min(index1 + k / 2 - 1, m - 1);
			int newIndex2 = min(index2 + k / 2 - 1, n - 1);
			int pivot1 = nums1[newIndex1];
			int pivot2 = nums2[newIndex2];
			
			if (pivot1 <= pivot2) {
				k -= newIndex1 - index1 + 1;
				index1 = newIndex1 + 1;
			}
			else {
				k -= newIndex2 - index2 + 1;
				index2 = newIndex2 + 1;
			}
		}
	}

	double findMedianSortedArrays(vector<int> nums1, vector<int> nums2) {
		int totalLength = nums1.size() + nums2.size();
		if (totalLength % 2 == 1) {
			// 长度和为奇数
			return getKthElement(nums1, nums2, (totalLength + 1) / 2);
		}
		else {
			// 长度和为偶数
			return (getKthElement(nums1, nums2, totalLength / 2) + getKthElement(nums1, nums2, totalLength / 2 + 1)) / 2.0;
		}
	}
};

class Solution5 {
public:
	string longestPalindrome(string s) {
		int n = s.size();
		if (n < 2) {
			return s;
		}

		int maxLen = 1;
		int begin = 0;

		vector<vector<int>> dp(n, vector<int>(n));

		for (int i = 0; i < n; ++i) {
			dp[i][j] = true;
		}

		for (int L = 2; L <= n; ++L) {
			for (int i = 0; i < n; ++i) {
				int j = L + i - 1;
				if (j >= n) {
					break;
				}

				if (s[i] != s[j]) {
					dp[i][j] = false;
				}
				else {
					if (j - i < 3) {
						dp[i][j] = true;
					}
					else {
						dp[i][j] = dp[i + 1][j - 1];
					}
				}

				if (dp[i][j] && j - i + 1 > maxLen) {
					maxLen = j - i + 1;
					begin = i;
				}
			}
		}
		
		return s.substr(begin, maxLen);
	}
};

int main() {
	// 1. 两数之和
	// vector<int> nums = { 2, 8, 0, 7, 13 };
	// vector<int> res = TwoSum(nums, 9);
	// cout << res[0] << " " << res[1] << endl;

	// 2. 两数相加
	// 3. 无重复字符的最长子串
	// 4. 寻找两个有序数组的中位数
	// 核心点: 需要满足交叉小于等于的关系
	vector<int> nums1 = { 3, 8, 9, 10 };
	vector<int> nums2 = { 2, 4, 6, 12, 18, 20 };
	Solution4 s;
	cout << s.findMedianSortedArrays(nums1, nums2) << endl;

	// 5. 最长回文子串

	return 0;
}
#endif