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
			dp[i][i] = true;
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

class Solution7{
public:
	int reverse(int x) {
		int rev = 0;
		while (x != 0) {
			if (rev < INT_MIN / 10 || rev > INT_MAX / 10) {
				return 0;
			}

			int digit = x % 10;
			x /= 10;
			rev = rev * 10 + digit;
		}

		return rev;
	}
};

class Automaton {
	string state = "start";
	unordered_map<string, vector<string>> table = {
		{"start", {"start", "signed", "in_number", "end"}},
		{"signed", {"end", "end", "in_number", "end"}},
		{"in_number", {"end", "end", "in_number", "end"}},
		{"end", {"end", "end", "end", "end"}}
	};

	int get_col(char c) {
		if (isspace(c)) return 0;
		if (c == '+' or c == '-') return 1;
		if (isdigit(c)) return 2;
		return 3;
	}
public:
	int sign = 1;
	long long ans = 0;

	void get(char c) {
		state = table[state][get_col(c)];
		if (state == "in_number") {
			ans = ans * 10 + c - '0';
			ans = sign == 1 ? min(ans, (long long)INT_MAX) : min(ans, -(long long)INT_MIN);
		}
		else if (state == "signed")
			sign = c == '+' ? 1 : -1;
	}
};

class Solution8 {
public:
	int myAtoi(string str) {
		Automaton automaton;
		for (char c : str)
			automaton.get(c);
		return automaton.sign * automaton.ans;
	}
};

class Solution10 {
public:
	bool isMatch(string s, string p) {
		int m = s.size();
		int n = p.size();

		auto matches = [&](int i, int j) {
			if (i == 0) {
				return false;
			}

			if (p[j - 1] == '.') {
				return true;
			}

			return s[i - 1] == p[j - 1];
		};

		vector<vector<int>> f(m + 1, vector<int>(n + 1));
		f[0][0] = true;
		for (int i = 0; i <= m; ++i) {
			for (int j = 1; j <= n; ++j) {
				if (p[j - 1] == '*') {
					f[i][j] |= f[i][j - 2];
					if (matches(i, j - 1)) {
						f[i][j] |= f[i - 1][j];
					}
				}
				else {
					if (matches(i, j)) {
						f[i][j] |= f[i - 1][j - 1];
					}
				}
			}
		}

		return f[m][n];
	}
};

// 木桶原理
class Solution11 {
public:
	int maxArea(vector<int> height) {
		if (height.empty() || height.size() == 1) {
			return 0;
		}

		int l = 0;
		int r = height.size() - 1;
		int maxArea = 0;
		while (l < r) {
			int minH = min(height[r], height[l]);
			maxArea = max(maxArea, minH * (r - l));
			if (height[l] <= height[r]) {
				++l;
			}
			else {
				--r;
			}
		}

		return maxArea;
	}
};

// 回顾 TwoSum
class Solution15{
public:
	vector<int> twoSum(vector<int> nums, int target) {
		if (nums.size() < 2) {
			return {};
		}

		int n = nums.size();
		
		// 暴力
		/*
		for (int i = 0; i < n - 1; ++i) {
			for (int j = i + 1; j < n; ++j) {
				if (nums[j] == target - nums[i]) {
					return { i, j };
				}
			}
		}
		*/
		
		// hash
		unordered_map<int, int> hash; // key = nums[x]; value = x;
		for (int i = 0; i < n; ++i) { // 注意:n不能-1
			if (hash.end() != hash.find(target - nums[i])) {
				return { hash[target - nums[i]], i };
			}
			else {
				hash[nums[i]] = i; 
			}
		}

		return { -1, -1 };
	}

	vector<vector<int>> twoSums(vector<int> nums, int start, int end, int target, int value) {
		vector<vector<int>> ans;
		while (start < end) {
			int sum = nums[start] + nums[end];
			if (sum == target) {
				vector<int> result;
				result.emplace_back(value);
				result.emplace_back(nums[start]);
				result.emplace_back(nums[end]);
				ans.emplace_back(result);
				while (start < end && nums[start] == nums[start + 1]) {
					start++;
				}

				while (start < end && nums[start] == nums[end - 1]) {
					end--;
				}

				end--;
			} else if (sum < target) {
				start++;
			}
			else {
				end--;
			}
		}

		return ans;
	}

	// 1. 暴力O(n3)->O(n2 * logN)
	// 2. 双指针优化. 总结：当需要枚举数组中的两个元素时, 如果发现随着第一个元素的递增, 第二个元素是递减的, 那么就可以
	// 使用双指针的方法, 将枚举的时间复杂度从O(N2)降到O(N). 为什么是O(N)呢? 这是因为在枚举的过程每一步中, 左指针会向右移动
	// 一个位置, 而右指针回向左移动若干个位置, 这个与数组的元素个数有关. 清楚的是我们知道它一共移动的位置数有O(N). 均摊
	// 下来, 每次也向左移动一个位置. 所以时间复杂度是O(N).
	vector<vector<int>> threeSums(vector<int> nums) {
		if (nums.empty()) {
			return {};
		}

		int n = nums.size();
		std::sort(nums.begin(), nums.end());

		vector<vector<int>> ans;
		for (int i = 0; i < n; ++i) {
			if (i > 0 && nums[i] == nums[i - 1]) {
				continue;
			}

			auto result = twoSums(nums, i + 1, n - 1, -nums[i], nums[i]);
			ans.insert(ans.end(), result.begin(), result.end());
		}
		
		return ans;
	}
};

class Solution17 {
public:
	// DFS 对照画图可以清晰地理解其递归+循环过程
	void backtrack(vector<string>& combinations, const unordered_map<char, string>& phoneMap,
					const string &digits, int index, string& combination) {
		
		if (index == digits.size()) {
			combinations.emplace_back(combination);
		}
		else {
			char digit = digits[index];
			const string& letters = phoneMap.at(digit);
			for (const char& letter : letters) {
				combination.push_back(letter);
				backtrack(combinations, phoneMap, digits, index + 1, combination);
				combination.pop_back();
			}
		}
	}

	vector<string> letterCombinations(string digits) {
		if (digits.empty()) {
			return {};
		}

		vector<string> combitions; 
		unordered_map<char, string> phoneMap = 
		{{'2', "abc"}, {'3', "def"},
		{'4', "ghi"}, {'5', "jkl"},
		{'6', "mno"}, {'7', "pqrs"},
		{'8', "tuv"}, {'9', "wxyz"}};

		string combition;
		backtrack(combitions, phoneMap, digits, 0, combition);
		return combitions;
	}

	// BFS 时间复杂度跟DFS一样
	vector<string> letterCombinationsBfs(string digits) {
	}
};

class ListNode {
public:
	ListNode(int val) {
		this->val = val;
		this->next = nullptr;
	}

	int val;
	ListNode* next;
};

class Solution19 {
public:
	ListNode* removeNthNodeFromEnd(ListNode* node, int n) {
		/* self 实现
		if (node == nullptr || n <= 0) {
			return nullptr;
		}

		if (node->next == nullptr && n == 1) {
			return node;
		}

		ListNode* head = node;
		ListNode* first = node;
		ListNode* second = node;

		while (second->next != nullptr) {
			second = second->next;
			n--;
		}

		// n大于节点个数
		if (second == nullptr && n > 0) {
			return nullptr;
		}

		// 找到first删除
		while (second->next != nullptr) {
			first = first->next;
			second = second->next;
		}

		return head;
		*/
		
		// 总结: dummy节点用法, 解决第一个节点被删除需要特殊处理的情况
		// 先画图 加深记忆
		ListNode dummy(ListNode(-1));
		dummy.next = node;

		ListNode* first = node; 
		ListNode* second = &dummy; // 节点index计算从1开始, second从哑节点开始

		for (int i = 0; i < n; ++i) {
			first = first->next;
		}

		while (first) {
			first = first->next;
			second = second->next;
		}

		delete second->next;
		second->next = second->next->next;

		return &dummy;
	}
};

class Solution22 {
public:
	// DFS 时间O(2^2n) 空间O(n)
	bool isVaild(string str) {
		int balance = 0;
		for (auto item : str) {
			if (item == '(') {
				balance++;
			}
			else {
				balance--;
			}

			// 处理类似 "))(("情况
			if (balance < 0) {
				return false;
			}
		}

		return balance == 0;
	}

	void backTrack(int n, string& res, vector<string>& results) {
		if (2 * n == res.size()) {
			if (isVaild(res)) {
				results.emplace_back(res);
			}
			
			return;
		}

		res += "(";
		backTrack(n, res, results);
		res.pop_back();
		res += ")";
		backTrack(n, res, results);
		res.pop_back();
	}

	vector<string> generateParenthesis(int n) {
		if (n == 0) {
			return { "" };
		}

		if (n == 1) {
			return { "()" };
		}

		vector<string> results;
		string res;
		backTrack(n, res, results);
		return results;
	}

	void backTrackHS(int n, string& res, vector<string>& results, int openNum, int closeNum) {
		if (2 * n == res.size()) {
			if (isVaild(res)) {
				results.emplace_back(res);
			}

			return;
		}

		// 统计左右括号数量剪枝
		if (openNum < n) {
			res += "(";
			backTrack(n, res, results);
			res.pop_back();
		}
		
		if (closeNum < n) {
			res += ")";
			backTrack(n, res, results);
			res.pop_back();
		}
	}

	// 回溯(递归+剪枝)
	vector<string> generateParenthesisHS(int n) {
		vector<string> results;
		string res;
		backTrackHS(n, res, results, 0, 0);
		return results;
	}

	// DP 时间O(n4)
	// dp[i] 表示前i组括号的所有有效组合
	// dp[i] = "(dp[p]的所有有效组合) + 【dp[q]的组合】", 其中 1 + p + q = i, 
	// p从0遍历到i - 1, q则相应从i - 1到0
	// 最后一步: (a)b
	vector<string> generateParenthesisDP(int n) {
		if (n == 0) return { "" };
		if (n == 1) return { "()" };

		vector<vector<string>> dp(n + 1);
		dp[0] = { "" };
		dp[1] = { "()" };

		for (int i = 0; i <= n; ++i) {
			for (int j = 0; j < i; ++j) {
				for (string p : dp[j]) {
					for (string q : dp[i - j - 1]) {
						string str = "(" + p + ")" + q;
						dp[i].emplace_back(str);
					}
				}
			}
		}

		return dp[n];
	}
};

class Solution29 {
public:
	int divide(int dividend, int divisor) {

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
	// vector<int> nums1 = { 3, 8, 9, 10 };
	// vector<int> nums2 = { 2, 4, 6, 12, 18, 20 };
	// Solution4 s;
	// cout << s.findMedianSortedArrays(nums1, nums2) << endl;

	// 5. 最长回文子串
	// Solution5 s;
	// cout << s.longestPalindrome("ababsdfdbabdfasdfasdfasd") << endl;
	
	// 7. 整数反转
	// Solution7 s;
	// cout << s.reverse(10012321312312312312) << endl;

	// 8. 字符串转为正数
	// Solution8 s;
	// cout << s.myAtoi("1232132321") << endl;

	// 10. 正则表达式匹配
	// Solution10 s;
	// cout << s.isMatch("adad", "ada*") << endl;

	// 11. 盛水最多的容器
	// Solution11 s;
	// vector<int> heights = {1, 8, 6, 2, 5, 4, 8, 3, 7};
	// cout << s.maxArea(heights) << endl;

	// 15. 三数之和 为0的所有三元组. 注意:同一个元素不能使用多次
	// Solution15 s;
	// vector<int> nums = { -1, 0, 1, 2, -1, -4 };
	// vector<vector<int>> res = s.threeSums(nums);
	// for (auto item : res) {
	// 	cout << item[0] << " " << item[1] << " " << item[2] << endl;
	// }

	// 17. 电话号码的字母组合
	// Solution17 s;
	// vector<string> res = s.letterCombinations("234");
	// for (auto item : res) {
	// 	cout << item << " ";
	// }
	// cout << endl;
	
	// 19. 删除链表的倒数第N个节点
	// 22. 括号生成
	// Solution22 s;
	// vector<string> res = s.generateParenthesisHS(3);
	// for (auto item : res) {
	// 	cout << item << " ";
	// }
	// cout << endl;

	// 23. 合并k个有序链表
	// 思路: ①堆
	//		 ②自下而上两两合并
	//       ③归并排序思想
	
	// 29. 两数相除

	return 0;
}
#endif