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
	// 思路：双序列型动规
	// 
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
	// 1. 暴力 O(n)
	// 2. 指数递增步长 O(logN)
	int divide(int dividend, int divisor/*被除数*/) {
		if (dividend == 0) {
			return 0;
		}

		if (dividend == 1) {
			return dividend;
		}

		if (dividend == -1) {
			if (dividend > INT_MIN) return -dividend;
			return INT_MAX;
		}

		long a = dividend;
		long b = divisor;
		int sign = 1;
		if ((a > 0 && b < 0) || (a < 0 && b > 0)) {
			sign = -1;
		}

		a = a > 0 ? a : -a;
		b = b > 0 ? b : -b;

		long res = div(a, b);
		if (sign > 0) return res > INT_MAX ? INT_MAX : res;
		return -res;
	}

	// 类二分
	int div(long a, long b) {
		if (a < b) {
			return 0;
		}

		long count = 1;
		long tb = b;
		while ((tb + tb) <= a) {
			count = count + count;
			tb = tb + tb;
		}

		return count + div(a - tb, b);
	}
};

class Solution34 {
public:
	// 二分查找特殊写法
	int getIndex(vector<int>& nums, int target, bool isLeft) {
		int start = 0;
		int end = nums.size() - 1;

		while (start + 1 < end) {
			int mid = start + (end - start) / 2;
			if (nums[mid] == target) { // 重点在==这块, end = mid表示第一个, start = mid表示最后一个.
				if (isLeft) {
					end = mid;
				}
				else {
					start = mid;
				}
			}
			else if (nums[mid] > target) {
				end = mid;
			}
			else {
				start = mid;
			}
		}

		if (nums[start] == target) {
			return start;
		}

		if (nums[end] == target) {
			return end;
		}

		return -1;
	}

	vector<int> searchRanges(vector<int>& nums, int target) {
		return { getIndex(nums, target, true), getIndex(nums, target, false) };
	}
};

class Solution36 {
public:
	bool isVaildSuduku(vector<vector<string>>& board) {
		if (board.empty() || board[0].empty()) {
			return true;
		}

		int rows[9][9];
		int columns[9][9];
		int subboxes[3][3][9];

		memset(rows, 0, sizeof(rows));
		memset(columns, 0, sizeof(columns));
		memset(subboxes, 0, sizeof(subboxes));

		for (int i = 0; i < board.size(); ++i) {
			for (int j = 0; j < board[0].size(); ++j) {
				char c = board[i][j][0];
				if (c != '.') {
					int index = c - '0' - 1;
					rows[i][index]++;
					columns[i][index]++;
					subboxes[i / 3][j / 3][index]++;
					if (rows[i][index] > 1 || columns[j][index] > 1 || subboxes[i / 3][j / 3][index] > 1) {
						return false;
					}
				}
			}
		}

		return true;
	}
};

#include <string>

class Solution38 {
public:
	string countAndSay(int n) {
		string prev = "1";
		for (int i = 2; i <= n; ++i) {
			string curr = "";
			int start = 0;
			int pos = 0;

			while (pos < prev.size()) {
				while (pos < prev.size() && prev[pos] == prev[start]) {
					pos++;
				}

				curr += std::to_string(pos - start) + prev[start];
				cout << curr << endl;
				start = pos;
			}

			prev = curr;
		}

		return prev;
	}
};

class Solution41 {
public:
	int firstMissingPosition(vector<int>& nums) {
		// self 实现
		// 思路: 遍历 找到大于0的最小值 找到大于0的最大值
		/*int minV = INT_MAX;
		int maxV = INT_MIN;
		for (int i = 0; i < nums.size(); ++i) {
			if (nums[i] > 0) {
				minV = min(minV, nums[i]);
				maxV = max(maxV, nums[i]);
			}
		}

		if (minV == INT_MAX || maxV == INT_MIN) {
			return -1;
		}

		return minV - 1;
		*/

		// 3 4 -1 1
		// 1. 对小于0的元素进行n + 1赋值操作
		int n = nums.size();
		for (int& num : nums) {
			if (num <= 0) {
				num = n + 1;
			}
		}

		// 3 4 5 1
		// 2. 对于元素值小于len的以此元素的值为下标, 对此下标-1的元素(下标从0开始)前面加上负号
		for (int i = 0; i < n; ++i) {
			int num = abs(nums[i]);
			if (num <= n) {
				nums[num - 1] = -abs(nums[num - 1]);
			}
		}

		// -3 4 -5 -1
		// 3. 顺序遍历数组找到第一个大于0的元素, 返回此元素的下标值+1
		for (int i = 0; i < n; ++i) {
			if (nums[i] > 0) {
				return i + 1;
			}
		}

		// 4. 如果没有, 则返回 n + 1
		return n + 1;
	}

	// 总结:领悟其精髓------>>>>>>>目标元素一定在0 ~ n + 1范围内
};

class Solution44 {
public:
	// 思路：双序列型动态规划
	bool isMatch(string s, string p) {
		int m = s.size();
		int n = p.size();

		int i, j;
		vector<vector<bool>> f(m + 1, vector<bool>(n + 1));

		for (i = 0; i <= m; ++i) {
			for (j = 0; j <= n; ++j) {
				if (i == 0 && j == 0) {
					f[i][j] = true;
					continue;
				}

				if (j == 0) {
					f[i][j] = false;
					continue;
				}

				f[i][j] = false;
				if (p[j - 1] != '*') {
					if (i > 0 && (s[i - 1] == p[j - 1] || p[j - 1] != '?')) {
						f[i][j] = f[i][j] || f[i - 1][j - 1];
					}
				}
				else {
					f[i][j] = f[i][j] || f[i][j - 1];
					if (i > 0) {
						f[i][j] = f[i][j] || f[i - 1][j];
					}
				}
			}
		}

		return f[m][n];
	}
};

class Solution46 {
public:
	// BFS 
	// DFS 回溯
	void backTrack(vector<int>& nums, int index, vector<int>& result, vector<vector<int>>& results) {
		if (nums.size() == result.size()) {
			results.emplace_back(result);
			return;
		}

		for (int i = 0; i < nums.size(); ++i) {
			result.emplace_back(nums[i]);
			
			backTrack(nums, index + 1, result, results);
			result.pop_back();
		}
	}

	vector<vector<int>> permute(vector<int>& nums) {
		if (nums.empty()) {
			return {};
		}

		vector<vector<int>> results;
		vector<int> result;
		backTrack(nums, 0, result, results);
		return results;
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
	// Solution29 s;
	// cout << s.divide(33, 10) << endl;

	// 33.
	// 34. 在排序数组中查找元素的第一个和最后一个位置
	// Solution34 s;
	// vector<int> nums = {3, 6, 9, 9, 9, 9, 10, 23};
	// vector<int> indexs = s.searchRanges(nums, 10);
	// cout << indexs[0] << " " << indexs[1] << endl;

	// 36. 有效的数独
	/*
	Solution36 s;
	vector<vector<string>> board = { {"5","3",".",".","7",".",".",".","."}
									,{"6",".",".","1","9","5",".",".","."}
								    ,{".","9","8",".",".",".",".","6","."}
									,{"8",".",".",".","6",".",".",".","3"}
									,{"4",".",".","8",".","3",".",".","1"}
									,{"7",".",".",".","2",".",".",".","6"}
									,{".","6",".",".",".",".","2","8","."}
									,{".",".",".","4","1","9",".",".","5"}
									,{".",".",".",".","8",".",".","7","9"} };
	cout << s.isVaildSuduku(board) << endl;
	*/

	// 38. 外观数列
	// Solution38 s;
	// cout << s.countAndSay(10) << endl;
	
	// 41. 缺失的第一个正数
	// Solution41 s;
	// vector<int> nums = {8, 9};
	// cout << s.firstMissingPosition(nums) << endl;
	
	// 42. 接雨水
	// 44. 通配符匹配
	// 相似题目: 正则表达式匹配
	// Solution44 s;
	// cout << s.isMatch("cb", "?a") << endl;

	// 46. 全排列
	Solution46 s;
	vector<int> nums = { 1, 2, 3 };
	vector<vector<int>> res = s.permute(nums);
	for (auto item : res) {
		for (auto subItem : item) {
			cout << subItem << " ";
		}

		cout << endl;
	}

	cout << endl;
	return 0;
}
#endif