#if 1
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
	for (int i = 0; i < nums.size(); ++i) {
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

// 2 �������
// 3 ���ظ��ַ�����Ӵ�
// 4 Ѱ�����������������λ��
class Solution4 {
public:
	/* 
	 * ��Ҫ˼·��Ҫ�ҵ��� k (k>1) С��Ԫ�أ���ô��ȡ pivot1 = nums1[k/2-1] �� pivot2 = nums2[k/2-1] ���бȽ�
	 * ����� "/" ��ʾ����
	 * nums1 ��С�ڵ��� pivot1 ��Ԫ���� nums1[0 .. k/2-2] ���� k/2-1 ��
	 * nums2 ��С�ڵ��� pivot2 ��Ԫ���� nums2[0 .. k/2-2] ���� k/2-1 ��
	 * ȡ pivot = min(pivot1, pivot2)������������С�ڵ��� pivot ��Ԫ�ع��Ʋ��ᳬ�� (k/2-1) + (k/2-1) <= k-2 ��
	 * ���� pivot �������Ҳֻ���ǵ� k-1 С��Ԫ��
	 * ��� pivot = pivot1����ô nums1[0 .. k/2-1] ���������ǵ� k С��Ԫ�ء�����ЩԪ��ȫ�� "ɾ��"��ʣ�µ���Ϊ�µ� nums1 ����
	 * ��� pivot = pivot2����ô nums2[0 .. k/2-1] ���������ǵ� k С��Ԫ�ء�����ЩԪ��ȫ�� "ɾ��"��ʣ�µ���Ϊ�µ� nums2 ����
	 * �������� "ɾ��" ��һЩԪ�أ���ЩԪ�ض��ȵ� k С��Ԫ��ҪС���������Ҫ�޸� k ��ֵ����ȥɾ�������ĸ���
	 */
	double getKthElement(vector<int> nums1, vector<int> nums2, int k) { // k��1��ʼ
		int m = nums1.size();
		int n = nums2.size();
		int index1 = 0, index2 = 0;

		while (true) {
			// �߽����
			if (index1 == m) {
				return nums2[index2 + k - 1];
			}

			if (index2 == n) {
				return nums1[index1 + k - 1];
			}

			if (k == 1) {
				return min(nums1[index1], nums2[index2]);
			}

			// �������
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
			// ���Ⱥ�Ϊ����
			return getKthElement(nums1, nums2, (totalLength + 1) / 2);
		}
		else {
			// ���Ⱥ�Ϊż��
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
	// ˼·��˫�����Ͷ���
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

// ľͰԭ��
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

// �ع� TwoSum
class Solution15{
public:
	vector<int> twoSum(vector<int> nums, int target) {
		if (nums.size() < 2) {
			return {};
		}

		int n = nums.size();
		
		// ����
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
		for (int i = 0; i < n; ++i) { // ע��:n����-1
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

	// 1. ����O(n3)->O(n2 * logN)
	// 2. ˫ָ���Ż�. �ܽ᣺����Ҫö�������е�����Ԫ��ʱ, ����������ŵ�һ��Ԫ�صĵ���, �ڶ���Ԫ���ǵݼ���, ��ô�Ϳ���
	// ʹ��˫ָ��ķ���, ��ö�ٵ�ʱ�临�Ӷȴ�O(N2)����O(N). Ϊʲô��O(N)��? ������Ϊ��ö�ٵĹ���ÿһ����, ��ָ��������ƶ�
	// һ��λ��, ����ָ��������ƶ����ɸ�λ��, ����������Ԫ�ظ����й�. �����������֪����һ���ƶ���λ������O(N). ��̯
	// ����, ÿ��Ҳ�����ƶ�һ��λ��. ����ʱ�临�Ӷ���O(N).
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
	// DFS ���ջ�ͼ���������������ݹ�+ѭ������
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

	// BFS ʱ�临�Ӷȸ�DFSһ��
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
		/* self ʵ��
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

		// n���ڽڵ����
		if (second == nullptr && n > 0) {
			return nullptr;
		}

		// �ҵ�firstɾ��
		while (second->next != nullptr) {
			first = first->next;
			second = second->next;
		}

		return head;
		*/
		
		// �ܽ�: dummy�ڵ��÷�, �����һ���ڵ㱻ɾ����Ҫ���⴦������
		// �Ȼ�ͼ �������
		ListNode dummy(ListNode(-1));
		dummy.next = node;

		ListNode* first = node; 
		ListNode* second = &dummy; // �ڵ�index�����1��ʼ, second���ƽڵ㿪ʼ

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
	// DFS ʱ��O(2^2n) �ռ�O(n)
	bool isVaild(string str) {
		int balance = 0;
		for (auto item : str) {
			if (item == '(') {
				balance++;
			}
			else {
				balance--;
			}

			// �������� "))(("���
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

		// ͳ����������������֦
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

	// ����(�ݹ�+��֦)
	vector<string> generateParenthesisHS(int n) {
		vector<string> results;
		string res;
		backTrackHS(n, res, results, 0, 0);
		return results;
	}

	// DP ʱ��O(n4)
	// dp[i] ��ʾǰi�����ŵ�������Ч���
	// dp[i] = "(dp[p]��������Ч���) + ��dp[q]����ϡ�", ���� 1 + p + q = i, 
	// p��0������i - 1, q����Ӧ��i - 1��0
	// ���һ��: (a)b
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
	// 1. ���� O(n)
	// 2. ָ���������� O(logN)
	int divide(int dividend, int divisor/*������*/) {
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

	// �����
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
	// ���ֲ�������д��
	int getIndex(vector<int>& nums, int target, bool isLeft) {
		int start = 0;
		int end = nums.size() - 1;

		while (start + 1 < end) {
			int mid = start + (end - start) / 2;
			if (nums[mid] == target) { // �ص���==���, end = mid��ʾ��һ��, start = mid��ʾ���һ��.
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
		// self ʵ��
		// ˼·: ���� �ҵ�����0����Сֵ �ҵ�����0�����ֵ
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
		// 1. ��С��0��Ԫ�ؽ���n + 1��ֵ����
		int n = nums.size();
		for (int& num : nums) {
			if (num <= 0) {
				num = n + 1;
			}
		}

		// 3 4 5 1
		// 2. ����Ԫ��ֵС��len���Դ�Ԫ�ص�ֵΪ�±�, �Դ��±�-1��Ԫ��(�±��0��ʼ)ǰ����ϸ���
		for (int i = 0; i < n; ++i) {
			int num = abs(nums[i]);
			if (num <= n) {
				nums[num - 1] = -abs(nums[num - 1]);
			}
		}

		// -3 4 -5 -1
		// 3. ˳����������ҵ���һ������0��Ԫ��, ���ش�Ԫ�ص��±�ֵ+1
		for (int i = 0; i < n; ++i) {
			if (nums[i] > 0) {
				return i + 1;
			}
		}

		// 4. ���û��, �򷵻� n + 1
		return n + 1;
	}

	// �ܽ�:�����侫��------>>>>>>>Ŀ��Ԫ��һ����0 ~ n + 1��Χ��
};

class Solution44 {
public:
	// ˼·��˫�����Ͷ�̬�滮
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
	// DFS ����
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
	// 1. ����֮��
	// vector<int> nums = { 2, 8, 0, 7, 13 };
	// vector<int> res = TwoSum(nums, 9);
	// cout << res[0] << " " << res[1] << endl;

	// 2. �������
	// 3. ���ظ��ַ�����Ӵ�
	// 4. Ѱ�����������������λ��
	// ���ĵ�: ��Ҫ���㽻��С�ڵ��ڵĹ�ϵ
	// vector<int> nums1 = { 3, 8, 9, 10 };
	// vector<int> nums2 = { 2, 4, 6, 12, 18, 20 };
	// Solution4 s;
	// cout << s.findMedianSortedArrays(nums1, nums2) << endl;

	// 5. ������Ӵ�
	// Solution5 s;
	// cout << s.longestPalindrome("ababsdfdbabdfasdfasdfasd") << endl;
	
	// 7. ������ת
	// Solution7 s;
	// cout << s.reverse(10012321312312312312) << endl;

	// 8. �ַ���תΪ����
	// Solution8 s;
	// cout << s.myAtoi("1232132321") << endl;

	// 10. ������ʽƥ��
	// Solution10 s;
	// cout << s.isMatch("adad", "ada*") << endl;

	// 11. ʢˮ��������
	// Solution11 s;
	// vector<int> heights = {1, 8, 6, 2, 5, 4, 8, 3, 7};
	// cout << s.maxArea(heights) << endl;

	// 15. ����֮�� Ϊ0��������Ԫ��. ע��:ͬһ��Ԫ�ز���ʹ�ö��
	// Solution15 s;
	// vector<int> nums = { -1, 0, 1, 2, -1, -4 };
	// vector<vector<int>> res = s.threeSums(nums);
	// for (auto item : res) {
	// 	cout << item[0] << " " << item[1] << " " << item[2] << endl;
	// }

	// 17. �绰�������ĸ���
	// Solution17 s;
	// vector<string> res = s.letterCombinations("234");
	// for (auto item : res) {
	// 	cout << item << " ";
	// }
	// cout << endl;
	
	// 19. ɾ������ĵ�����N���ڵ�
	// 22. ��������
	// Solution22 s;
	// vector<string> res = s.generateParenthesisHS(3);
	// for (auto item : res) {
	// 	cout << item << " ";
	// }
	// cout << endl;

	// 23. �ϲ�k����������
	// ˼·: �ٶ�
	//		 �����¶��������ϲ�
	//       �۹鲢����˼��
	
	// 29. �������
	// Solution29 s;
	// cout << s.divide(33, 10) << endl;

	// 33.
	// 34. �����������в���Ԫ�صĵ�һ�������һ��λ��
	// Solution34 s;
	// vector<int> nums = {3, 6, 9, 9, 9, 9, 10, 23};
	// vector<int> indexs = s.searchRanges(nums, 10);
	// cout << indexs[0] << " " << indexs[1] << endl;

	// 36. ��Ч������
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

	// 38. �������
	// Solution38 s;
	// cout << s.countAndSay(10) << endl;
	
	// 41. ȱʧ�ĵ�һ������
	// Solution41 s;
	// vector<int> nums = {8, 9};
	// cout << s.firstMissingPosition(nums) << endl;
	
	// 42. ����ˮ
	// 44. ͨ���ƥ��
	// ������Ŀ: ������ʽƥ��
	// Solution44 s;
	// cout << s.isMatch("cb", "?a") << endl;

	// 46. ȫ����
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