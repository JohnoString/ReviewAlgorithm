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
	Solution15 s;
	vector<int> nums = { -1, 0, 1, 2, -1, -4 };
	vector<vector<int>> res = s.threeSums(nums);
	for (auto item : res) {
		cout << item[0] << " " << item[1] << " " << item[2] << endl;
	}

	return 0;
}
#endif