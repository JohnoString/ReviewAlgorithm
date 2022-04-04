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

int equalSubString(string s, string t, int cost) {
	if (s.empty() || t.empty()) {
		return 0;
	}

	int n = s.size();
	vector<int> diff(n, 0);
	for (int i = 0; i < n; ++i) {
		diff[i] = abs(s[i] - t[i]);
	}

	int left = 0, right = 0;
	int sum = 0;
	int maxLen = 0;

	while (right < n) {
		sum += diff[right];
		if (sum > cost) {
			sum -= diff[left];
			left++;
		}

		maxLen = max(maxLen, right - left + 1);
		cout << "maxLen:" << maxLen << endl;
		right++;
	}

	return maxLen;
}

int minSubArrayLen(vector<int> nums, int target) {
	if (nums.empty() || target <= 0) {
		return 0;
	}

	int n = nums.size();
	int left = 0, right = 0;
	int sum = 0;
	int minLen = INT_MAX;

	while (right < n) {
		sum += nums[right];

		while (sum >= target) { // 注意:此处不能用if了, 逻辑进来很多次
			//if (sum == target) { // 题意是大于
				minLen = min(right - left + 1, minLen);
			//}

			sum -= nums[left];
			left++;
		}

		right++;
	}

	return minLen;
}

/* 自己实现考虑不周的地方
* string minWindow(string s, string t) {
	int sLen = s.size();
	int tLen = t.size();

	if (s.empty() || t.empty() || sLen < tLen) {
		return "";
	}

	unordered_map<int, int> eleNums;
	for (int i = 0; i < tLen; ++i) {
		eleNums[t[i] - 'A']++;
	}

	int left = 0, right = 0;
	int minLen = INT_MAX;

	while (right < sLen) {
		if (eleNums.find(s[right] - 'A') != eleNums.end() && eleNums[s[right]] > 0) {
			eleNums[s[right]]--;
			right++;
		}
		else {
			minLen = min(minLen, right - left + 1);
			left++;
		}
	}

	return s.substr(left, minLen);
}
*/

// 注意:传参数会超时 这种最好写成成员变量的想形式
/*
*/
class SolutionMinWindow {
public:
	string minWindow(string s, string t) {
		int sLen = s.size();
		int tLen = t.size();

		if (s.empty() || t.empty() || sLen < tLen) {
			return "";
		}

		for (int i = 0; i < tLen; ++i) {
			eleNums[t[i]]++;
		}

		/* right需要从-1开始
		* int left = 0, right = 0;
		int minLen = INT_MAX;
		int l = -1;

		while (right < sLen) {
			if (eleNums.find(s[right]) != eleNums.end()) {
				curNums[s[right]]++;
			}

			while (check(eleNums, curNums) && left <= right) {
				minLen = min(minLen, right - left + 1);
				l = left;
				if (eleNums.find(s[left]) != eleNums.end()) {
					curNums[s[left]]--;
				}

				left++;
			}

			right++; // 注意right++要放最后面, minLen = min(minLen, right - left + 1);这种写法会有影响
		}
		*/

		/* 报超时！用例差一个
		* int left = 0, right = -1;
		int minLen = INT_MAX;
		int l = -1;

		while (right < sLen) {
			if (eleNums.find(s[++right]) != eleNums.end()) {
				curNums[s[right]]++;
			}

			while (check(eleNums, curNums) && left <= right) {
				minLen = min(minLen, right - left + 1);
				if (minLen == right - left + 1) {
					l = left;
				}

				if (eleNums.find(s[left]) != eleNums.end()) {
					curNums[s[left]]--;
				}

				left++;
			}
		}
		*/

		int left = 0, right = -1;
		int minLen = INT_MAX;
		int l = -1;

		while (right < sLen) {
			if (eleNums.find(s[++right]) != eleNums.end()) {
				curNums[s[right]]++;
			}

			while (check() && left <= right) {
				minLen = min(minLen, right - left + 1);
				if (minLen == right - left + 1) {
					l = left; // 需要更新才保存left
				}

				if (eleNums.find(s[left]) != eleNums.end()) {
					curNums[s[left]]--;
				}

				left++;
			}
		}

		return l == -1 ? "" : s.substr(l, minLen);
	}

private:
	unordered_map<char, int> eleNums, curNums;
	bool check() {
		for (auto item : eleNums) {
			if (curNums[item.first] < item.second) {
				return false;
			}
		}

		return true;
	}
};

class SolutionFindAnagrams {
public:
	/*自己实现版本:问题很大
	* 	vector<int> findAnagrams(string s, string p) {
		vector<int> res;
		int sLen = s.size();
		int pLen = p.size();

		if (s.empty() || p.empty() || pLen > sLen) {
			return res;
		}

		for (auto item : p) {
			m_ori[item]++;
		}

		int left = 0, right = 0;
		
		while (right < sLen) {
			if (m_ori.find(s[right]) != m_ori.end()) {
				m_cur[s[right]]++;
			}
			else {
				left = right + 1;
				right = left;
			}

			if (check() && left <= right) {
				res.emplace_back(left);
				m_cur[s[left]]--;

				if (m_ori.find(s[left]) != m_ori.end()) {
					left++;
				}
			}

			right++;
		}

		return res;
	}
	*/

	/*
	* private:
	* 	bool check() {
		for (auto item : m_ori) {
			if (m_cur[item.first] < item.second) {
				return false;
			}
		}

		return true;
	}
	private:
	unordered_map<char, int> m_ori, m_cur;
	*/

	vector<int> findAnagrams(string s, string p) {
		vector<int> res;
		int sLen = s.size();
		int pLen = p.size();

		if (s.empty() || p.empty() || pLen > sLen) {
			return res;
		}

		vector<int> pNum(26);
		vector<int> sNum(26);

		// index从0开始的特殊处理
		for (int i = 0; i < pLen; ++i) {
			pNum[p[i] - 'a']++;
			sNum[s[i] - 'a']++;
		}

		if (pNum == sNum) {
			res.emplace_back(0);
		}

		// index从1开始往后推移
		for (int i = 0; i < sLen - pLen; ++i) {
			sNum[s[i] - 'a']--;
			sNum[s[i + pLen] - 'a']++;

			if (pNum == sNum) {
				res.emplace_back(i + 1);
			}
		}

		return res;
	}
};

class SolutionCheckInclusion {
public:
	bool checkInclusion(string s1, string s2) {
		int len1 = s1.size();
		int len2 = s2.size();

		if (len2 < len1) {
			return false;
		}

		vector<int> pNum(26);
		vector<int> sNum(26);

		// index从0开始的特殊处理
		for (int i = 0; i < len1; ++i) {
			pNum[s1[i] - 'a']++;
			sNum[s2[i] - 'a']++;
		}

		if (pNum == sNum) {
			return true;
		}

		// index从1开始往后推移
		for (int i = 0; i < len2 - len1; ++i) {
			sNum[s2[i] - 'a']--;
			sNum[s2[i + len1] - 'a']++;

			if (pNum == sNum) {
				return true;
			}
		}

		return false;
	}
};

int main() {
	// lc 424
	// cout << characterReplacement("ABMCDBASDFDSGSD", 2) << endl;
	
	// lc 1004
	// vector<int> nums = { 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0 };
	// cout << longestOnes(nums, 2) << endl;

	// lc 1208
	// cout << equalSubString("abcd", "asdf", 3) << endl;

	// lc 1993
	// lc 209
	//vector<int> nums = { 1, 2, 3, 4, 5 };
	//cout << minSubArrayLen(nums, 11) << endl;

	// lc 76 只有目标子串的长度不定的时候才需要双hash作为比对
	// SolutionMinWindow s;
	// cout << s.SolutionminWindow("ab", "a");

	// lc 438
	// SolutionFindAnagrams s;
	// vector<int> pos = s.findAnagrams("cbaebabacd", "abc");
	// for (auto item : pos) {
	// 	cout << item << " ";
	// }
	// cout << endl;

	// lc 567
	// SolutionCheckInclusion s;
	// cout << s.checkInclusion("ab", "eidbaooo") << endl;
	return 0;
}

#endif