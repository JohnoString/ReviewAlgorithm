#if 0
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

#include <queue>
class SolutionMaxSlidingWindow {
public:
	// 优先级队列
	vector<int> maxSlidingWindow(vector<int>& nums, int k) {
		int n = nums.size();

		priority_queue<pair<int, int>> q;
		for (int i = 0; i < k; ++i) {
			q.emplace(nums[i], i);
		}

		vector<int> ans = { q.top().first };
		for (int i = k; i < n; ++i) {
			q.emplace(nums[i], i);
			/*
				当这个最大值不在滑动窗口中时, 将其弹出队列。 如：2, 1, -2, 1
				当前堆顶元素的下标是0, 说明当前入堆的元素不是最大的. 此时需要
				将堆顶元素弹出堆. 剩下的元素中最大的就是当前窗口中最大元素.
			*/
			while (q.top().second <= i - k) {
				q.pop();
			}

			ans.emplace_back(q.top().first);
		}

		return ans;
 	}

	// 单调队列
	/*
	输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
	输出: [3,3,5,5,6,7]

	解释过程中队列中都是具体的值，方便理解，具体见代码。
	初始状态：L=R=0,队列:{}
	i=0,nums[0]=1。队列为空,直接加入。队列：{1}
	i=1,nums[1]=3。队尾值为1，3>1，弹出队尾值，加入3。队列：{3}
	i=2,nums[2]=-1。队尾值为3，-1<3，直接加入。队列：{3,-1}。此时窗口已经形成，L=0,R=2，result=[3]
	i=3,nums[3]=-3。队尾值为-1，-3<-1，直接加入。队列：{3,-1,-3}。队首3对应的下标为1，L=1,R=3，有效。result=[3,3]
	i=4,nums[4]=5。队尾值为-3，5>-3，依次弹出后加入。队列：{5}。此时L=2,R=4，有效。result=[3,3,5]
	i=5,nums[5]=3。队尾值为5，3<5，直接加入。队列：{5,3}。此时L=3,R=5，有效。result=[3,3,5,5]
	i=6,nums[6]=6。队尾值为3，6>3，依次弹出后加入。队列：{6}。此时L=4,R=6，有效。result=[3,3,5,5,6]
	i=7,nums[7]=7。队尾值为6，7>6，弹出队尾值后加入。队列：{7}。此时L=5,R=7，有效。result=[3,3,5,5,6,7]
	*/
	vector<int> maxSlidingWindow1(vector<int>& nums, int k) {
		int n = nums.size();
		deque<int> q;
		for (int i = 0; i < k; ++i) {
			while (!q.empty() && nums[i] >= nums[q.back()]) {
				q.pop_back();
			}

			q.emplace_back(i);
		}

		vector<int> ans = { nums[q.front()] };
		for (int i = k; i < n; ++i) {
			while (!q.empty() && nums[i] >= nums[q.back()]) {
				q.pop_back();
			}

			q.emplace_back(i);

			/*
				当这个最大值不在滑动窗口中时, 将其弹出队列。 如：2, 1, -2, 1
				当前队列头元素的下标是0, 说明当前入队列的元素不是最大的. 此时需要
				将队头元素弹出. 剩下的元素中最大的就是当前窗口中最大元素.
			*/
			while (q.front() <= i - k) {
				q.pop_front();
			}

			ans.emplace_back(nums[q.front()]);
		}

		return ans;
	}

	// 分块预处理_稀疏表
	vector<int> maxSlidingWindow2(vector<int>& nums, int k) {
		int n = nums.size();
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

	// lc 632 给你k个数组，找一个最小区间[a,b]，可以包含k个数组中的数字各至少一个。

	// lc 727
	/*
	Given strings S and T, find the minimum (contiguous) substring W of S, so that T is a subsequenceof W.
	If there is no such window in S that covers all characters in T, return the empty string "". If there 
	are multiple such minimum-length windows, return the one with the left-most starting index.

	Example 1:
	Input: 
	S = "abcdebdde", T = "bde"
	Output: "bcde"
	Explanation: 
	"bcde" is the answer because it occurs before "bdde" which has the same length.
	"deb" is not a smaller window because the elements of T in the window must occur in order.

	Note:
	All the strings in the input will only contain lowercase letters.
	The length of S will be in the range [1, 20000].
	The length of T will be in the range [1, 100].
	Runtime: 44 ms, faster than 73.35% of C++ online submissions for Minimum Window Subsequence.
	*/

	// lc 159
	/*
	给定一个字符串 s ，找出 至多 包含两个不同字符的最长子串 t 。
	示例 1:
	输入: “eceba”
	输出: 3
	解释: t 是 “ece”，长度为3。

	示例 2:
	输入: “ccaabbb”
	输出: 5
	解释: t 是 “aabbb”，长度为5。
	*/

	// lc 239 O(n*k)需要优化
	// 思路1：维护一个大根堆, 窗口移动的时候
	// 思路2：单调队列
	// 思路3: 分块+预处理
	/*
	* SolutionMaxSlidingWindow s;
	  vector<int> res;
	  vector<int> nums = { 1, 2, -2, 4, -2, 8, 9 };
	  res = s.maxSlidingWindow1(nums, 3);
	  for (auto it : res) {
	  	cout << it << " ";
	  }
	  cout << endl;
	*/
	
	// TODO: 总结归类
	return 0;
}

#endif