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
	* // ��һ������ʵ��
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
// ���� O(n3) ȡ�����ִ�  
/*
	1. �ִ��кܶ๫���Ĳ��������ظ�ɨ��
	2. �ҵ�����ΪL���滻k���Ժ�ȫ����ȵ��Ӵ�, ��û�б�Ҫ���ǳ���С��L���Ӵ�;
	   ����ҵ�����ΪL���滻k���Ժ���Ȼ����ȫ��ȵ��Ӵ�, û�б�Ҫ�ٿ��ǳ��ȴ���L���Ӵ�.
*/

// Two Pointers ʱ��:O(n) �ռ�:O(26)=O(1)
int characterReplacement(string s, int k) {
	vector<int> nums(26); // ͳ���ַ����ֵĴ���
	
	int n = s.size();
	int maxn = 0;
	int left = 0, right = 0;

	while (right < n) {
		nums[s[right] - 'A']++;
		maxn = max(maxn, nums[s[right] - 'A']); // ά��һ����ĿǰΪֹ���ֹ���Ԫ�صĴ��������ֵ

		// ע�⣺Ϊʲô��if����, ����ж�ֻ�����һ��, ��Ϊ���������left�����
		if (right - left + 1 - maxn > k) { // ĿǰΪֹ���ֵ���Ԫ�ظ��� - Ŀǰ���ֹ��Ĵ�������Ԫ�� > �ɱ任���� 
			/* 
			���ڵ���k��Ҳ���ܵ�������Ԫ����ͬ��״̬, 
			�����Ե�ǰ��leftΪ���ڵ���߽������ҵ���Ŀ
			���Ӵ�����󳤶Ȳ�����������, ��ΪҪ������
			�����ִ�. ��Ҫ��left���� 
			*/
			nums[s[left] - 'A']--; // �ǵõ�ǰԪ�س��ֵĴ���-1
			left++;
		}

		right++;
	}

	return right - left /*(right - 1) - left + 1 ��ʱright�����1*/;
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
			maxNumsOfOne = max(maxNumsOfOne, counts[nums[right]]); // ������Ҫͳ��1�ĸ���
		}

		if (right - left + 1 - maxNumsOfOne > k) {
			counts[nums[left]]--; // ע��: �����right��
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

		while (sum >= target) { // ע��:�˴�������if��, �߼������ܶ��
			//if (sum == target) { // �����Ǵ���
				minLen = min(right - left + 1, minLen);
			//}

			sum -= nums[left];
			left++;
		}

		right++;
	}

	return minLen;
}

/* �Լ�ʵ�ֿ��ǲ��ܵĵط�
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

// ע��:�������ᳬʱ �������д�ɳ�Ա����������ʽ
bool check() {
	for (auto item : eleNums) {
		if (curNums[item.first] < item.second) {
			return false;
		}
	}

	return true;
}

unordered_map<char, int> eleNums, curNums;

string minWindow(string s, string t) {
	int sLen = s.size();
	int tLen = t.size();

	if (s.empty() || t.empty() || sLen < tLen) {
		return "";
	}

	for (int i = 0; i < tLen; ++i) {
		eleNums[t[i]]++;
	}

	/* right��Ҫ��-1��ʼ
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

		right++; // ע��right++Ҫ�������, minLen = min(minLen, right - left + 1);����д������Ӱ��
	}
	*/

	/* ����ʱ��������һ��
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
				l = left;
			}

			if (eleNums.find(s[left]) != eleNums.end()) {
				curNums[s[left]]--;
			}

			left++;
		}
	}

	return l == -1 ? "" : s.substr(l, minLen);
}

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

	// lc 76
	// cout << minWindow("ab", "a");

	// lc 438


	// lc 567
	return 0;
}

#endif