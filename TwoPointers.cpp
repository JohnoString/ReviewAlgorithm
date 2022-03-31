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