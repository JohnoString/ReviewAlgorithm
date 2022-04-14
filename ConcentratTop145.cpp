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
	// 1. ����֮��
	// vector<int> nums = { 2, 8, 0, 7, 13 };
	// vector<int> res = TwoSum(nums, 9);
	// cout << res[0] << " " << res[1] << endl;

	// 2. �������
	// 3. ���ظ��ַ�����Ӵ�
	// 4. Ѱ�����������������λ��
	// ���ĵ�: ��Ҫ���㽻��С�ڵ��ڵĹ�ϵ
	vector<int> nums1 = { 3, 8, 9, 10 };
	vector<int> nums2 = { 2, 4, 6, 12, 18, 20 };
	Solution4 s;
	cout << s.findMedianSortedArrays(nums1, nums2) << endl;

	// 5. ������Ӵ�

	return 0;
}
#endif