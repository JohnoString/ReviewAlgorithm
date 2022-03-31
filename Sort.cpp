#if 0
#include <iostream>
#include <stack>
#include <vector>
using namespace std;

// 快速排序
int getPartSortIndex(vector<int>& nums, int low, int high) {
    
    int tmp = nums[low]; // TODO：随机取值
    int i = low;
    int j = high;
    /*
    while (i < j) {
        while (i < j && nums[j] >= tmp) {
            j--;
        }

        nums[i] = nums[j];

        while (i < j && nums[i] <= tmp) {
            i++;
        }

        nums[j] = nums[i];
    }

    nums[i] = tmp;
    */

    while (i <= j) { // 不加=死循环 如:[3, 2, 1, 4, 5]
        while (i <= j && nums[j] /*>=*/ > tmp) { // 加=死循环 如:[1, 1, 1, 1, 1]
            j--; 
        }

        while (i <= j && nums[i] /*<=*/ < tmp) {
            i++;
        }

        if (i <= j) {
            int tmp1 = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp1;

            i++;
            j--;
        }
    }
    /*
    if (nums[i] < tmp) {
        return i + 1;
    }
    */

    return i;
}

// 递归版本
void quickSort(vector<int>& nums, int low, int high) {
    if (low >= high) {
        return;
    }

    int index = getPartSortIndex(nums, low, high);
    quickSort(nums, low, index - 1);
    quickSort(nums, index + 1, high);
}

// 非递归版本
void quickSortNotR(vector<int>& nums, int low, int high) { 
    if (low >= high) {
        return;
    }

    stack<int> s;
    s.emplace(low);
    s.emplace(high);
    
    while (!s.empty()) {
        int right = s.top();
        s.pop();

        int left = s.top();
        s.pop();

        int index = getPartSortIndex(nums, left, right);
        if (index - 1 > left) {
            s.emplace(left);
            s.emplace(index - 1);
        }

        if (index + 1 < right) {
            s.emplace(index + 1);
            s.emplace(right);
        }
    }
}

/* 归并排序 */

// merge
void merge(vector<int>& nums, int low, int high, vector<int>& tmp) {
    int mid = (low + high) / 2;
    int leftIndex = low;
    int rightIndex = mid + 1;
    int resLeftIndex = leftIndex;

    while (leftIndex <= mid && rightIndex <= high) {
        // leftIndex 135 
        // rightIndex 246
        // resLeftIndex 123456
        if (nums[leftIndex] >= nums[rightIndex]) {
            tmp[resLeftIndex++] = nums[rightIndex++];
        }
        else {
            tmp[resLeftIndex++] = nums[leftIndex++];
        }
    }

    while (leftIndex <= mid) {
        tmp[resLeftIndex++] = nums[leftIndex++];
    }
    while (rightIndex <= high) {
        tmp[resLeftIndex++] = nums[rightIndex++];
    }

    for (int i = low; i <= high; ++i) { // 易错点 <=
        nums[i] = tmp[i];
    }
}

// 分治
void divideConquer(vector<int>& nums, int low, int high, vector<int>& tmp) {
    if (low >= high) {
        return;
    }

    // 分而治之
    divideConquer(nums, low, (high + low) / 2, tmp);
    divideConquer(nums, (high + low) / 2 + 1, high, tmp);

    // 合并有序数组
    merge(nums, low, high, tmp);
}

void mergeSort(vector<int>& nums) {
    if (nums.empty()) {
        return;
    }

    vector<int> tmp(nums.size());
    divideConquer(nums, 0, nums.size() - 1, tmp);
}

int main()
{
    vector<int> nums = { 3, 2, 1, 4, 5};

    quickSortNotR(nums, 0, nums.size() - 1);
    //mergeSort(nums);

    for (auto item : nums) {
        cout << item << " ";
    }

    cout << endl;
	return 0;
}
#endif